import torch
import torch.nn as nn
import numpy as np
import os
from collections import OrderedDict
from tqdm import tqdm
import cv2
from retinaface import RetinaFace
import torch.nn.functional as F
import sys

# --- 1. 설정 (Configuration) ---
CONFIG = {
    "rppg_model_path": "weights/UBFC/MA-UBFC_deepphys.pth",
    "bp_model_path": "weights/CNN_LSTM/bestCNNLSTM_augmented_loss.pt",
    "img_size": 72,
    "time_length": 180,
}

# --- 2. 필요한 모든 모델 클래스 정의 ---

# --- rPPG 추출 모델 (DeepPhys) ---
class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Conv2d(in_channels, 1, kernel_size=1, padding=0)
    def forward(self, x):
        mask = torch.sigmoid(self.attention(x)); B, _, H, W = x.shape
        norm = 2 * torch.norm(mask, p=1, dim=(1,2,3)) + 1e-6
        return torch.div(mask * H * W, norm.view(B,1,1,1))

class AppearanceModel(nn.Module):
    def __init__(self, in_c=3, out_c=32, k=3):
        super().__init__()
        self.a_conv1 = nn.Conv2d(in_c, out_c, k, padding=1)
        self.a_conv2 = nn.Conv2d(out_c, out_c, k)
        self.attention_mask1 = AttentionBlock(out_c)
        self.a_avg1 = nn.AvgPool2d(2); self.a_dropout1 = nn.Dropout2d(0.25)
        self.a_conv3 = nn.Conv2d(out_c, out_c*2, k, padding=1)
        self.a_conv4 = nn.Conv2d(out_c*2, out_c*2, k)
        self.attention_mask2 = AttentionBlock(out_c*2); self.a_dropout2 = nn.Dropout2d(0.25)
    def forward(self, x):
        x = torch.tanh(self.a_conv1(x)); x = torch.tanh(self.a_conv2(x))
        m1 = self.attention_mask1(x); x = self.a_avg1(x); x = self.a_dropout1(x)
        x = torch.tanh(self.a_conv3(x)); x = torch.tanh(self.a_conv4(x))
        m2 = self.attention_mask2(x); return m1, m2

class MotionModel(nn.Module):
    def __init__(self, in_c=3, out_c=32, k=3):
        super().__init__(); self.m_conv1 = nn.Conv2d(in_c, out_c, k, padding=1)
        self.m_conv2 = nn.Conv2d(out_c, out_c, k); self.m_avg1 = nn.AvgPool2d(2)
        self.m_dropout1 = nn.Dropout2d(0.5); self.m_conv3 = nn.Conv2d(out_c, out_c*2, k, padding=1)
        self.m_conv4 = nn.Conv2d(out_c*2, out_c*2, k); self.m_avg2 = nn.AvgPool2d(2)
        self.m_dropout2 = nn.Dropout2d(0.25)
    def forward(self, x, m1, m2):
        x = torch.tanh(self.m_conv1(x)); x = torch.tanh(self.m_conv2(x))
        x = x * m1; x = self.m_avg1(x); x = self.m_dropout1(x)
        x = torch.tanh(self.m_conv3(x)); x = torch.tanh(self.m_conv4(x))
        x = x * m2; x = self.m_avg2(x); x = self.m_dropout2(x)
        return x

class DeepPhys(nn.Module):
    def __init__(self, img_size=72):
        super().__init__()
        self.appearance_model = AppearanceModel()
        self.motion_model = MotionModel()
        # 입력 차원을 동적으로 계산
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, img_size, img_size)
            dummy_mask1, dummy_mask2 = self.appearance_model(dummy_input)
            dummy_motion_out = self.motion_model(dummy_input, dummy_mask1, dummy_mask2)
            in_features = dummy_motion_out.flatten(1).shape[1]
        self.linear_model = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
    def forward(self, inputs):
        x_a, x_m = inputs[0], inputs[1]
        m1, m2 = self.appearance_model(x_a)
        motion_output = self.motion_model(x_m, m1, m2)
        return self.linear_model(motion_output.view(motion_output.shape[0], -1))

def remap_deepphys_keys(pre_trained_state_dict):
    new_state_dict = OrderedDict()
    for old_key, value in pre_trained_state_dict.items():
        new_key = old_key

        # 1. module 제거
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        # 2. appearance 관련
        if new_key.startswith("apperance_att_conv1"):
            new_key = "appearance_model.attention_mask1.attention." + new_key.split('.')[-1]
        elif new_key.startswith("apperance_att_conv2"):
            new_key = "appearance_model.attention_mask2.attention." + new_key.split('.')[-1]
        elif new_key.startswith("apperance_conv1"):
            new_key = "appearance_model.a_conv1." + new_key.split('.')[-1]
        elif new_key.startswith("apperance_conv2"):
            new_key = "appearance_model.a_conv2." + new_key.split('.')[-1]
        elif new_key.startswith("apperance_conv3"):
            new_key = "appearance_model.a_conv3." + new_key.split('.')[-1]
        elif new_key.startswith("apperance_conv4"):
            new_key = "appearance_model.a_conv4." + new_key.split('.')[-1]

        # 3. motion 관련
        elif new_key.startswith("motion_conv1"):
            new_key = "motion_model.m_conv1." + new_key.split('.')[-1]
        elif new_key.startswith("motion_conv2"):
            new_key = "motion_model.m_conv2." + new_key.split('.')[-1]
        elif new_key.startswith("motion_conv3"):
            new_key = "motion_model.m_conv3." + new_key.split('.')[-1]
        elif new_key.startswith("motion_conv4"):
            new_key = "motion_model.m_conv4." + new_key.split('.')[-1]

        # 4. 최종 선형층
        elif new_key.startswith("final_dense_1"):
            new_key = "linear_model.1." + new_key.split('.')[-1]
        elif new_key.startswith("final_dense_2"):
            new_key = "linear_model.3." + new_key.split('.')[-1]

        new_state_dict[new_key] = value
    return new_state_dict

# BP 예측 모델
class BPPredictor_CNNLSTM(nn.Module):
    def __init__(self, input_channels=1, output_size=2):
        super().__init__(); self.cnn_feature_extractor = nn.Sequential(nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=7, padding='same'),nn.ReLU(), nn.BatchNorm1d(32), nn.MaxPool1d(kernel_size=2, stride=2),nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding='same'),nn.ReLU(), nn.BatchNorm1d(64), nn.MaxPool1d(kernel_size=2, stride=2),nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding='same'),nn.ReLU(), nn.BatchNorm1d(128)); self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True, dropout=0.2); self.fc = nn.Linear(in_features=128, out_features=output_size)
    def forward(self, x):
        features = self.cnn_feature_extractor(x); features = features.permute(0, 2, 1); lstm_out, _ = self.lstm(features); last_step_out = lstm_out[:, -1, :]; predictions = self.fc(last_step_out); return predictions

# --- 3. 유틸리티 함수 ---
def load_and_remap_rppg_weights(model, model_path, device):
    pre_trained_state_dict = torch.load(model_path, map_location=device)
    remapped = remap_deepphys_keys(pre_trained_state_dict)
    model.load_state_dict(remapped)
    print("rPPG-Toolbox 가중치 매핑 및 로딩 성공!")
    return model

def extract_rppg_from_video(video_path, rppg_model, device, cfg):
    # (이전 답변의 rPPG 추출 함수 구현체와 동일)
    print(f"'{os.path.basename(video_path)}'에서 rPPG 신호 추출 중...")
    cap = cv2.VideoCapture(video_path); frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    if not frames: raise ValueError("비디오에서 프레임을 읽지 못함.")
    
    rois = []
    for frame in tqdm(frames, desc="얼굴 탐지"):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = RetinaFace.detect_faces(frame_rgb)
        if faces:
            first_face_key = list(faces.keys())[0]
            box = faces[first_face_key]['facial_area']  
            x1, y1, x2, y2 = box
            cropped = frame[y1:y2, x1:x2]
            rois.append(cv2.resize(cropped, (cfg['img_size'], cfg['img_size'])))
    if not rois: raise ValueError("비디오에서 얼굴을 찾지 못함.")
    
    rois_np = np.array(rois, dtype=np.float32) / 255.0
    motion_data = (np.diff(rois_np, axis=0) - np.mean(np.diff(rois_np, axis=0), axis=(1,2,3), keepdims=True)) / (np.std(np.diff(rois_np, axis=0), axis=(1,2,3), keepdims=True) + 1e-8)
    appearance_data = rois_np[1:]
    
    predicted_points = []
    with torch.no_grad():
        for i in tqdm(range(len(motion_data)), desc="rPPG 추론"):
            app_frame = torch.from_numpy(np.transpose(appearance_data[i], (2,0,1))).unsqueeze(0).to(device)
            mot_frame = torch.from_numpy(np.transpose(motion_data[i], (2,0,1))).unsqueeze(0).to(device)
            output_point = rppg_model((app_frame, mot_frame))
            predicted_points.append(output_point.cpu().item())
    return np.array(predicted_points)

if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # --- 인자에서 비디오 경로 받아오기 ---
    if len(sys.argv) < 2:
        print("사용법: python main.py <video_path>")
        sys.exit(1)
    
    video_path_arg = sys.argv[1]
    if not os.path.exists(video_path_arg):
        print(f"[오류] '{video_path_arg}' 경로에 파일이 존재하지 않습니다.")
        sys.exit(1)

    CONFIG["video_path"] = video_path_arg

    # 1. rPPG 추출 모델 로드 (가중치 매핑 적용)
    rppg_model = DeepPhys(img_size=CONFIG['img_size']).to(DEVICE)
    rppg_model = load_and_remap_rppg_weights(rppg_model, CONFIG["rppg_model_path"], DEVICE)
    
    # 2. BP 예측 모델 로드
    bp_model = BPPredictor_CNNLSTM().to(DEVICE)
    bp_model.load_state_dict(torch.load(CONFIG["bp_model_path"], map_location=DEVICE))
    print("혈압 예측 모델 로딩 완료.")

    # 3. 비디오에서 rPPG 신호 추출
    raw_rppg = extract_rppg_from_video(CONFIG["video_path"], rppg_model, DEVICE, CONFIG)
    
    # 4. 혈압 예측
    rppg_tensor = torch.from_numpy(raw_rppg).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        predicted_bp_tensor = bp_model(rppg_tensor)
    predicted_bp = predicted_bp_tensor.cpu().numpy()[0]
    
    # --- 결과 출력 ---
    print("\n" + "="*30); print(" 최종 혈압 예측 결과"); print("="*30)
    print(f"수축기 혈압 (SBP): {predicted_bp[0]:.2f} mmHg")
    print(f"이완기 혈압 (DBP): {predicted_bp[1]:.2f} mmHg")
    print("="*30)