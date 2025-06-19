import numpy as np
import torch
from imports.models import Denoise_Model

# 경로 설정
X_train_path = 'data/noisy_px/X_train.npy'
X_valid_path = 'data/noisy_px/X_valid.npy'
denoised_train_path = 'data/denoised_px/denoised_train.npy'
denoised_valid_path = 'data/denoised_px/denoised_valid.npy'
denoise_model_path = 'data/models/denoising_model_2.pt'

# 하이퍼파라미터
BATCH_SIZE = 128

# 모델 로딩
model = Denoise_Model().cuda()
model.load_state_dict(torch.load(denoise_model_path))
model.eval()

def denoise_batchwise(X_array, batch_size=128):
    X_array = np.expand_dims(X_array, axis=1) if X_array.ndim == 2 else X_array
    X_array = np.expand_dims(X_array, axis=1) if X_array.shape[1] != 2 else X_array
    denoised = []

    for i in range(0, len(X_array), batch_size):
        x = torch.Tensor(X_array[i:i+batch_size]).cuda()
        with torch.no_grad():
            out = model(x)
        denoised.append(out.cpu().numpy())

    return np.concatenate(denoised, axis=0)

# 데이터 불러오기
X_train = np.load(X_train_path)
X_valid = np.load(X_valid_path)

# Denoising
print("Denoising X_train...")
denoised_train = denoise_batchwise(X_train, BATCH_SIZE)
print("Denoising X_valid...")
denoised_valid = denoise_batchwise(X_valid, BATCH_SIZE)

# 저장 전: reshape (2, 512) → 1024
denoised_train_flat = denoised_train.reshape(denoised_train.shape[0], -1)
denoised_valid_flat = denoised_valid.reshape(denoised_valid.shape[0], -1)

# 저장
np.save(denoised_train_path, denoised_train_flat)
np.save(denoised_valid_path, denoised_valid_flat)

print("✅ Denoised data saved:")
print(f"- {denoised_train_path}")
print(f"- {denoised_valid_path}")
