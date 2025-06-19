import numpy as np
import os

# ABlists 불러오기
train_ABlists_dir = 'data/ABlists/'
valid_ABlists_dir = 'data/ABlists/'

n_cases_train = 128
n_cases_valid = 32

# 함수: ABlists로부터 라벨 생성
def generate_labels_from_ABlists(ablist_dir, prefix, n_cases):
    labels = []
    for i in range(n_cases):
        path = os.path.join(ablist_dir, f'{prefix}_{i}.npy')
        ablist = np.load(path)
        spin_counts = np.array([ab.shape[0] for ab in ablist])
        labels.extend(spin_counts)
    return np.array(labels)

# 생성
print("Generating training labels...")
train_labels = generate_labels_from_ABlists(train_ABlists_dir, 'ABlists_train', n_cases_train)

print("Generating validation labels...")
valid_labels = generate_labels_from_ABlists(valid_ABlists_dir, 'ABlists_valid', n_cases_valid)

# 클래스 정규화 (예: 최소값 기준으로 정수 라벨 만들기)
unique_classes = sorted(np.unique(train_labels))
class_mapping = {k: i for i, k in enumerate(unique_classes)}
train_class_labels = np.array([class_mapping[k] for k in train_labels])
valid_class_labels = np.array([class_mapping[k] for k in valid_labels])

# 저장
np.save('./data/denoised_px/Y_train_class.npy', train_class_labels)
np.save('./data/denoised_px/Y_valid_class.npy', valid_class_labels)

print("✅ 라벨 저장 완료:")
print("- ./data/denoised_px/Y_train_class.npy")
print("- ./data/denoised_px/Y_valid_class.npy")
print(f"클래스 수: {len(class_mapping)} → {class_mapping}")
