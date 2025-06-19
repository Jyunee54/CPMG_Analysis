import numpy as np
import pickle
from HPC import *
from imports.utils import *

# ----------------------------- #
# 실행 설정 (기존 코드 그대로 유지)
# ----------------------------- #
CUDA_DEVICE = 0
N_PULSE = 32
IMAGE_WIDTH = 10
TIME_RANGE_32  = 7000
TIME_RANGE_256  = 0
EXISTING_SPINS = 0
EVALUATION_ALL = 0

target_side_distance = 3000
A_init  = 10000
A_final = 20000
A_step  = 200
A_range = 200
B_init  = 20000
B_final = 80000
noise_scale = 0.5
zero_scale = 0.05
SAVE_DIR_NAME = "./data/results/"
is_CNN = 0
is_remove_model_index = 0

model_lists = get_AB_model_lists(A_init, A_final, A_step, A_range, B_init, B_final)

# 필요한 설정
predicted_periods = np.load('./data/results/predicted_periods.npy', allow_pickle=True)

regression_args = (CUDA_DEVICE, N_PULSE, IMAGE_WIDTH, TIME_RANGE_32, TIME_RANGE_256,
                   EXISTING_SPINS, A_init, A_final, A_step, A_range,
                   B_init, B_final, zero_scale, noise_scale, SAVE_DIR_NAME,
                   model_lists, target_side_distance, is_CNN)

regression_model = Regression_Model(*regression_args)

# 그룹 5개 단위로 쪼개기
def split_grouped_periods(periods, group_size=5):
    result = []
    for group in periods:
        for i in range(0, len(group), group_size):
            result.append(group[i:i + group_size])
    return result

split_predicted_periods = split_grouped_periods(predicted_periods, group_size=5)

# 회귀만 실행
regression_results_all = []

for i, subgroup in enumerate(split_predicted_periods):
    print(f"[INFO] Running regression on subgroup {i + 1}: A {subgroup[0]} to {subgroup[-1]}")
    try:
        regression_results = regression_model.estimate_specific_AB_values([subgroup])
        if regression_results:
            regression_results_all.extend(regression_results)
        else:
            print(f"[WARNING] Subgroup {i + 1} returned no results.")
    except Exception as e:
        print(f"[ERROR] Subgroup {i + 1} failed with error: {e}")


print('✅ 회귀 완료. 전체 결과 저장됨.')
print(regression_results_all)