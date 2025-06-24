import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(suppress=True)
import glob
import sys
from imports.utils import *
from imports.models import *
from HPC import *
from imports.adabound import AdaBound
import time
import argparse
from datetime import datetime
import multiprocessing


# ----------------------------- #
# 실행 설정 (기존 코드 그대로 유지)
# ----------------------------- #
CUDA_DEVICE = 0
N_PULSE = 32
IMAGE_WIDTH = 100
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
SAVE_DIR_NAME = "./data/"
is_CNN = 0
is_remove_model_index = 0

model_lists = get_AB_model_lists(A_init, A_final, A_step, A_range, B_init, B_final)

# ✅ 핵심 실행부를 __main__ 블록으로 감쌈
if __name__ == '__main__':
    multiprocessing.freeze_support()
    # multiprocessing.set_start_method('spawn')

    tic = time.time()

    # HPC 클래스 분류
    hpc_args = (CUDA_DEVICE, N_PULSE, IMAGE_WIDTH, TIME_RANGE_32, EXISTING_SPINS, A_init, A_final,
            A_step, A_range, B_init, B_final, noise_scale, SAVE_DIR_NAME, model_lists,
            target_side_distance, is_CNN, is_remove_model_index)

    hpc_model = HPC_Model(*hpc_args)
    total_A_lists, total_raw_pred_list, total_deno_pred_list = hpc_model.binary_classification_train()
    hpc_model.close_pool()

    predicted_periods = return_filtered_A_lists_wrt_pred(np.array(total_deno_pred_list[1,:]), np.array(total_A_lists), 0.8)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    np.save(f'./data/results/predicted_periods/predicted_periods_{timestamp}.npy', np.array(predicted_periods, dtype=object))

    # predicted_periods = np.load('./data/results/predicted_periods/predicted_periods_20250619-171443.npy', allow_pickle=True)

    print("predicted periods:", predicted_periods)


    # 스핀 개수 예측
    regression_args = (CUDA_DEVICE, N_PULSE, IMAGE_WIDTH, TIME_RANGE_32, TIME_RANGE_256, EXISTING_SPINS, A_init, A_final,
                       A_step, A_range, B_init, B_final, zero_scale, noise_scale, SAVE_DIR_NAME, model_lists, target_side_distance, is_CNN)

    regression_model = Regression_Model(*regression_args)

    spin_nums = regression_model.estimate_the_number_of_spins(predicted_periods)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    np.save(f'./data/results/spin_nums/spin_nums_{timestamp}.npy', np.array(spin_nums, dtype=object))

    print("spin_nums:", spin_nums)


    # 데이터 전처리
    A_lists = return_the_number_of_spins(predicted_periods, spin_nums)
    regression_results = regression_model.estimate_specific_AB_values(np.array(A_lists, dtype=object))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    np.save(f'./data/results/regression_results/regression_results_{timestamp}.npy', np.array(regression_results, dtype=object))

    regression_model.close_pool()

    print("regression_results:", regression_results)

    print('✅ 예측 완료. 결과가 저장되었습니다.')
    print('✅ 총 실행 시간:', time.time() - tic)
