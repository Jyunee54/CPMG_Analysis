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
TIME_RANGE_32  = 15000
TIME_RANGE_256  = 0
EXISTING_SPINS = 0
EVALUATION_ALL = 0

target_side_distance = 3000
A_init  = 14500
A_final = 15500
A_step  = 200
A_range = 200
B_init  = 12000
B_final = 80000
noise_scale = 0.5
zero_scale = 0.05
SAVE_DIR_NAME = "./data/"
is_CNN = 0
is_remove_model_index = 0
filtered_results = []

model_lists = get_AB_model_lists(A_init, A_final, A_step, A_range, B_init, B_final)

# ✅ 핵심 실행부를 __main__ 블록으로 감쌈
if __name__ == '__main__':
    multiprocessing.freeze_support()

    tic = time.time()

    # HPC 클래스 분류
    hpc_args = (CUDA_DEVICE, N_PULSE, IMAGE_WIDTH, TIME_RANGE_32, EXISTING_SPINS, A_init, A_final,
            A_step, A_range, B_init, B_final, noise_scale, SAVE_DIR_NAME, model_lists,
            target_side_distance, is_CNN, is_remove_model_index, filtered_results)

    hpc_model = HPC_Model(*hpc_args)
    total_A_lists, total_raw_pred_list, total_deno_pred_list = hpc_model.binary_classification_train()
    hpc_model.close_pool()

    predicted_periods = return_filtered_A_lists_wrt_pred(np.array(total_deno_pred_list[1,:]), np.array(total_A_lists), 0.8)
    predicted_periods = np.array(predicted_periods, dtype=object)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    np.save(f'./data/results/predicted_periods/predicted_periods_{timestamp}.npy', predicted_periods)

    # predicted_periods = np.load('./data/results/predicted_periods/predicted_periods_20250624-161014.npy', allow_pickle=True)

    print("predicted periods:", predicted_periods)
    print("predicted_periods length:", len(predicted_periods))


    # 스핀 개수 예측
    regression_args = (CUDA_DEVICE, N_PULSE, IMAGE_WIDTH, TIME_RANGE_32, TIME_RANGE_256, EXISTING_SPINS,
    A_init, A_final, A_step, A_range, B_init, B_final, zero_scale, noise_scale, SAVE_DIR_NAME, model_lists,
    target_side_distance, is_CNN, filtered_results)
    #
    regression_model = Regression_Model(*regression_args)

    spin_nums = regression_model.estimate_the_number_of_spins(predicted_periods)
    spin_nums = np.array(spin_nums, dtype=object)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    np.save(f'./data/results/spin_nums/spin_nums_{timestamp}.npy', spin_nums)

    spin_nums = np.load('./data/results/spin_nums/spin_nums_20250624-161719.npy', allow_pickle=True)

    print("spin_nums length:", len(spin_nums))


    # 데이터 전처리
    A_lists = return_the_number_of_spins(predicted_periods, spin_nums)
    print("A_lists length:", len(A_lists))
    regression_results = regression_model.estimate_specific_AB_values(np.array(A_lists, dtype=object))
    regression_results = np.array(regression_results, dtype=object)
    regression_model.close_pool()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    np.save(f'./data/results/regression_results/regression_results_{timestamp}.npy', regression_results)


    print("regression_results:", regression_results)

    print('✅ 예측 완료. 결과가 저장되었습니다.')
    print('✅ 총 실행 시간:', time.time() - tic)


    # regression_results = np.load('./data/results/regression_results/regression_results_20250625-145239.npy', allow_pickle=True)

    # B 값이 15000 이상인 하이퍼핀 파라미터만 필터링해서 저장
    total_spins = return_total_spins(regression_results)
    filtered_results = [res for res in total_spins if res[1] > 15000]  # res = [A, B] 형태라고 가정
    filtered_results = np.array(filtered_results)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = f'./data/results/filtered_results/predicted_results_N32_B15000above_{timestamp}.npy'
    np.save(save_path, filtered_results)
    print(f"✅ B > 15000인 결과를 {save_path} 에 저장했습니다.")
    print(f"filtered results: {filtered_results}")
    #
    EXISTING_SPINS = 1
    B_init, B_final = 6000, 12000
    model_lists = get_AB_model_lists(A_init, A_final, A_step, A_range, B_init, B_final)

    hpc_args = (CUDA_DEVICE, N_PULSE, IMAGE_WIDTH, TIME_RANGE_32, EXISTING_SPINS, A_init, A_final,
            A_step, A_range, B_init, B_final, noise_scale, SAVE_DIR_NAME, model_lists, target_side_distance, is_CNN, is_remove_model_index, filtered_results)
    hpc_model = HPC_Model(*hpc_args)
    total_A_lists, total_raw_pred_list, total_deno_pred_list = hpc_model.binary_classification_train()
    hpc_model.close_pool()

    predicted_periods = return_filtered_A_lists_wrt_pred(np.array(total_deno_pred_list[1,:]), np.array(total_A_lists), 0.8)
    predicted_periods = np.array(predicted_periods, dtype=object)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    np.save(f'./data/results/predicted_periods/predicted_periods_{timestamp}.npy', predicted_periods)

    print("predicted periods:", predicted_periods)

    # total_raw_pred_list, total_deno_pred_list 이결과를 가지고 개수를 파악
    regression_args = (CUDA_DEVICE, N_PULSE, IMAGE_WIDTH, TIME_RANGE_32, TIME_RANGE_256, EXISTING_SPINS, A_init, A_final,
                       A_step, A_range, B_init, B_final, zero_scale, noise_scale, SAVE_DIR_NAME, model_lists, target_side_distance, is_CNN, filtered_results)
    regression_model = Regression_Model(*regression_args)

    spin_nums = regression_model.estimate_the_number_of_spins(predicted_periods)
    spin_nums = np.array(spin_nums, dtype=object)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    np.save(f'./data/results/spin_nums/spin_nums_{timestamp}.npy', spin_nums)


    print("spin_nums length:", len(spin_nums))
    A_lists = return_the_number_of_spins(predicted_periods, spin_nums)
    print("A_lists length:", len(A_lists))
    regression_results = regression_model.estimate_specific_AB_values(np.array(A_lists, dtype=object))
    regression_results = np.array(regression_results, dtype=object)
    regression_model.close_pool()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    np.save(f'./data/results/regression_results/regression_results_{timestamp}.npy', regression_results)


    print("regression_results:", regression_results)

    print('✅ 예측 완료. 결과가 저장되었습니다.')
    print('✅ 총 실행 시간:', time.time() - tic)
