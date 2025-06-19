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
import pickle
import multiprocessing


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
A_init  = 15000
A_final = 20000
A_step  = 500
A_range = 500
B_init  = 20000
B_final = 80000
noise_scale = 0.5
zero_scale = 0.05
SAVE_DIR_NAME = "./data/results/"
is_CNN = 0
is_remove_model_index = 0

model_lists = get_AB_model_lists(A_init, A_final, A_step, A_range, B_init, B_final)

# ✅ 핵심 실행부를 __main__ 블록으로 감쌈
if __name__ == '__main__':
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn')

    tic = time.time()

    if EVALUATION_ALL:
        EXISTING_SPINS = 0
        A_init, A_final, B_init, B_final, target_side_distance = -50000, 50000, 12000, 80000, 50
        args = (CUDA_DEVICE, N_PULSE, IMAGE_WIDTH, TIME_RANGE_32, EXISTING_SPINS, A_init, A_final,
                A_step, A_range, B_init, B_final, noise_scale, SAVE_DIR_NAME, model_lists, target_side_distance, is_CNN, is_remove_model_index)

        hpc_model = HPC_Model(*args)
        total_A_lists, total_raw_pred_list, total_deno_pred_list = hpc_model.binary_classification_train()
        predicted_periods = return_filtered_A_lists_wrt_pred(total_deno_pred_list[1,:])

        regression_args = (CUDA_DEVICE, N_PULSE, IMAGE_WIDTH, TIME_RANGE_32, TIME_RANGE_256, EXISTING_SPINS, A_init, A_final,
                            A_step, A_range, B_init, B_final, noise_scale, SAVE_DIR_NAME, model_lists, target_side_distance, is_CNN)
        regression_model = Regression_Model(*regression_args)
        regression_results = regression_model.estimate_specific_AB_values(predicted_periods)

        # B 값이 15000 이상인 하이퍼핀 파라미터만 필터링해서 저장
        filtered_results = [res for res in regression_results if res[1] > 15000]  # res = [A, B] 형태라고 가정
        filtered_results = np.array(filtered_results)

        save_path = './data/predicted_results_N32_B15000above.npy'
        np.save(save_path, filtered_results)
        print(f"✅ B > 15000인 결과를 {save_path} 에 저장했습니다.")

        # total_raw_pred_list, total_deno_pred_list 이 결과를 가지고 개수를 파악.
        # regression_model = Regression_Model(*args)
        # regression_model.estimate_the_number_of_spins(A_lists)
        # regression_model.estimate_specific_AB_values(A_lists_with_the_number_of_spins)
        # 여기서 얻은 것 중에, B=15000보다 큰 리스트를 저장해놓음. --> 아래 경로로.
        # np.load('./data/predicted_results_N32_B15000above.npy')

        EXISTING_SPINS = 1
        B_init, B_final = 6000, 12000
        model_lists = get_AB_model_lists(A_init, A_final, A_step, A_range, B_init, B_final)
        args = (CUDA_DEVICE, N_PULSE, IMAGE_WIDTH, TIME_RANGE_32, EXISTING_SPINS, A_init, A_final,
                A_step, A_range, B_init, B_final, noise_scale, SAVE_DIR_NAME, model_lists, target_side_distance, is_CNN, is_remove_model_index)
        total_A_lists, total_raw_pred_list, total_deno_pred_list = hpc_model.binary_classification_train()
        # total_raw_pred_list, total_deno_pred_list 이결과를 가지고 개수를 파악
        # regression_model = Regression_Model(*args)
        # regression_model.estimate_the_number_of_spins(A_lists)
        # regression_model.estimate_specific_AB_values(A_lists_with_the_number_of_spins)

        N_PULSE = 256
        EXISTING_SPINS = 1
        B_init, B_final = 2000, 15000 #### *** 단 여기는 N256에서 A가10보다 작을 때는 B_init = 1500으로 해야함!!
        model_lists = get_AB_model_lists(A_init, A_final, A_step, A_range, B_init, B_final)
        args = (CUDA_DEVICE, N_PULSE, IMAGE_WIDTH, TIME_RANGE_256, EXISTING_SPINS, A_init, A_final,
                A_step, A_range, B_init, B_final, noise_scale, SAVE_DIR_NAME, model_lists, target_side_distance, is_CNN, is_remove_model_index)
        total_A_lists, total_raw_pred_list, total_deno_pred_list = hpc_model.binary_classification_train()
        # total_raw_pred_list, total_deno_pred_list 이결과를 가지고 개수를 파악
        # regression_model = Regression_Model(*args)
        # regression_model.estimate_the_number_of_spins(total_raw_pred_list, total_deno_pred_list)
        # regression_model.estimate_specific_AB_values()
        print('Total computational time:', time.time() - tic)

        hpc_model.close_pool()
        regression_model.close_pool()

    else:
        if N_PULSE==32:
            args = (CUDA_DEVICE, N_PULSE, IMAGE_WIDTH, TIME_RANGE_32, EXISTING_SPINS, A_init, A_final,
                    A_step, A_range, B_init, B_final, noise_scale, SAVE_DIR_NAME, model_lists, target_side_distance, is_CNN, is_remove_model_index)
        elif N_PULSE==256:
            args = (CUDA_DEVICE, N_PULSE, IMAGE_WIDTH, TIME_RANGE_256, EXISTING_SPINS, A_init, A_final,
                    A_step, A_range, B_init, B_final, noise_scale, SAVE_DIR_NAME, model_lists, target_side_distance, is_CNN, is_remove_model_index)
        hpc_model = HPC_Model(*args)
        total_A_lists, total_raw_pred_list, total_deno_pred_list = hpc_model.binary_classification_train()

        print('Total computational time:', time.time() - tic)
        hpc_model.close_pool()

        predicted_periods = return_filtered_A_lists_wrt_pred(np.array(total_deno_pred_list[1,:]), np.array(total_A_lists), 0.8)
        # predicted_periods = np.load('/home/sonic/Coding/Git/Paper_git_repo/Deep_Learning_CPMG_Analysis/data/models/predicted_periods.npy')

        zero_scale = 0.
        regression_args = (CUDA_DEVICE, N_PULSE, IMAGE_WIDTH, TIME_RANGE_32, TIME_RANGE_256, EXISTING_SPINS, A_init, A_final,
                            A_step, A_range, B_init, B_final, zero_scale, noise_scale, SAVE_DIR_NAME, model_lists, target_side_distance, is_CNN)
        regression_model = Regression_Model(*regression_args)

        try:
            regression_results = regression_model.estimate_specific_AB_values(predicted_periods)
            np.save('./data/models/regression_results.npy', regression_results)
        finally:
            regression_model.close_pool()


    # args = (CUDA_DEVICE, N_PULSE, IMAGE_WIDTH, TIME_RANGE_32, EXISTING_SPINS, A_init, A_final, A_step, A_range, B_init, B_final, noise_scale, SAVE_DIR_NAME, model_lists, target_side_distance, is_CNN, is_remove_model_index)
    #
    # hpc_model = HPC_Model(*args)
    # total_A_lists, total_raw_pred_list, total_deno_pred_list = hpc_model.binary_classification_train()
    # print('Total computational time:', time.time() - tic)
    #
    # predicted_periods = return_filtered_A_lists_wrt_pred(np.array(total_deno_pred_list[1,:]), np.array(total_A_lists), 0.8)
    # predicted_periods = np.load('/home/sonic/Coding/Git/Paper_git_repo/Deep_Learning_CPMG_Analysis/data/models/predicted_periods.npy')
    # zero_scale = 0.
    # regression_args = (CUDA_DEVICE, N_PULSE, IMAGE_WIDTH, TIME_RANGE_32, TIME_RANGE_256, EXISTING_SPINS, A_init, A_final, A_step, A_range, B_init, B_final, zero_scale, noise_scale, SAVE_DIR_NAME, model_lists, target_side_distance, is_CNN)
    # regression_model = Regression_Model(*regression_args)
    # regression_results = regression_model.estimate_specific_AB_values(predicted_periods)
    # np.save('/home/sonic/Coding/Git/Paper_git_repo/Deep_Learning_CPMG_Analysis/data/models/regression_results.npy', regression_results)
