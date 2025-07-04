import matplotlib.pyplot as plt
import numpy as np
import glob
import sys
from imports.utils import *
from imports.models import *
from imports.adabound import AdaBound 
import time
import argparse
np.set_printoptions(suppress=True)

from multiprocessing import Pool 
POOL_PROCESS = 23  
FILE_GEN_INDEX = 2 
# pool = Pool(processes=POOL_PROCESS)     # 수정 필요

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import shuffle
import itertools

PRE_PROCESS = False
PRE_SCALE = 1
MAGNETIC_FIELD = 403.553                        # # The external magnetic field strength. Unit: Gauss
GYRO_MAGNETIC_RATIO = 1.0705*1000               # Unit: Herts 
WL_VALUE = MAGNETIC_FIELD*GYRO_MAGNETIC_RATIO*2*np.pi 
# 첫 번재 파일 로드하여 초기 AB_lists_dic  todtjd
AB_lists_dic = np.load('./data/AB_target_dic/AB_target_dic_v4_s0.npy', allow_pickle=True).item() # a pre-calculated grouped list of nuclear spins with the same target period.
# 나머지 파일들을 반복문으로 로드하여  update
for i in range(1, 16):
    temp = np.load(f'./data/AB_target_dic/AB_target_dic_v4_s{i}.npy', allow_pickle=True).item()
    AB_lists_dic.update(temp)


# 파라미터 값
A_SIDE_NUM = 5
A_SIDE_RESOL = 600
A_TARGET_MARGIN = 25
A_SIDE_MARGIN = 300
A_FAR_SIDE_MARGIN = 5000
A_HIER_MARGIN = 25
A_HIER_MARGIN = 25
SIDE_CANDI_NUM = 3
B_TARGET_GAP = 0


class Regression_Model():

    def __init__(self, *args):
        
        self.CUDA_DEVICE, self.N_PULSE, self.IMAGE_WIDTH, self.TIME_RANGE_32, self.TIME_RANGE_256, self.EXISTING_SPINS, \
            self.A_init, self.A_final, self.A_step, self.A_range, self.B_init, self.B_final, self.zero_scale, \
                self.noise_scale, self.SAVE_DIR_NAME, self.model_lists, self.target_side_distance, self.is_CNN, self.filtered_results = args

        self.exp_data_32 = np.load('./data/exp_data/exp_data_32.npy')              # the experimental data to be evalutated
        self.exp_data_deno_32 = np.load('./data/exp_data/exp_data_32_deno.npy')    # the denoised experimental data to be evalutated
        self.exp_data_256 = np.load('./data/exp_data/exp_data_256.npy')            # the experimental data to be evalutated
        self.exp_data_deno_256 = np.load('./data/exp_data/exp_data_256_deno.npy')  # the denoised experimental data to be evalutated
        self.time_data_32 = np.load('data/time_data/time_data_32.npy')            # the time data for the experimental data to be evalutated
        self.time_data_256 = np.load('./data/time_data/time_data_256.npy')          # the time data for the experimental data to be evalutated
        self.spin_bath_32 = np.load('./data/spin_bath/spin_bath_M_value_N32.npy')   # the spin bath data for the experimental N_PULSE (it is not pre-requisite so one can just ignore this line.)
        self.spin_bath_256 = np.load('./data/spin_bath/spin_bath_M_value_N256.npy') # the spin bath data for the experimental N_PULSE (it is not pre-requisite so one can just ignore this line.)
        self.total_indices_32 = np.load('./data/total_indices/total_indices_v4_N32.npy', allow_pickle=True).item()   # pre-calculated time_indexing file by using Eqn.(4) in the maintext
        self.total_indices_256 = np.load('./data/total_indices/total_indices_v4_N256.npy', allow_pickle=True).item()

        self.pool = Pool(processes = POOL_PROCESS)

    def close_pool(self):
        self.pool.close()
        self.pool.terminate()
        self.pool.join()

    def estimate_the_number_of_spins(self, predicted_periods):

        model_lists = np.array([[A_temp_list[0], A_temp_list[-1], self.B_init, self.B_final] for A_temp_list in predicted_periods])

        total_raw_pred_list = []
        total_deno_pred_list = []
        total_A_lists = []
        total_results = []
        time_data = self.time_data_32[:self.TIME_RANGE_32] if self.N_PULSE == 32 else self.time_data_256[:self.TIME_RANGE_256]

        print("==========================================================================================")
        print('image_width:{}, TIME_RANGE_32:{}, TIME_RANGE_256:{}, B_init:{}, B_end:{}'.format(self.IMAGE_WIDTH, self.TIME_RANGE_32, self.TIME_RANGE_256, self.B_init, self.B_final))
        print("==========================================================================================")

        for model_idx, [A_first, A_end, B_first, B_end] in enumerate(model_lists):
            print("========================================================================")
            print('A_first:{}, A_end:{}, B_first:{}, B_end:{}'.format(A_first, A_end, B_first, B_end))
            print("========================================================================")
            A_num = 1
            B_num = 1
            A_resol, B_resol = 50, B_end-B_first

            print("Generating AB_idx-set...", flush=True)
            A_idx_list = np.arange(A_first, A_end+A_resol, A_num*A_resol)
            if (B_end-B_first)%B_resol==0:
                B_idx_list = np.arange(B_first, B_first+B_resol, B_num*B_resol)
            else:
                B_idx_list = np.arange(B_first, B_end, B_num*B_resol)
            print(f"A_idx_list size: {len(A_idx_list)}, B_idx_list size: {len(B_idx_list)}", flush=True)

            AB_idx_set = [[A_idx, B_idx] for A_idx, B_idx in itertools.product(A_idx_list, B_idx_list)]
            print(f"Generated AB_idx_set of length: {len(AB_idx_set)}", flush=True)

            A_side_num = A_SIDE_NUM
            A_side_resol = A_SIDE_RESOL
            A_target_margin = A_TARGET_MARGIN
            A_side_margin = A_SIDE_MARGIN
            A_far_side_margin = A_FAR_SIDE_MARGIN
            A_hier_margin = A_HIER_MARGIN
            side_candi_num = SIDE_CANDI_NUM             # the number of "how many times" to generate 'AB_side_candidate'

            A_existing_margin = 150
            B_existing_margin = 2500

            if self.N_PULSE==32:
                B_side_min, B_side_max = 6000, 70000
                B_side_gap = 5000
                B_target_gap = 1000  # distance between targets only valid when B_num >= 2.
                distance_btw_target_side = A_target_margin+A_side_margin+self.target_side_distance # the final distance between target and side = distance_btw_target_side - (A_target_margin+A_side_margin)

            elif self.N_PULSE==256:
                B_side_min, B_side_max = 1000, 20000
                B_side_gap = 0      # distance between target and side (applied for both side_same and side)
                B_target_gap = 0
                distance_btw_target_side = A_target_margin+A_side_margin+self.target_side_distance

            PRE_PROCESS, PRE_SCALE = False, 1
            if ((self.N_PULSE == 32) & (B_first<12000)):
                PRE_PROCESS = True
                PRE_SCALE = 8
                print("==================== PRE_PROCESSING:True =====================")

            class_num = A_num*B_num + 1
            cpu_num_for_multi = 10
            batch_for_multi = 128
            class_batch = cpu_num_for_multi*batch_for_multi

            spin_zero_scale = {'same':self.zero_scale, 'side':0.50, 'mid':0.05, 'far':0.05}  # setting 'same'=1.0 for hierarchical model

            epochs = 20
            valid_batch = 4096
            valid_mini_batch = 1024
            num_of_summation = 4

            args = (AB_lists_dic, self.N_PULSE, A_num, B_num, A_resol, B_resol, A_side_num, A_side_resol, B_side_min,
                        B_side_max, B_target_gap, B_side_gap, A_target_margin, A_side_margin, A_far_side_margin,
                        class_batch, class_num, spin_zero_scale, distance_btw_target_side, side_candi_num)

            for class_idx in range(num_of_summation):

                TPk_AB_candi, _, temp_hier_target_AB_candi  = gen_TPk_AB_candidates(AB_idx_set, True, *args)

                temp_hier_target_AB_candi[:,:,0] = get_marginal_arr(temp_hier_target_AB_candi[:,:,0], A_hier_margin)
                if class_idx==0:
                    total_hier_target_AB_candi = temp_hier_target_AB_candi[:]
                    target_candidates = TPk_AB_candi[1, :, 0, :]
                    side_candidates   = TPk_AB_candi[0, :, 0, :]
                    rest_candidates   = TPk_AB_candi[1, :, 1:, :]
                else:
                    total_hier_target_AB_candi = np.concatenate((total_hier_target_AB_candi, temp_hier_target_AB_candi[:]), axis=1)
                    target_candidates = np.concatenate((target_candidates, TPk_AB_candi[1, :, 0, :]), axis=0)
                    side_candidates   = np.concatenate((side_candidates, TPk_AB_candi[0, :, 0, :]), axis=0)
                    rest_candidates   = np.concatenate((rest_candidates, TPk_AB_candi[1, :, 1:, :]), axis=0)

            hier_indices = return_total_hier_index_list(A_idx_list, cut_threshold=3)
            total_class_num = hier_indices[-1][0].__len__() + 1

            total_TPk_AB_candidates = np.zeros((total_class_num, num_of_summation*TPk_AB_candi.shape[1], total_class_num+TPk_AB_candi.shape[2]+2, 2))
            indices = np.random.randint(rest_candidates.shape[0], size=(total_class_num, rest_candidates.shape[0]))
            total_TPk_AB_candidates[:, :, (total_class_num-1):-4, :] = rest_candidates[indices]
            indices = np.random.randint(side_candidates.shape[0], size=(total_class_num, side_candidates.shape[0], 2))
            total_TPk_AB_candidates[:, :, -4:-2, :] = side_candidates[indices]
            indices = np.random.randint(side_candidates.shape[0], size=(total_class_num, side_candidates.shape[0], 2))
            total_TPk_AB_candidates[:, :, -2:, :] = side_candidates[indices]

            if self.EXISTING_SPINS:
                deno_pred_N32_B15000_above = self.filtered_results
                total_TPk_AB_candidates = return_existing_spins_wrt_margins(deno_pred_N32_B15000_above, total_TPk_AB_candidates, A_existing_margin, B_existing_margin)
            final_TPk_AB_candidates = total_TPk_AB_candidates[:1]
            for class_idx, hier_index in enumerate(hier_indices):
                temp_batch = total_TPk_AB_candidates.shape[1] // len(hier_index)
                for idx2, index in enumerate(hier_index):
                    temp = np.swapaxes(total_hier_target_AB_candi[index], 0, 1)
                    if idx2 < (len(hier_index)-1):
                        temp_idx = np.random.randint(total_hier_target_AB_candi.shape[1], size=(temp_batch))
                        total_TPk_AB_candidates[class_idx+1, (idx2)*temp_batch:(idx2+1)*temp_batch, :len(index)] = temp[temp_idx]
                    else:
                        residual_batch = total_TPk_AB_candidates[class_idx+1, (idx2)*temp_batch:, :len(index)].shape[0]
                        temp_idx = np.random.randint(total_hier_target_AB_candi.shape[1], size=(residual_batch))
                        total_TPk_AB_candidates[class_idx+1, (idx2)*temp_batch:, :len(index)] = temp[temp_idx]
                final_TPk_AB_candidates = np.concatenate((final_TPk_AB_candidates, total_TPk_AB_candidates[class_idx+1:class_idx+2, :, :]), axis=0)
            median_A_idx = len(AB_idx_set) // 2

            # below is for generating N32 datasets
            print("Generating N32 data..")
            model_index_32 = get_model_index(self.total_indices_32, AB_idx_set[median_A_idx][0], time_thres_idx=self.TIME_RANGE_32, image_width=self.IMAGE_WIDTH)
            if (AB_idx_set[median_A_idx][0] > 13000) | (AB_idx_set[median_A_idx][0] < -13000):
                model_index_32, _ = return_index_without_A_idx(self.total_indices_32, model_index_32, 0, self.TIME_RANGE_32, 5)
            total_class_num = final_TPk_AB_candidates.shape[0]
            X_train_arr = np.zeros((total_class_num, final_TPk_AB_candidates.shape[1], model_index_32.shape[0], 2*self.IMAGE_WIDTH+1))
            Y_train_arr = np.zeros((total_class_num, final_TPk_AB_candidates.shape[1], total_class_num))
            mini_batch = X_train_arr.shape[1] // cpu_num_for_multi

            results = []
            for class_idx in range(total_class_num):
                Y_train_arr[class_idx, :, class_idx] = 1
                for idx1 in range(cpu_num_for_multi):
                    AB_lists_batch = final_TPk_AB_candidates[class_idx, idx1*mini_batch:(idx1+1)*mini_batch]
                    globals()["pool_{}".format(idx1)] = self.pool.apply_async(gen_M_arr_batch, [AB_lists_batch, model_index_32, self.time_data_32[:self.TIME_RANGE_32],
                                                                                            WL_VALUE, 32, PRE_PROCESS, PRE_SCALE,
                                                                                            self.noise_scale, self.spin_bath_32[:self.TIME_RANGE_32]])

                for idx2 in range(cpu_num_for_multi):
                    X_train_arr[class_idx, idx2*mini_batch:(idx2+1)*mini_batch] = globals()["pool_{}".format(idx2)].get(timeout=None)

                results.clear()
                print("class_idx:", class_idx, end=' ')

            X_train_arr = X_train_arr.reshape(-1, model_index_32.flatten().shape[0])
            Y_train_arr = Y_train_arr.reshape(-1, Y_train_arr.shape[2])

            # below is for generating N256 datasets
            if self.TIME_RANGE_256:
                print("Generating N256 data..")
                model_index_256 = get_model_index(self.total_indices_256, AB_idx_set[median_A_idx][0], time_thres_idx=self.TIME_RANGE_256, image_width=self.IMAGE_WIDTH)
                if (AB_idx_set[median_A_idx][0] > 13000) | (AB_idx_set[median_A_idx][0] < -13000):
                    model_index_256, _ = return_index_without_A_idx(self.total_indices_256, model_index_256, 0, self.TIME_RANGE_256, 5)
                total_class_num = final_TPk_AB_candidates.shape[0]
                X_train_256 = np.zeros((total_class_num, final_TPk_AB_candidates.shape[1], model_index_256.shape[0], 2*self.IMAGE_WIDTH+1))
                Y_train_256 = np.zeros((total_class_num, final_TPk_AB_candidates.shape[1], total_class_num))
                mini_batch = X_train_256.shape[1] // cpu_num_for_multi

                results = []
                for class_idx in range(total_class_num):
                    Y_train_256[class_idx, :, class_idx] = 1
                    for idx1 in range(cpu_num_for_multi):
                        AB_lists_batch = final_TPk_AB_candidates[class_idx, idx1*mini_batch:(idx1+1)*mini_batch]
                        globals()["pool_{}".format(idx1)] = self.pool.apply_async(gen_M_arr_batch, [AB_lists_batch, model_index_256, time_data[:self.TIME_RANGE_256],
                                                                                                WL_VALUE, 256, PRE_PROCESS, PRE_SCALE,
                                                                                                self.noise_scale, self.spin_bath[:self.TIME_RANGE_256]])

                    for idx2 in range(cpu_num_for_multi):
                        X_train_256[class_idx, idx2*mini_batch:(idx2+1)*mini_batch] = globals()["pool_{}".format(idx2)].get(timeout=None)

                    results.clear()
                    print("class_idx:", class_idx, end=' ')

                X_train_256 = X_train_256.reshape(-1, model_index_256.flatten().shape[0])
                Y_train_256 = Y_train_256.reshape(-1, Y_train_256.shape[2])
                X_train_arr = np.concatenate((X_train_32, X_train_256), axis=1)

            model = HPC(X_train_arr.shape[-1], Y_train_arr.shape[-1]).cuda()
            try:
                model(torch.Tensor(X_train_arr[:16]).cuda())
            except:
                raise NameError("The input shape should be revised")

            total_parameter = sum(p.numel() for p in model.parameters())
            print('total_parameter: ', total_parameter / 1000000, 'M')

            MODEL_PATH = './data/models/'
            mini_batch_list = [2048]
            learning_rate_list = [5e-6]
            op_list = [['Adabound', [30,15,7,1]]]
            criterion = nn.BCELoss().cuda()

            hyperparameter_set = [[mini_batch, learning_rate, selected_optim_name] for mini_batch, learning_rate, selected_optim_name in itertools.product(mini_batch_list, learning_rate_list, op_list)]
            print("==================== A_idx: {}, B_idx: {} ======================".format(A_first, B_first))

            total_loss, total_val_loss, total_acc, trained_model = train(MODEL_PATH, self.N_PULSE, X_train_arr, Y_train_arr, model, hyperparameter_set, criterion,
                                                                        epochs, valid_batch, valid_mini_batch, self.exp_data_32, is_pred=False, is_print_results=False,
                                                                        is_preprocess=PRE_PROCESS, PRE_SCALE=PRE_SCALE, model_index=model_index_32,
                                                                        exp_data_deno=self.exp_data_deno_32, is_regression=False)

            model.load_state_dict(torch.load(trained_model[0][0]))
            model.eval()
            print("Model loaded as evalutation mode. Model path:", trained_model[0][0])

            if self.TIME_RANGE_256:
                exp_data_test = np.hstack((self.exp_data_256[model_index_256.flatten()], self.exp_data_32[model_index_32.flatten()]))
            else:
                exp_data_test = self.exp_data_32[model_index_32.flatten()]
            exp_data_test = 1-(2*exp_data_test - 1)
            exp_data_test = exp_data_test.reshape(1, -1)
            exp_data_test = torch.Tensor(exp_data_test).cuda()

            pred = model(exp_data_test)
            pred = pred.detach().cpu().numpy()
            total_raw_pred_list.append([pred[0], total_class_num, hier_indices[-1][0].__len__()])
            print("raw", np.argmax(pred), np.max(pred), pred)

            if self.TIME_RANGE_256:
                exp_data_test = np.hstack((self.exp_data_deno_256[model_index_256.flatten()], self.exp_data_deno_32[model_index_32.flatten()]))
            else:
                exp_data_test = self.exp_data_deno_32[model_index_32.flatten()]
            exp_data_test = 1-(2*exp_data_test - 1)
            exp_data_test = exp_data_test.reshape(1, -1)
            exp_data_test = torch.Tensor(exp_data_test).cuda()

            pred = model(exp_data_test)
            pred = pred.detach().cpu().numpy()
            total_deno_pred_list.append([pred[0], total_class_num, hier_indices[-1][0].__len__()])
            print("deno", np.argmax(pred), np.max(pred), pred)
            print("Config:N_PULSE{}, B_init{}, B_final{}, w{}_batch{}, zero{}".format(self.N_PULSE, self.B_init, self.B_final, self.IMAGE_WIDTH, batch_for_multi, self.zero_scale))
            total_results.append([total_raw_pred_list[model_idx], total_deno_pred_list[model_idx], hier_indices])

        return total_results

    def estimate_specific_AB_values(self, A_lists):
        total_indices = np.load('./data/total_indices/total_indices_v4_full.npy', allow_pickle=True).item()

        total_pred_list = []
        for batch_for_multi in [128]:
            num_of_summation = 4
            noise_scale = 0.07

            cpu_num_for_multi = 10
            class_batch = cpu_num_for_multi * batch_for_multi

            A_target_margin = 25
            A_side_margin = 250
            A_final_margin = 20
            distance_btw_target_side = 500

            A_far_side_margin = 5000
            side_candi_num = 5

            A_num = 1
            B_num = 1
            class_num = A_num * B_num + 1

            A_hier_margin = 75
            A_existing_margin = 150
            B_existing_margin = 2500

            image_width = self.IMAGE_WIDTH
            TIME_RANGE = self.TIME_RANGE_32 if self.N_PULSE==32 else self.TIME_RANGE_256
            time_data = self.time_data_32[:self.TIME_RANGE_32] if self.N_PULSE==32 else self.time_data_256[:self.TIME_RANGE_256]
            B_final = self.B_final
            B_init = self.B_init
            N_PULSE = self.N_PULSE
            samespin_zero_scale = 0.5

            model_lists = np.array([[A_temp_list[0][0], A_temp_list[0][-1], B_init, B_final] for A_temp_list in A_lists])

            print("================================================================================")
            print('image_width:{}, TIME_RANGE:{}, B_init:{}, B_end:{}'.format(image_width, TIME_RANGE, B_init, B_final))
            print("================================================================================")

            for model_idx, [A_first, A_end, B_first, B_end] in enumerate(model_lists):
                hier_target_index = A_lists[model_idx][1][0]  # hier_index
                print("========================================================================")
                print('A_first:{}, A_end:{}, B_first:{}, B_end:{}, Bat_mul:{}'.format(A_first, A_end, B_first, B_end, batch_for_multi))
                print("========================================================================")
                data_gen_time = time.time()
                A_resol, B_resol = 50, B_end - B_first

                A_idx_list = np.arange(A_first, A_end + A_resol, A_num * A_resol)
                if (B_end - B_first) % B_resol == 0:
                    B_idx_list = np.arange(B_first, B_first + B_resol, B_num * B_resol)
                else:
                    B_idx_list = np.arange(B_first, B_end, B_num * B_resol)
                AB_idx_set = [[A_idx, B_idx] for A_idx, B_idx in itertools.product(A_idx_list, B_idx_list)]

                A_side_num = 8
                A_side_resol = 600
                B_side_min, B_side_max = 1000, 14000
                B_side_gap = 500
                B_target_gap = 0

                spin_zero_scale = {'same': samespin_zero_scale, 'side': 0.50, 'mid': 0.05, 'far': 0.05}  # setting 'same'=1.0 for hierarchical model

                epochs = 20
                valid_batch = 4096
                valid_mini_batch = 1024

                args = (AB_lists_dic, N_PULSE, A_num, B_num, A_resol, B_resol, A_side_num, A_side_resol, B_side_min,
                        B_side_max, B_target_gap, B_side_gap, A_target_margin, A_side_margin, A_far_side_margin,
                        class_batch, class_num, spin_zero_scale, distance_btw_target_side, side_candi_num)

                for class_idx in range(num_of_summation):
                    TPk_AB_candi, _, temp_hier_target_AB_candi = gen_TPk_AB_candidates(AB_idx_set, True, *args)
                    temp_hier_target_AB_candi[:, :, 0] = get_marginal_arr(temp_hier_target_AB_candi[:, :, 0],
                                                                          A_hier_margin)
                    if class_idx == 0:
                        total_hier_target_AB_candi = temp_hier_target_AB_candi[:]
                        target_candidates = TPk_AB_candi[0, :, 0, :]
                        side_candidates = TPk_AB_candi[1, :, 0, :]
                        rest_candidates = TPk_AB_candi[1, :, 1:, :]
                    else:
                        total_hier_target_AB_candi = np.concatenate(
                            (total_hier_target_AB_candi, temp_hier_target_AB_candi[:]), axis=1)
                        target_candidates = np.concatenate((target_candidates, TPk_AB_candi[0, :, 0, :]), axis=0)
                        side_candidates = np.concatenate((side_candidates, TPk_AB_candi[1, :, 0, :]), axis=0)
                        rest_candidates = np.concatenate((rest_candidates, TPk_AB_candi[1, :, 1:, :]), axis=0)

                hier_indices = return_total_hier_index_list(A_idx_list, cut_threshold=3)
                hier_indices = return_reduced_hier_indices(hier_indices)
                total_class_num = hier_indices[-1][0].__len__() + 1

                total_TPk_AB_candidates = np.zeros((total_class_num, num_of_summation * TPk_AB_candi.shape[1],
                                                    total_class_num + TPk_AB_candi.shape[2], 2))
                indices = np.random.randint(rest_candidates.shape[0],
                                            size=(total_class_num, rest_candidates.shape[0]))
                total_TPk_AB_candidates[:, :, (total_class_num - 1):-2, :] = rest_candidates[indices]
                indices = np.random.randint(side_candidates.shape[0],
                                            size=(total_class_num, side_candidates.shape[0], 2))
                total_TPk_AB_candidates[:, :, -2:, :] = side_candidates[indices]

                final_TPk_AB_candidates = total_TPk_AB_candidates[:1]
                for class_idx, hier_index in enumerate(hier_indices):
                    temp_batch = total_TPk_AB_candidates.shape[1] // len(hier_index)
                    for idx2, index in enumerate(hier_index):
                        temp = np.swapaxes(total_hier_target_AB_candi[index], 0, 1)
                        if idx2 < (len(hier_index) - 1):
                            temp_idx = np.random.randint(total_hier_target_AB_candi.shape[1], size=(temp_batch))
                            total_TPk_AB_candidates[class_idx + 1, (idx2) * temp_batch:(idx2 + 1) * temp_batch,
                            :len(index)] = temp[temp_idx]
                        else:
                            residual_batch = \
                            total_TPk_AB_candidates[class_idx + 1, (idx2) * temp_batch:, :len(index)].shape[0]
                            temp_idx = np.random.randint(total_hier_target_AB_candi.shape[1], size=(residual_batch))
                            total_TPk_AB_candidates[class_idx + 1, (idx2) * temp_batch:, :len(index)] = temp[
                                temp_idx]
                    final_TPk_AB_candidates = np.concatenate(
                        (final_TPk_AB_candidates, total_TPk_AB_candidates[class_idx + 1:class_idx + 2, :, :]),
                        axis=0)

                median_idx = len(AB_idx_set) // 2
                model_index = get_model_index(total_indices, AB_idx_set[median_idx][0],
                                              time_thres_idx=TIME_RANGE - 20, image_width=image_width)

                final_TPk_hier_index = len(hier_indices) + hier_target_index + 1
                final_TPk_AB_candidates = final_TPk_AB_candidates[final_TPk_hier_index]
                num_of_target_spins = hier_indices[-1][0].__len__() + hier_target_index + 1
                Y_train_arr = final_TPk_AB_candidates[:, :num_of_target_spins]
                Y_train_arr = Y_train_arr.reshape(-1, num_of_target_spins * 2)
                Y_train_arr, scale_lists = Y_train_normalization(Y_train_arr)

                X_train_arr = np.zeros(
                    (final_TPk_AB_candidates.shape[0], model_index.shape[0], 2 * image_width + 1))
                mini_batch = X_train_arr.shape[0] // cpu_num_for_multi
                for idx1 in range(cpu_num_for_multi):
                    AB_lists_batch = final_TPk_AB_candidates[idx1 * mini_batch:(idx1 + 1) * mini_batch]
                    globals()["pool_{}".format(idx1)] = self.pool.apply_async(gen_M_arr_batch,
                                                                         [AB_lists_batch, model_index,
                                                                          time_data[:TIME_RANGE],
                                                                          WL_VALUE, N_PULSE, PRE_PROCESS, PRE_SCALE,
                                                                          noise_scale])

                for idx2 in range(cpu_num_for_multi):
                    X_train_arr[idx2 * mini_batch:(idx2 + 1) * mini_batch] = globals()["pool_{}".format(idx2)].get(timeout=None)
                print("class_idx:", class_idx, end=' ')
                print("Time consumed(s): ", round(time.time() - data_gen_time, 2))
                X_train_arr = X_train_arr.reshape(-1, model_index.flatten().shape[0])

                X_train_arr, Y_train_arr = shuffle(X_train_arr, Y_train_arr)

                model = Regression_model(X_train_arr.shape[-1], Y_train_arr.shape[-1]).cuda()
                try:
                    model(torch.Tensor(X_train_arr[:16]).cuda())
                except:
                    raise NameError("The input shape should be revised")
                total_parameter = sum(p.numel() for p in model.parameters())
                print('total_parameter: ', total_parameter / 1000000, 'M')

                MODEL_DIR = '/test_regression/'
                try:
                    os.mkdir(MODEL_DIR)
                except:
                    pass
                MODEL_PATH = MODEL_DIR + 'reg_model'

                mini_batch_list = [2048]
                learning_rate_list = [5e-6]
                op_list = [['Adabound', [30, 15, 7, 1]]]
                criterion = nn.MSELoss().cuda()

                hyperparameter_set = [[mini_batch, learning_rate, selected_optim_name] for
                                      mini_batch, learning_rate, selected_optim_name in
                                      itertools.product(mini_batch_list, learning_rate_list, op_list)]
                print("==================== A_idx: {}, B_idx: {} ======================".format(A_first, B_first))

                total_loss, total_val_loss, total_acc, trained_model = train(MODEL_PATH, N_PULSE, X_train_arr,
                                                                             Y_train_arr,
                                                                             model, hyperparameter_set, criterion,
                                                                             epochs, valid_batch, valid_mini_batch,
                                                                             is_regression=True)

                model.load_state_dict(torch.load(trained_model[0][0]))
                model.eval()
                print("Model loaded as evalutation mode. Model path:", trained_model[0][0])

                test_data_test = self.exp_data_32[model_index.flatten()]
                test_data_test = 1 - (2 * test_data_test - 1)
                test_data_test = test_data_test.reshape(1, -1)
                test_data_test = torch.Tensor(test_data_test).cuda()
                pred = model(test_data_test)
                pred = pred.detach().cpu().numpy()

                reversed_pred = reverse_normalization(pred.flatten(), scale_lists)
                total_pred_list.append(reversed_pred)
                # print('Prediction Value: ', reversed_pred)
                # print('\n This predicted value is used as the initial guess value for the fine-tuning step.')
        return  total_pred_list
        
class HPC_Model():

    def __init__(self, *args):
        
        self.CUDA_DEVICE, self.N_PULSE, self.IMAGE_WIDTH, self.TIME_RANGE, self.EXISTING_SPINS, \
            self.A_init, self.A_final, self.A_step, self.A_range, self.B_init, self.B_final, \
                self.noise_scale, self.SAVE_DIR_NAME, self.model_lists, self.target_side_distance, self.is_CNN, self.is_remove_model_index, self.filtered_results = args
        self.time_range_store = self.TIME_RANGE
        self.image_width_store = self.IMAGE_WIDTH

        self.exp_data = np.load('./data/exp_data/exp_data_{}.npy'.format(self.N_PULSE)).flatten()  # the experimental data to be evalutated
        self.exp_data_deno = np.load('./data/exp_data/exp_data_{}_deno.npy'.format(self.N_PULSE))  # the denoised experimental data to be evalutated
        self.time_data = np.load('./data/time_data/time_data_{}.npy'.format(self.N_PULSE))          # the time data for the experimental data to be evalutated
        self.spin_bath = np.load('./data/spin_bath/spin_bath_M_value_N{}.npy'.format(self.N_PULSE)) # the spin bath data for the experimental N_PULSE (it is not pre-requisite so one can just ignore this line.)
        self.total_indices = np.load('./data/total_indices/total_indices_v4_N{}.npy'.format(self.N_PULSE), allow_pickle=True).item()

        self.pool = Pool(processes = POOL_PROCESS)

    def close_pool(self):
        self.pool.close()
        self.pool.terminate()
        self.pool.join()
        
    def binary_classification_train(self):

        tic = time.time() 
        if self.EXISTING_SPINS:
            deno_pred_N32_B15000_above = self.filtered_results
        total_raw_pred_list = []
        total_deno_pred_list = []
        total_A_lists = []

        for model_idx, [A_first, A_end, B_first, B_end] in enumerate(self.model_lists):

            if self.N_PULSE==256:
                if (A_first >= -10000) & (A_first <= (10000-self.A_range)):
                    self.IMAGE_WIDTH = 4
                    B_first = 1500
                    print("B_first is changed to {} Hz.".format(B_first))
                else:
                    B_first = self.B_init
                    print("B_first is changed to {} Hz.".format(B_first))

            if self.N_PULSE==256:
                if (A_first <= -12000) | (A_first >= (12000-self.A_range)):
                    self.TIME_RANGE = 3000
                    self.IMAGE_WIDTH = self.image_width_store
                    print("self.TIME_RANGE is changed to 3000 points.")
                else:
                    self.TIME_RANGE = self.time_range_store

            print("========================================================================")
            print('A_first:{}, A_end:{}, B_first:{}, B_end:{}, Time range:{}'.format(A_first, A_end, B_first, B_end, self.TIME_RANGE))
            print("========================================================================")
            A_num = 1
            B_num = 1
            A_resol, B_resol = 50, B_end-B_first+500
            A_idx_list = np.arange(A_first, A_end+A_resol, A_num*A_resol)
            if (B_end-B_first)%B_resol==0:
                B_idx_list = np.arange(B_first, B_end+B_resol, B_num*B_resol)
            else:
                B_idx_list = np.arange(B_first, B_end, B_num*B_resol)
            AB_idx_set = [[A_idx, B_idx] for A_idx, B_idx in itertools.product(A_idx_list, B_idx_list)]

            A_side_num = A_SIDE_NUM
            A_side_resol = A_SIDE_RESOL
            B_target_gap = B_TARGET_GAP
            A_target_margin = A_TARGET_MARGIN
            A_side_margin = A_SIDE_MARGIN
            A_far_side_margin = A_FAR_SIDE_MARGIN
            side_candi_num = SIDE_CANDI_NUM         # the number of "how many times" to generate 'AB_side_candidate'

            class_num = A_num*B_num + 1
            cpu_num_for_multi = 20
            batch_for_multi = 256
            class_batch = cpu_num_for_multi*batch_for_multi

            spin_zero_scale = {'same':0.5, 'side':0.20, 'mid':0.05, 'far':0.05}

            torch.cuda.set_device(device=self.CUDA_DEVICE) 
            epochs = 15
            valid_batch = 4096
            valid_mini_batch = 1024

            if self.N_PULSE==32:
                B_side_min, B_side_max = 6000, 70000
                B_side_gap = 5000
                B_target_gap = 1000
                distance_btw_target_side = A_target_margin+A_side_margin+self.target_side_distance # the final distance between target and side = distance_btw_target_side - (A_target_margin+A_side_margin)

            elif self.N_PULSE==256:
                B_side_min, B_side_max = 1500, 25000
                B_side_gap = 100  # distance between target and side (applied for both side_same and side)
                B_target_gap = 0  # distance between targets only valid when B_num >= 2.
                distance_btw_target_side = A_target_margin+A_side_margin+self.target_side_distance

            PRE_PROCESS, PRE_SCALE = False, 1
            if ((self.N_PULSE == 32) & (B_first<12000)):
                PRE_PROCESS = True
                PRE_SCALE = 8  
                print("==================== PRE_PROCESSING:True =====================")

            args = (AB_lists_dic, self.N_PULSE, A_num, B_num, A_resol, B_resol, A_side_num, A_side_resol, B_side_min,
                        B_side_max, B_target_gap, B_side_gap, A_target_margin, A_side_margin, A_far_side_margin,
                        class_batch, class_num, spin_zero_scale, distance_btw_target_side, side_candi_num) 

            TPk_AB_candi, Y_train_arr, _  = gen_TPk_AB_candidates(AB_idx_set, False, *args)
            if self.EXISTING_SPINS:
                A_existing_margin = 500
                B_existing_margin = 4000
                TPk_AB_candi = return_existing_spins_wrt_margins(deno_pred_N32_B15000_above, TPk_AB_candi, A_existing_margin, B_existing_margin)

            if (AB_idx_set[-1][0] < -15000) | (AB_idx_set[0][0] > 15000):
                is_removal = True
                temp_list0 = np.zeros(len(AB_idx_set))
                temp_list1 = np.zeros(len(AB_idx_set))
                temp_list2 = []
                for idx in range(len(AB_idx_set)):
                    model_index_temp = get_model_index(self.total_indices, AB_idx_set[idx][0], time_thres_idx=self.TIME_RANGE, image_width=self.IMAGE_WIDTH)
                    temp_list0[idx] = len(model_index_temp)
                    _, removed_index = return_index_without_A_idx(self.total_indices, model_index_temp, 0, self.TIME_RANGE, 4)
                    if (AB_idx_set[-1][0] < -15000) & (self.is_remove_model_index):
                        for [A,B] in deno_pred_N32_B15000_above:
                            print("Model index is reduced w.r.t the spin of {}".format([A,B]))
                            A_temp = return_TPk_from_AB(A, B, WL_VALUE)
                            _, removed_index_temp = return_index_without_A_idx(self.total_indices, model_index_temp, A_temp, self.TIME_RANGE, 5)
                            removed_index += removed_index_temp
                    temp_list1[idx] = len(removed_index)
                    temp_list2.append(removed_index)
                cut_idx = int(np.min(temp_list0))
                temp_idx = np.argmax(temp_list1)
                removed_index = temp_list2[temp_idx]
                selected_index = [int(k) for k in np.arange(cut_idx) if k not in removed_index] 
                X_train_arr = np.zeros((class_num, len(AB_idx_set)*class_batch, len(selected_index), 2*self.IMAGE_WIDTH+1))
            else:
                is_removal = False
                temp_list0 = np.zeros(len(AB_idx_set))
                for idx in range(len(AB_idx_set)):
                    model_index_temp = get_model_index(self.total_indices, AB_idx_set[idx][0], time_thres_idx=self.TIME_RANGE, image_width=self.IMAGE_WIDTH)
                    temp_list0[idx] = len(model_index_temp)
                cut_idx = int(np.min(temp_list0))
                selected_index = []
                X_train_arr = np.zeros((class_num, len(AB_idx_set)*class_batch, cut_idx, 2*self.IMAGE_WIDTH+1))

            for idx1, [A_idx, B_idx] in enumerate(AB_idx_set):
                model_index = get_model_index(self.total_indices, A_idx, time_thres_idx=self.TIME_RANGE, image_width=self.IMAGE_WIDTH)
                if is_removal:
                    model_index = model_index[selected_index]
                else:
                    model_index = model_index[:cut_idx]
                for class_idx in range(class_num):
                    for idx2 in range(cpu_num_for_multi):
                        AB_lists_batch = TPk_AB_candi[class_idx, idx1*class_batch+idx2*batch_for_multi:idx1*class_batch+(idx2+1)*batch_for_multi]
                        globals()["pool_{}".format(idx2)] = self.pool.apply_async(gen_M_arr_batch, [AB_lists_batch, model_index, self.time_data[:self.TIME_RANGE],
                                                                                                WL_VALUE, self.N_PULSE, PRE_PROCESS, PRE_SCALE,
                                                                                                self.noise_scale, self.spin_bath[:self.TIME_RANGE]])

                    for idx3 in range(cpu_num_for_multi):  
                        X_train_arr[class_idx, idx1*class_batch+idx3*batch_for_multi:idx1*class_batch+(idx3+1)*batch_for_multi] = globals()["pool_{}".format(idx3)].get(timeout=None)
                    print("_", end=' ') 
            
            if self.is_CNN:
                X_train_arr = X_train_arr.reshape(class_num*len(AB_idx_set)*class_batch, 1, model_index.shape[0], model_index.shape[1]) 
                Y_train_arr = Y_train_arr.reshape(class_num*len(AB_idx_set)*class_batch, class_num) 
                X_train_arr, Y_train_arr = shuffle(X_train_arr, Y_train_arr)
                model = HPC_CNN(X_train_arr[0,0].shape, Y_train_arr.shape[1]).cuda() 
            else:
                X_train_arr = X_train_arr.reshape(class_num*len(AB_idx_set)*class_batch, model_index.flatten().shape[0])
                Y_train_arr = Y_train_arr.reshape(class_num*len(AB_idx_set)*class_batch, class_num) 
                X_train_arr, Y_train_arr = shuffle(X_train_arr, Y_train_arr)
                model = HPC(X_train_arr.shape[1], Y_train_arr.shape[1]).cuda()
            
            try:
                model(torch.Tensor(X_train_arr[:5]).cuda()) 
            except:
                raise NameError("The input shape should be revised")

            total_parameter = sum(p.numel() for p in model.parameters()) 
            print('total_parameter: ', total_parameter / 1000000, 'M')

            MODEL_PATH = './data/models/'
            if not os.path.exists(MODEL_PATH): os.mkdir(MODEL_PATH)

            mini_batch_list = [1024]  
            learning_rate_list = [5e-6] 
            op_list = [['Adabound', [30,15,7,1]]] 
            criterion = nn.BCELoss().cuda()
            hyperparameter_set = [[mini_batch, learning_rate, selected_optim_name] for mini_batch, learning_rate, selected_optim_name in itertools.product(mini_batch_list, learning_rate_list, op_list)]
            print("==================== A_idx: {}, B_idx: {} ======================".format(A_first, B_first))

            total_loss, total_val_loss, total_acc, trained_model = train(MODEL_PATH, self.N_PULSE, X_train_arr, Y_train_arr, model, hyperparameter_set, criterion,
                                                                        epochs, valid_batch, valid_mini_batch, self.exp_data, is_pred=False, is_print_results=False, is_preprocess=PRE_PROCESS, PRE_SCALE=PRE_SCALE,
                                                                        model_index=model_index, exp_data_deno=self.exp_data_deno)
            min_A = np.min(np.array(AB_idx_set)[:,0])
            max_A = np.max(np.array(AB_idx_set)[:,0])

            model.load_state_dict(torch.load(trained_model[0][0])) 

            total_A_lists, total_raw_pred_list, total_deno_pred_list = HPC_prediction(model, AB_idx_set, self.total_indices, self.TIME_RANGE, self.IMAGE_WIDTH, 
                                                                                        selected_index, cut_idx, is_removal, self.exp_data, self.exp_data_deno, total_A_lists, total_raw_pred_list, 
                                                                                        total_deno_pred_list, self.is_CNN, PRE_PROCESS, PRE_SCALE, save_to_file=False)

        total_raw_pred_list  = np.array(total_raw_pred_list).T
        total_deno_pred_list = np.array(total_deno_pred_list).T

        RESULT_PATH = f'./data/results/'
        if not os.path.exists(RESULT_PATH):
            os.makedirs(RESULT_PATH)

        np.save(RESULT_PATH+'total_N{}_A_idx.npy'.format(self.N_PULSE), total_A_lists)
        np.save(RESULT_PATH+'total_N{}_raw_pred.npy'.format(self.N_PULSE), total_raw_pred_list)
        np.save(RESULT_PATH+'total_N{}_deno_pred.npy'.format(self.N_PULSE), total_deno_pred_list)

        print('================================================================')
        print('Training Completed. Parsing parameters as follows.')
        print('N:{}, A_init:{}, A_final:{}, A_range:{}, A_step:{}, B_init:{}, B_final:{}, Image Width:{}, Time range:{}, noise:{}'.format(self.N_PULSE, 
                                                            self.A_init, self.A_final, self.A_range, self.A_step, self.B_init, self.B_final, self.IMAGE_WIDTH, self.TIME_RANGE, self.noise_scale))
        print('================================================================')
        print('total_time_consumed', time.time() - tic)

        return total_A_lists, total_raw_pred_list, total_deno_pred_list