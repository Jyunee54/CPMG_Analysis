import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
from imports.utils import *
from imports.models import *
from imports.adabound import AdaBound

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import shuffle
import itertools
import copy


# Generate simulation CPMG data for the denoising model
time_resolution = 0.004
time_data = np.arange(0, 60, time_resolution)

MAGNETIC_FIELD = 403.553
GYRO_MAGNETIC_RATIO = 1.07*1000       # Unit: Herts
WL_VALUE = MAGNETIC_FIELD*GYRO_MAGNETIC_RATIO*2*np.pi
N_PULSE_32 = 32
N_PULSE_256 = 256

SAVE_MLISTS = './data/'

N_SAMPLES_TRAIN = 8192*128
N_SAMPLES_VALID = 8192
data_size = 1024  # Initially defined as 1024 (will reshape to 512*2)
GPU_INDEX = '0'


# Making gaussian slope in simulated data
# 가우시안 감쇠 곡선(디코히런스 곡선)을 계산하는 함수
def gaussian_slope(time_data, time_index, px_mean_value_at_time, M_list=None):
    m_value_at_time = (px_mean_value_at_time * 2) - 1
    Gaussian_co = -time_data[time_index] / np.log(m_value_at_time)

    slope = np.exp(-(time_data / Gaussian_co)**2)
    if M_list != None:
        M_list_slope = M_list * slope
        px_list_slope = (1 + M_list_slope) / 2
        return px_list_slope, slope
    return slope

SLOPE_INDEX = 11812
MEAN_PX_VALUE = np.linspace(0.65, 0.94, 20)
slope = {}
for idx, mean_px_value in enumerate(MEAN_PX_VALUE):
    slope[idx] = gaussian_slope(time_data[:24000], SLOPE_INDEX, mean_px_value, None)


#### Range of A & B (Hz)
A_search_min = -80000
A_search_max = 80000
B_search_min = 2000
B_search_max = 100000
A_steps = 500
B_steps = 500

n_A_samples = (A_search_max - A_search_min) // A_steps
n_B_samples = (B_search_max - B_search_min) // B_steps

TOTAL_TEST_ARRAY = np.zeros((n_A_samples * n_B_samples, 2))

B_FIXED_VALUE = B_search_min
for i in range(n_B_samples):
    test_array = np.array([np.arange(A_search_min, A_search_max, A_steps), np.full((n_A_samples,), B_FIXED_VALUE)])
    test_array = test_array.transpose()
    test_array = np.round(test_array, 2)
    TOTAL_TEST_ARRAY[i * n_A_samples:(i + 1) * n_A_samples] = test_array
    B_FIXED_VALUE += B_steps

A_candidate_max = 80000
A_candidate_min = -80000
B_candidate_max = 90000
B_candidate_min = 2000

A_small_max = 70000
A_small_min = -70000
B_small_max = 15000
B_small_min = 2000

candidate_boolen_index = ((TOTAL_TEST_ARRAY[:, 1] >= B_candidate_min) & (TOTAL_TEST_ARRAY[:, 1] <= B_candidate_max)) \
                         & ((TOTAL_TEST_ARRAY[:, 0] >= A_candidate_min) & (TOTAL_TEST_ARRAY[:, 0] <= A_candidate_max))
AB_candidate_array = TOTAL_TEST_ARRAY[candidate_boolen_index]

small_AB_boolen_index = ((TOTAL_TEST_ARRAY[:, 1] >= B_small_min) & (TOTAL_TEST_ARRAY[:, 1] <= B_small_max)) \
                        & ((TOTAL_TEST_ARRAY[:, 0] >= A_small_min) & (TOTAL_TEST_ARRAY[:, 0] <= A_small_max))
small_AB_array = TOTAL_TEST_ARRAY[small_AB_boolen_index]

AB_candidate_array *= (2 * np.pi)
small_AB_array *= (2 * np.pi)

n_of_cases_train = 128
n_of_samples = N_SAMPLES_TRAIN // n_of_cases_train



tic = time.time()
for i in range(n_of_cases_train):

    total_n_of_spin_lists = np.arange(24, 33)
    n_of_spins = np.random.choice(total_n_of_spin_lists)
    n_of_small_spins = n_of_spins // 2
    n_of_spins, n_of_small_spins

    globals()['ABlists_train_{}'.format(i)] = np.zeros((n_of_samples, n_of_spins, 2))

    for idx in range(n_of_samples):

        indices_candi = np.random.randint(len(AB_candidate_array), size=n_of_spins - n_of_small_spins)
        while len(set(indices_candi)) != (n_of_spins - n_of_small_spins):
            indices_candi = np.random.randint(len(AB_candidate_array), size=n_of_spins - n_of_small_spins)
        globals()['ABlists_train_{}'.format(i)][idx, :n_of_spins - n_of_small_spins] = AB_candidate_array[indices_candi]

        indices_candi = np.random.randint(len(small_AB_array), size=n_of_small_spins)
        while len(set(indices_candi)) != (n_of_small_spins):
            indices_candi = np.random.randint(len(small_AB_array), size=n_of_small_spins)
        globals()['ABlists_train_{}'.format(i)][idx, n_of_spins - n_of_small_spins:] = small_AB_array[indices_candi]

    # 저장: 각 훈련 데이터셋을 .npy 파일로 저장
    np.save(f'data/ABlists/ABlists_train_{i}.npy', globals()['ABlists_train_{}'.format(i)])

print("ABlists for training dataset is generated: {} s".format(round(time.time() - tic, 3)))

n_of_cases_valid = 32
n_of_samples = N_SAMPLES_VALID // n_of_cases_valid

tic = time.time()
for i in range(n_of_cases_valid):

    total_n_of_spin_lists = np.arange(24, 36)
    n_of_spins = np.random.choice(total_n_of_spin_lists)
    n_of_small_spins = n_of_spins // 2
    n_of_spins, n_of_small_spins

    globals()['ABlists_valid_{}'.format(i)] = np.zeros((n_of_samples, n_of_spins, 2))

    for idx in range(n_of_samples):

        indices_candi = np.random.randint(len(AB_candidate_array), size=n_of_spins - n_of_small_spins)
        while len(set(indices_candi)) != (n_of_spins - n_of_small_spins):
            indices_candi = np.random.randint(len(AB_candidate_array), size=n_of_spins - n_of_small_spins)
        globals()['ABlists_valid_{}'.format(i)][idx, :n_of_spins - n_of_small_spins] = AB_candidate_array[indices_candi]

        indices_candi = np.random.randint(len(small_AB_array), size=n_of_small_spins)
        while len(set(indices_candi)) != (n_of_small_spins):
            indices_candi = np.random.randint(len(small_AB_array), size=n_of_small_spins)
        globals()['ABlists_valid_{}'.format(i)][idx, n_of_spins - n_of_small_spins:] = small_AB_array[indices_candi]

    # 저장: 각 검증 데이터셋을 .npy 파일로 저장
    np.save(f'data/ABlists/ABlists_valid_{i}.npy', globals()['ABlists_valid_{}'.format(i)])

print("ABlists for validation dataset is generated: {} s".format(round(time.time() - tic, 3)))


def Px_noise_data(time_table, wL_value, AB_list, n_pulse, rand_idx, data_size, y_train_pure=False):
    noise = 0.02  # Maximum Height of Noise
    rescale = np.random.random() / 2 + 0.75
    noise *= rescale

    AB_list = np.array(AB_list)
    A = AB_list[:, 0].reshape(len(AB_list), 1)
    B = AB_list[:, 1].reshape(len(AB_list), 1)

    w_tilda = pow(pow(A + wL_value, 2) + B * B, 1 / 2)
    mz = (A + wL_value) / w_tilda
    mx = B / w_tilda

    alpha = w_tilda * time_table.reshape(1, len(time_table))
    beta = wL_value * time_table.reshape(1, len(time_table))

    phi = np.arccos(np.cos(alpha) * np.cos(beta) - mz * np.sin(alpha) * np.sin(beta))
    K1 = (1 - np.cos(alpha)) * (1 - np.cos(beta))
    K2 = 1 + np.cos(phi)
    K = pow(mx, 2) * (K1 / K2)
    M_list_temp = 1 - K * pow(np.sin(n_pulse * phi / 2), 2)
    Y_train = np.prod(M_list_temp, axis=0)

    slope_temp = np.zeros(data_size)
    if np.random.uniform() > 0.4:
        temp_idx = np.random.randint(len(slope))
        slope_temp[:data_size // 3] = slope[temp_idx][rand_idx:rand_idx + data_size // 3]
        temp_idx = np.random.randint(len(slope))
        slope_temp[data_size // 3:2 * (data_size // 3)] = slope[temp_idx][
                                                          rand_idx + data_size // 3:rand_idx + 2 * (data_size // 3)]
        temp_idx = np.random.randint(len(slope))
        slope_temp[2 * (data_size // 3):data_size] = slope[temp_idx][
                                                     rand_idx + 2 * (data_size // 3):rand_idx + data_size]

    else:
        temp_idx = np.random.randint(len(slope))
        slope_temp = slope[temp_idx][rand_idx:rand_idx + data_size]

    if y_train_pure == False:
        Y_train = (1 + slope_temp * Y_train) / 2
        X_train = Y_train + noise * (np.random.random(Y_train.shape[0]) - np.random.random(Y_train.shape[0]))
        return X_train, Y_train

    else:
        Y_train_pure = (1 + copy.deepcopy(Y_train)) / 2
        Y_train = (1 + slope_temp * Y_train) / 2
        X_train = Y_train + noise * (np.random.random(Y_train.shape[0]) - np.random.random(Y_train.shape[0]))
        return X_train, Y_train, Y_train_pure


n_of_cases = 128

X_train = np.zeros((N_SAMPLES_TRAIN, data_size))
Y_train = np.zeros((N_SAMPLES_TRAIN, data_size))
Y_train_pure = np.zeros((N_SAMPLES_TRAIN, data_size))

ABlists_train_0 = np.load('data/ABlists/ABlists_train_0.npy')
ABlists_valid_0 = np.load('data/ABlists/ABlists_valid_0.npy')
for i in range(n_of_cases):
    print(i, end=' ')
    for j in range(len(ABlists_train_0)):
        rand_idx = np.random.randint(11000)
        time_data_temp = time_data[rand_idx:rand_idx + data_size]

        X_train[i * len(ABlists_train_0) + j], \
            Y_train[i * len(ABlists_train_0) + j], \
            Y_train_pure[i * len(ABlists_train_0) + j] \
            = Px_noise_data(time_data_temp, WL_VALUE, globals()['ABlists_train_{}'.format(i)][j], N_PULSE_32, rand_idx,
                            data_size, y_train_pure=True)

print('\n')
X_valid = np.zeros((N_SAMPLES_VALID, data_size))
Y_valid = np.zeros((N_SAMPLES_VALID, data_size))
Y_valid_pure = np.zeros((N_SAMPLES_VALID, data_size))

for i in range(int(N_SAMPLES_VALID / ABlists_valid_0.shape[0])):
    print(i, end=' ')
    for j in range(len(ABlists_valid_0)):
        rand_idx = np.random.randint(11000)
        time_data_temp = time_data[rand_idx:rand_idx + data_size]

        X_valid[i * len(ABlists_valid_0) + j], \
            Y_valid[i * len(ABlists_valid_0) + j], \
            Y_valid_pure[i * len(ABlists_valid_0) + j], \
            = Px_noise_data(time_data_temp, WL_VALUE, globals()['ABlists_valid_{}'.format(i)][j], N_PULSE_32, rand_idx,
                            data_size, y_train_pure=True)


# X_train, Y_train 크기 맞추기 (2채널 구조로 변경)
X_train = np.expand_dims(X_train, axis=-2)
Y_train = np.expand_dims(Y_train, axis=-2)
X_valid = np.expand_dims(X_valid, axis=-2)
Y_valid = np.expand_dims(Y_valid, axis=-2)

X_train = torch.Tensor(X_train.reshape(X_train.shape[0], 2, -1)).cuda()
Y_train = torch.Tensor(Y_train.reshape(X_train.shape[0], 2, -1)).cuda()
X_valid = torch.Tensor(X_valid.reshape(X_valid.shape[0], 2, -1)).cuda()
Y_valid = torch.Tensor(Y_valid.reshape(Y_valid.shape[0], 2, -1)).cuda()

# 저장: 각 훈련 데이터셋을 .npy 파일로 저장
np.save('data/noisy_px/X_train.npy', X_train.cpu().numpy())
np.save('data/noisy_px/Y_train.npy', Y_train.cpu().numpy())
# np.save('data/Y_train_pure.npy', Y_train_pure)

np.save('data/noisy_px/X_valid.npy', X_valid.cpu().numpy())
np.save('data/noisy_px/Y_valid.npy', Y_valid.cpu().numpy())
# np.save('data/Y_valid_pure.npy', Y_valid_pure)