import os
import math
import h5py
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from datasets import transforms
# import transforms

def cal_hr(output : torch.Tensor, Fs : float):
    '''
    args:
        output: (1, T)
        Fs: sampling rate
    return:
        hr: heart rate
    '''
    def compute_complex_absolute_given_k(output : torch.Tensor, k : torch.Tensor, N : int):
        two_pi_n_over_N = 2 * math.pi * torch.arange(0, N, dtype=torch.float) / N
        hanning = torch.from_numpy(np.hanning(N)).type(torch.FloatTensor).view(1, -1)

        k = k.type(torch.FloatTensor)
        two_pi_n_over_N = two_pi_n_over_N
        hanning = hanning
            
        output = output.view(1, -1) * hanning
        output = output.view(1, 1, -1).type(torch.FloatTensor)
        k = k.view(1, -1, 1)
        two_pi_n_over_N = two_pi_n_over_N.view(1, 1, -1)
        complex_absolute = torch.sum(output * torch.sin(k * two_pi_n_over_N), dim=-1) ** 2 \
                           + torch.sum(output * torch.cos(k * two_pi_n_over_N), dim=-1) ** 2
        return complex_absolute
    
    output = output.view(1, -1)

    N = output.size()[1]
    bpm_range = torch.arange(40, 180, dtype=torch.float)
    unit_per_hz = Fs / N
    feasible_bpm = bpm_range / 60.0
    k = feasible_bpm / unit_per_hz
    
    # only calculate feasible PSD range [0.7, 4]Hz
    complex_absolute = compute_complex_absolute_given_k(output, k, N)
    complex_absolute = (1.0 / complex_absolute.sum()) * complex_absolute
    whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0) # max返回（values, indices）
    whole_max_idx = whole_max_idx.type(torch.float) # 功率谱密度的峰值对应频率即为心率

    return whole_max_idx + 40	# Analogous Softmax operator

# 返回T帧或者整个视频
# 每一个数据集实现自己的读视频，读帧率的方法
# T=-1表示读整个视频
class BaseDataset(Dataset):
    def __init__(self, data_dir, train='train', T=-1, w=64, h=64, aug=''):
        '''
        :param data_dir: root dir of dataset
        :param train: which part of dataset to use
        :param T: number of frames to use, -1 means use all frames
        '''
        self.data_dir = data_dir
        if 'val' in train:
            self.train = 'test'
        else:
            self.train = train
        self.T = T
        self.w = w
        self.h = h
        self.data_list = list()
        self.get_data_list()
        self.aug = aug
        self.speed_slow = 0.6
        self.speed_fast = 1.4
        self.set_augmentations()

    def get_data_list(self):
        '''
        return: self.data_list list(dict)
            for each sample in self.data_list:
                sample['location']: sample's location
                sample['start_idx']: start index of sample
                sample['video_length']: length of sample
            each os.path.join(sample['location'], 'sample.hdf5') contains:
                'video_data': (C, T, H, W)  ----- for all video
                'ecg_data': (T)             ----- for all video
        '''
        raise NotImplementedError
    
    def set_augmentations(self):
        # self.aug_flip = False
        # self.aug_illum = False
        # self.aug_gauss = False
        # self.aug_speed = False
        # self.aug_resizedcrop = False
        # if self.train == 'train_all' or self.train == 'train':
        self.aug_flip = True if 'f' in self.aug else False
        self.aug_illum = True if 'i' in self.aug else False
        self.aug_gauss = True if 'g' in self.aug else False
        self.aug_speed = True if 's' in self.aug else False
        self.aug_resizedcrop = True if 'c' in self.aug else False
        self.aug_reverse = True ## Don't use this with supervised
    
    def apply_transformations(self, clip, idcs, augment=True):
        speed = 1.0
        if True:
            ## Time resampling
            if self.aug_speed and np.random.rand() < 0.9:
                clip, idcs, speed = transforms.augment_speed(clip, idcs, self.T, self.speed_slow, self.speed_fast) # clip: (T, H, W, C) -> (C, T, H, W)
            else:
                clip = clip[idcs].transpose(3, 0, 1, 2) # (T, H, W, C) -> (C, T, H, W)

            ## Randomly horizontal flip
            if self.aug_flip:
                clip = transforms.augment_horizontal_flip(clip)

            ## Randomly reverse time
            if self.aug_reverse:
                clip = transforms.augment_time_reversal(clip)

            ## Illumination noise
            if self.aug_illum:
                clip = transforms.augment_illumination_noise(clip)

            ## Gaussian noise for every pixel
            if self.aug_gauss:
                clip = transforms.augment_gaussian_noise(clip)

            ## Random resized cropping
            if self.aug_resizedcrop:
                clip = transforms.random_resized_crop(clip)

        clip = np.clip(clip, 0, 255)
        clip = clip / 255
        clip = torch.from_numpy(clip).float()

        return clip, idcs, speed

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        '''return: video, ecg, transform_rate, frame_start, frame_end'''
        sample = self.data_list[index]
        start_idx = sample['start_idx']
        video_length = sample['video_length']
        exist_gt_hr = False
        if self.T == -1 and 'test' in self.train:
            with h5py.File(os.path.join(sample['location'], 'sample.hdf5'), 'r') as f:
                video_x = np.array(f['video_data']).transpose(1, 2, 3, 0) # T, H, W, C
                ecg = np.array(f['ecg_data']) # T
                if 'gt_hr' in f.keys():
                    gt_hr = np.array(f['gt_hr'])
                    exist_gt_hr = True
        elif start_idx + int(self.T * 1.5) > video_length:
            with h5py.File(os.path.join(sample['location'], 'sample.hdf5'), 'r') as f:
                video_x = np.array(f['video_data'][:, start_idx: start_idx + self.T]).transpose(1, 2, 3, 0) # T, H, W, C
                ecg = np.array(f['ecg_data'][start_idx: start_idx + self.T]) # T
        else:
            with h5py.File(os.path.join(sample['location'], 'sample.hdf5'), 'r') as f:
                video_x = np.array(f['video_data'][:, start_idx: start_idx + int(self.T * 1.5)]).transpose(1, 2, 3, 0) # T, H, W, C
                ecg = np.array(f['ecg_data'][start_idx: start_idx + int(self.T * 1.5)]) # T

        idcs = np.arange(0, self.T, dtype=int) if self.T != -1 else np.arange(len(video_x), dtype=int)
        video_x_aug, speed_idcs, speed = self.apply_transformations(video_x, idcs)

        # print(f'shape of video_x_aug: {video_x_aug.shape}')

        if speed != 1.0:
            min_idx = int(speed_idcs[0])
            max_idx = int(speed_idcs[-1])+1
            orig_x = np.arange(min_idx, max_idx, dtype=int)
            orig_wave = ecg[orig_x]
            wave = np.interp(speed_idcs, orig_x, orig_wave)
            
        else:
            wave = ecg[idcs]

        # print(f'shape of wave: {wave.shape}')

        # resize to hxw
        if [self.h, self.w] != video_x_aug.shape[1:3]:
            video_x_aug = torch.nn.functional.interpolate(video_x_aug, size=(self.h, self.w), mode='bilinear', align_corners=False)

        wave = wave - wave.mean()
        wave = wave / np.abs(wave).max()
        wave = torch.from_numpy(wave).float()

        sample_item = {}
        sample_item['video'] = video_x_aug
        sample_item['ecg'] = wave
        sample_item['clip_avg_hr'] = gt_hr if exist_gt_hr else cal_hr(wave, 30)

        return sample_item

class PURE(BaseDataset):
    def __init__(self, data_dir='', train='train', T=-1, w=64, h=64, aug=''):
        super().__init__(data_dir, train, T, w, h, aug=aug)
        
    def get_data_list(self):
        date_list = os.listdir(self.data_dir)
        date_list.sort()
        train_list = ['06-01', '06-03', '06-04', '06-05', '06-06', '08-01', '08-02', '08-03', '08-04', '08-05', '08-06',\
                    '05-01', '05-02', '05-03', '05-04', '05-05', '05-06', '01-01', '01-02', '01-03', '01-04', '01-05', '01-06',\
                    '04-01', '04-02', '04-03', '04-04', '04-05', '04-06', '09-01', '09-02', '09-03', '09-04', '09-05', '09-06',\
                    '07-01', '07-02', '07-03', '07-04', '07-05', '07-06']
        if self.train == 'train':
            date_list = [i for i in date_list if i in train_list]
        elif self.train == 'test':
            date_list = [i for i in date_list if i not in train_list]
        elif 'all' in self.train:
            pass
        else:
            raise NotImplementedError

        for date in date_list:
            sample_dir = os.path.join(self.data_dir, date)
            with h5py.File(os.path.join(sample_dir, 'sample.hdf5'), 'r') as f:
                video_length = f['video_data'].shape[1]   # C, T, H, W
            sample_num = video_length // self.T if self.T != -1 else 1
            for i in range(sample_num):
                sample = {}
                sample['location'] = sample_dir
                sample['start_idx'] = i * self.T
                sample['video_length'] = video_length
                self.data_list.append(sample)

class PUREA(PURE):
    def __init__(self, data_dir='', train='train', T=-1, w=64, h=64):
        super().__init__(data_dir, train, T, w, h, aug='figc')
