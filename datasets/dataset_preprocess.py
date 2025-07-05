'''
preprocess all rppg datasets: 
    read all rppg signals & videos and save them as hdf5 files (with fps -> 30)
for each .hdf5 file:
    "video_data" : (C, T, H, W), T = 30 / origin_fps * origin_T
    "ecg_data" : (T), T = 30 / origin_ecg_fps * origin_ecg_T
'''

import os
import cv2
import torch
import argparse
import h5py
import json
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat

class PreprocessRppgDatasets:
    def __init__(self, args) -> None:
        self.dataset = args.dataset
        self.dataset_dir = args.dataset_dir
        self.h = args.h
        self.w = args.w

    def read_video(self, frame_path, cur_fps, target_fps=30):
        # read video
        cv2.ocl.setUseOpenCL(False)
        cv2.setNumThreads(0)
        video_x = np.zeros((len(frame_path), self.h, self.w, 3))
        for i, frame in enumerate(frame_path):
            imageBGR = cv2.imread(frame)
            try:
                imageRGB = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)
            except:
                print(f'error in {frame}')
            video_x[i, :, :, :] = cv2.resize(imageRGB, (self.h, self.w), interpolation=cv2.INTER_CUBIC) # T, H, W, C
        
        # resample when the difference of fps is larger than 1
        target_len = int(len(video_x) * target_fps / cur_fps)
        video_x_torch = torch.from_numpy(video_x.transpose(3, 0, 1, 2)).float().unsqueeze(0).cuda()    # (1, c, T, h, w)
        video_x_torch = torch.nn.functional.interpolate(video_x_torch, size=(target_len, self.h, self.w), mode='trilinear', align_corners=False)
        video_x = video_x_torch.squeeze(0).cpu().numpy().transpose(1, 2, 3, 0) # (T, h, w, c)
        return video_x  # (T, h, w, c)
    
    def resample_ecg(self, ecg, target_len):
        # resample when the difference of fps is larger than 1
        if len(ecg) != target_len:
            ecg_torch = torch.from_numpy(ecg).float().unsqueeze(0).unsqueeze(0)    # (1, T)
            ecg_torch = torch.nn.functional.interpolate(ecg_torch, size=(target_len), mode='linear', align_corners=False)
            ecg = ecg_torch.view(-1).numpy() # (T)
        return ecg

    def preprocess(self):
        if self.dataset == 'PURE':
            self.preprocess_PURE()
        else:
            raise NotImplementedError

    def preprocess_PURE(self):
        date_list = os.listdir(self.dataset_dir)
        date_list.sort()
        for date in date_list:
            # read video
            pic_type = 'align_crop_pic'
            video_dir = os.path.join(self.dataset_dir, date, pic_type)
            # read label
            json_file = os.path.join(self.dataset_dir, date, date + ".json")
            with open(json_file, 'r') as f:
                data = json.load(f)
            ecg_time_stamp = np.array([i['Timestamp'] for i in data['/FullPackage']])
            ecg = np.array([i['Value']['waveform'] for i in data['/FullPackage']])
            video_time_stamp = np.array([i['Timestamp'] for i in data['/Image']])
            # 基于json文件中的时间戳，保证视频图片文件名的顺序正确
            frame_path = []
            for i in range(len(video_time_stamp)):
                frame_path.append(os.path.join(video_dir, f"Image{video_time_stamp[i]}.png"))

            ecg_time_diffs = np.diff(ecg_time_stamp / 1e9)
            ecg_rate = 1 / ecg_time_diffs.mean()
            frame_time_diffs = np.diff(video_time_stamp / 1e9)
            frame_rate = 1 / frame_time_diffs.mean()

            video_x = self.read_video(frame_path, cur_fps=frame_rate, target_fps=30)
            ecg_signals = self.resample_ecg(ecg, target_len=len(video_x))

            sample = {}
            sample["video_data"] = video_x.transpose(3, 0, 1, 2) # (C, T, H, W)
            sample["ecg_data"] = ecg_signals
            print(f'\rsaving {self.dataset_dir}/{date}/sample.hdf5', end='')
            with h5py.File(f'{self.dataset_dir}/{date}/sample.hdf5', 'w') as f:
                for key in sample.keys():
                    f.create_dataset(key, data=sample[key])


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', type=str, default='')
    args.add_argument('--dataset_dir', type=str, default='')
    args.add_argument('--h', type=int, default=64)
    args.add_argument('--w', type=int, default=64)
    args = args.parse_args()

    preprocess = PreprocessRppgDatasets(args)
    preprocess.preprocess()