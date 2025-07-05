import math
import torch, os
import numpy as np

import random
import time
import argparse
import logging
import shutil

from copy import deepcopy
from scipy import io as sio
from scipy import signal
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils.engine import build_dataset, build_optimizer, build_scheduler, build_criterion, build_model
from utils.util import AvgrageMeter, pearson_correlation_coefficient, update_avg_meters, cal_psd_hr, \
    augment_flip, augment_time_reversal, random_resized_crop, augment_gaussian_noise


def set_seed(seed=92):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class RppgEstimatorTrainer:
    def __init__(self, args) -> None:
        self.args = args
        
        self.gpu_list = [int(i) for i in args.gpu.split(',')]
        self.gpu_num = len(self.gpu_list)
        self.actual_batch_size = args.batch_size * self.gpu_num
        self.device = torch.device(f'cuda:{self.gpu_list[0]}')
        
        self.rppg_estimator_stu = build_model(args).to(self.device)
        self.rppg_estimator_teacher = build_model(args).to(self.device)

        ## generate save path
        self.run_date = time.strftime('%m%d_%H%M', time.localtime(time.time()))
        self.current_date = self.run_date
        self.save_path = f'{args.save_path}/{self.run_date}'
        
        ## dataloader NOTE: SELECT YOUR DATASET
        self.all_datasets = self.args.datasets.split('_') # VIPL_BUAA_UBFC_PURE' => ['VIPL', 'BUAA', 'UBFC', 'PURE']
        self.train_dataloaders = build_dataset(args, mode='train_all', batch_size=self.actual_batch_size)
        self.val_dataloaders = build_dataset(args, mode='test_all', batch_size=1)

        ## optimizer
        self.optimizer = build_optimizer(args, self.rppg_estimator_stu)
        self.scheduler = build_scheduler(args, self.optimizer)
        self.loss_funcs = build_criterion(args)
        self.loss_funcs_weight = dict(zip(eval(args.loss), eval(args.loss_weight)))
        
        ## loss & metrics saver
        self.loss_meters = dict([(key, AvgrageMeter()) for key in self.loss_funcs.keys()])
        self.metrics_meters = {
            'mae': AvgrageMeter(),
        }

        ## constant
        self.bpm_range = torch.arange(40, 180, dtype=torch.float).to(self.device)
        self.best_epoch = 0
        self.best_val_mae = 1000    # mean absolute error
        self.best_val_rmse = 1000   # root mean square error
        self.best_sd = 1000         # standard deviation
        self.best_r = 0             # Pearson’s correlation coefficient
        self.frame_rate = 30        
    
    def prepare_train(self, start_dataset_idx, continue_log):
        """Prepares the training process.

        Sets up the necessary directories for saving checkpoints and logs. Initializes the logger for logging training progress.
        Copies the current file to the save path. Loads the checkpoint if starting from a specific epoch.
        Sets the rppg_estimator to train mode.

        Args:
            start_dataset_idx (int): The starting start_dataset_idx for training.
            continue_log (str): The name of the log to continue from.

        Raises:
            Exception: If the rppg_estimator checkpoint file for the previous epoch is not found.

        Returns:
            None
        """
        if start_dataset_idx != 0:
            self.save_path = f'{self.args.save_path}/{continue_log}'
            self.run_date = continue_log

        self.save_ckpt_path = f'{self.save_path}/ckpt'
        self.save_rppg_path = f'{self.save_path}/rppg'
        if not os.path.exists(self.save_ckpt_path):
            os.makedirs(self.save_ckpt_path)
        if not os.path.exists(self.save_rppg_path):
            os.makedirs(self.save_rppg_path)

        all_dataset_first_name = ''.join([i[0] for i in self.all_datasets])
        logging.basicConfig(filename=f'./logs/{self.args.model}_{all_dataset_first_name}_{self.args.num_rppg}_S{self.run_date}_N{self.current_date}.log',\
                            format='%(message)s', filemode='a')
        self.logger = logging.getLogger(f'./logs/{self.args.model}_{all_dataset_first_name}_{self.args.num_rppg}_S{self.run_date}_N{self.current_date}')
        self.logger.setLevel(logging.INFO)

        ## save proj_file to save_path
        cur_file = os.getcwd()
        cur_file_name = cur_file.split('/')[-1]
        shutil.copytree(f'{cur_file}', f'{self.save_path}/{self.current_date}/{cur_file_name}', dirs_exist_ok=True)

        if start_dataset_idx != 0:
            if not os.path.exists(f'{self.save_ckpt_path}/rppg_estimator_stu_dataset_{start_dataset_idx-1}.pth'):
                raise Exception(f'rppg_estimator_stu ckpt file {start_dataset_idx-1} not found')
            self.rppg_estimator_stu.load_state_dict(torch.load(f'{self.save_ckpt_path}/rppg_estimator_stu_dataset_{start_dataset_idx-1}.pth', map_location=self.device))
            self.rppg_estimator_teacher.load_state_dict(torch.load(f'{self.save_ckpt_path}/rppg_estimator_teacher_dataset_{start_dataset_idx-1}.pth', map_location=self.device))
        print(f'save_path: {self.save_path}, log_path: ./logs/{self.args.model}_{self.args.datasets}_{self.args.num_rppg}_{self.run_date}')

        ## block gradient and set train
        self.rppg_estimator_stu.train()
        self.rppg_estimator_teacher.eval()
        
    def draw_rppg_ecg(self, rPPG, ecg, save_path_epoch, train=False, mini_batch=0):
        """Draws rPPG and ECG signals, saves the results, and plots the power spectral density.

        Filters the rPPG signal using a bandpass filter. Saves the filtered rPPG and ECG signals as a .mat file.
        Plots the rPPG and ECG signals, as well as the power spectral density, and saves the figure as a .jpg file.

        Args:
            rPPG (Tensor): The rPPG signal.
            ecg (Tensor): The ECG signal.
            save_path_epoch (str): The path to save the results.
            train (bool, optional): Whether it is in training mode. Defaults to False.
            mini_batch (int, optional): The mini-batch number. Defaults to 0.

        Returns:
            None
        """
        rPPG_sample, ecg_sample = rPPG[0], ecg[0]
        ## save the results
        b, a = signal.butter(2, [0.67 / 15, 3 / 15], 'bandpass')
        # 使用 lfilter 函数进行滤波 
        rPPG_np = rPPG_sample.cpu().data.numpy()
        rPPG_np = signal.lfilter(b, a, rPPG_np)
        y1 = rPPG_np
        y2 = ecg_sample.cpu().data.numpy()
        results_rPPG = [y1, y2]
        if not train:
            sio.savemat(
                os.path.join(save_path_epoch, 'test_rPPG.mat'),
                {'results_rPPG': results_rPPG},
            )
        else:
            sio.savemat(os.path.join(save_path_epoch, f'minibatch_{mini_batch+1:0>4}_rPPG.mat'), {'results_rPPG': results_rPPG})
        # show the ecg images
        fig, ax = plt.subplots(2, 1, figsize=(20, 10))
        psd_pred = cal_psd_hr(rPPG_sample, self.frame_rate, return_type='psd')
        psd_gt = cal_psd_hr(ecg_sample, self.frame_rate, return_type='psd')
        ax[0].set_title('rPPG')
        ax[0].plot(y1, label='rPPG')
        ax[0].plot(y2, label='ecg')
        ax[0].legend()
        ax[1].set_title('psd')
        ax[1].plot(psd_pred.cpu().data.numpy(), label='pred')
        ax[1].plot(psd_gt.cpu().data.numpy(), label='gt')
        ax[1].legend()
        if not train:
            fig.savefig(os.path.join(save_path_epoch, 'test_rPPG.jpg'))
        else:
            fig.savefig(os.path.join(save_path_epoch, f'minibatch_{mini_batch+1:0>4}_rPPG.jpg'))
        plt.close(fig)
    
    def update_best(self, epoch, hr_pred, hr_gt, val_type='video'):
        """Updates the best validation metrics and saves the model if the current metrics are better.

        Calculates the mean absolute error (MAE), root mean squared error (RMSE), standard deviation (SD),
        and Pearson correlation coefficient (R) between the predicted and ground truth heart rates.
        If the current MAE is lower than the best MAE, updates the best MAE, RMSE, epoch, SD, and R,
        and saves the model.

        Args:
            epoch (int): The current epoch number.
            hr_pred (List[float]): The predicted heart rates.
            hr_gt (List[float]): The ground truth heart rates.
            val_type (str, optional): The type of validation. Defaults to 'video'.

        Returns:
            None
        """
        cur_mae = np.mean(np.abs(np.array(hr_gt) - np.array(hr_pred)))
        cur_rmse = np.sqrt(np.mean(np.square(np.array(hr_gt) - np.array(hr_pred))))
        cur_sd = np.std(np.array(hr_gt) - np.array(hr_pred))
        cur_r = pearson_correlation_coefficient(np.array(hr_gt), np.array(hr_pred))

        self.logger.info(f'evaluate epoch {epoch}, total val {len(hr_gt)} ----------------------------------')
        self.logger.info(f'{val_type}-level mae of model: {np.mean(np.abs(np.array(hr_gt) - np.array(hr_pred)))}')
        self.logger.info(f'{val_type}-level cur mae: {cur_mae:.2f}, cur rmse: {cur_rmse:.2f}, cur sd: {cur_sd:.2f}, cur r: {cur_r:.4f}')
        self.logger.info(f'{val_type}-level best mae of model: {self.best_val_mae:.2f}, best rmse: {self.best_val_rmse:.2f}, best epoch: {self.best_epoch}, ' \
                         f'best sd: {self.best_sd:.2f}, best r: {self.best_r:.4f}')
        self.logger.info(
            '------------------------------------------------------------------'
        )
        
        return cur_mae, cur_rmse, cur_sd, cur_r
    
    def evaluate_clip(self, epoch = 0, val_dataloader=None):
        """Evaluates the clip data and saves the results.

        Evaluates the clip data by calculating the power spectral density (PSD) for each clip.
        Saves the PSD results and updates the best validation metrics.

        Args:
            epoch (int, optional): The current epoch number. Defaults to 0.

        Returns:
            None
        """
        save_path_epoch = f'{self.save_rppg_path}/{epoch:0>3}'
        hr_gt = []
        hr_pred = []

        val_rppg_estimator = build_model(self.args).to(self.device)
        val_rppg_estimator.load_state_dict(torch.load(f'{self.save_ckpt_path}/rppg_estimator_stu_epoch_{epoch}.pth'))
        val_rppg_estimator.eval()

        with torch.no_grad():
            for sample_batched in tqdm(val_dataloader):
                # get the inputs
                inputs, ecg, clip_average_HR = sample_batched['video'].to(self.device),\
                    sample_batched['ecg'].to(self.device), sample_batched['clip_avg_hr'].to(self.device)
              
                ## for gt:
                for hr in clip_average_HR:
                    hr_gt.append(hr.cpu()) # [M : [B, 1]]

                ## for rppg_estimator:
                all_inputs = {
                    'input_clip': inputs,
                }
                outputs = val_rppg_estimator(all_inputs)
                rPPG = outputs['rPPG']
                for batch_idx in range(rPPG.shape[0]):
                    psd_pred = cal_psd_hr(rPPG[batch_idx], self.frame_rate, return_type='psd')
                    hr_pred.append(psd_pred.max(0)[1].cpu() + 40)

        ## save the results
        self.draw_rppg_ecg(rPPG, ecg, save_path_epoch)
        self.update_best(epoch, hr_pred, hr_gt, val_type='clip')
        
    def initial_train_one_epoch(self, epoch, save_path_epoch, train_dataloader):
        # sourcery skip: merge-dict-assign
        with tqdm(range(len(train_dataloader))) as pbar:
            for iter_idx, sample_batched in zip(pbar, train_dataloader):
                inputs, ecg, clip_average_HR = sample_batched['video'].to(self.device), \
                    sample_batched['ecg'].to(self.device), sample_batched['clip_avg_hr'].to(self.device)

                self.optimizer.zero_grad()
                # forward + backward + optimize
                ## for backbone (PhysNet or others):
                all_inputs = {
                    'input_clip': inputs,
                    'gra_sharp': 2.0
                }
                outputs = self.rppg_estimator_stu(all_inputs)            # estimate rPPG signal
                rPPG = outputs['rPPG']
                ## calculate loss
                train_losses = {}

                train_losses['np_loss'] = self.loss_funcs['np_loss'](rPPG, ecg)   # calculate the loss of rPPG signal

                fre_loss, kl_loss, train_mae = self.loss_funcs['ce_loss'](rPPG, clip_average_HR)  # calculate the loss of KL divergence
                train_losses['ce_loss'] = fre_loss + kl_loss

                total_loss = sum(
                    train_losses[key] * self.loss_funcs_weight[key]
                    for key in train_losses
                )
                total_loss.backward()
                self.optimizer.step()
                ## update loss saver and metrics saver
                train_metrics = {
                    'mae': train_mae,
                }
                update_avg_meters(self.loss_meters, train_losses, self.actual_batch_size)
                update_avg_meters(self.metrics_meters, train_metrics, self.actual_batch_size)

                mini_batch_info = f'epoch : {epoch:0>3}, mini-batch : {iter_idx:0>4}, lr = {self.optimizer.param_groups[0]["lr"]:.5f}'
                loss_info = ', '.join([f'{key} = {self.loss_meters[key].avg:.4f}' for key in self.loss_meters.keys()])
                metrics_info = ', '.join([f'{key} = {self.metrics_meters[key].avg:.4f}' for key in self.metrics_meters.keys()])

                if iter_idx % self.args.echo_batches == self.args.echo_batches - 1:  # info every mini-batches
                    self.logger.info(', '.join([mini_batch_info, loss_info, metrics_info]))
                    # save the ecg images
                    self.draw_rppg_ecg(rPPG, ecg, save_path_epoch, train=True, mini_batch=iter_idx)

                pbar.set_description(', '.join([mini_batch_info, loss_info, metrics_info]))
        self.scheduler.step()
        return train_losses
             
    def initial_train(self, dataset_idx):
        for epoch in range(0, self.args.epochs):
            save_path_epoch = f'{self.save_rppg_path}/{epoch:0>3}'
            if not os.path.exists(save_path_epoch):
                os.makedirs(save_path_epoch)
            self.logger.info(f'train epoch: {epoch} lr: {self.optimizer.param_groups[0]["lr"]:.5f}')
            self.initial_train_one_epoch(epoch, save_path_epoch, self.train_dataloaders[dataset_idx])
            
            # save the model
            torch.save(self.rppg_estimator_stu.state_dict(), os.path.join(self.save_ckpt_path, f'rppg_estimator_stu_epoch_{epoch}.pth'))
                    
        torch.save(self.rppg_estimator_stu.state_dict(), os.path.join(self.save_ckpt_path, f'rppg_estimator_stu_dataset_{dataset_idx}.pth'))
        torch.save(self.rppg_estimator_teacher.state_dict(), os.path.join(self.save_ckpt_path, f'rppg_estimator_teacher_dataset_{dataset_idx}.pth'))
    
    def continue_tta(self, dataset_idx):
        """TTA the model && cal the metrics.
        """
        
        def argumation(inputs):
            # inputs : [B, T, C, H, W]
            # outputs : argumation_inputs [N: [B, T, C, H, W]]
            N = 10
            aug_videos = []
            available_augs = [augment_gaussian_noise, random_resized_crop, augment_flip, augment_time_reversal]
            for _ in range(N):
                aug_videos.append(random.choice(available_augs)(inputs))
            return aug_videos

        LOAD_DATASET = dataset_idx-1
        # LOAD_DATASET = 0 #! 注意，TTA设置下应当保证 teacher 每次加载的是 0 的 stu
        self.rppg_estimator_stu.load_state_dict(torch.load(f'{self.save_ckpt_path}/rppg_estimator_stu_dataset_{LOAD_DATASET}.pth', map_location=self.device))
        if dataset_idx == 1:
            self.rppg_estimator_teacher.load_state_dict(torch.load(f'{self.save_ckpt_path}/rppg_estimator_stu_dataset_{LOAD_DATASET}.pth', map_location=self.device))
        else:
            self.rppg_estimator_teacher.load_state_dict(torch.load(f'{self.save_ckpt_path}/rppg_estimator_teacher_dataset_{LOAD_DATASET}.pth', map_location=self.device))
        
        self.rppg_estimator_stu.train()
        self.rppg_estimator_teacher.eval()
        
        tta_dataloader = self.val_dataloaders[dataset_idx]
        hr_gt = []
        hr_pred = []
        

        for sample_batched in tqdm(tta_dataloader):
            # get the inputs
            inputs, ecg, clip_average_HR = sample_batched['video'].to(self.device),\
                sample_batched['ecg'].to(self.device), sample_batched['clip_avg_hr'].to(self.device)
                
            # split the video-level inputs [B,C,T,H,W] to clip-level inputs : T//160 [B,C,160,H,W]
            
            B, C, T, H, W = inputs.shape
            num_clip_per_video = T // self.args.num_rppg
            inputs = inputs[:, :, :num_clip_per_video * self.args.num_rppg, :, :]
            clip_level_inputs = inputs.view(B, C, T//self.args.num_rppg, self.args.num_rppg, H, W).permute(2, 0, 1, 3, 4, 5)
            clip_level_inputs = clip_level_inputs[:num_clip_per_video//self.args.batch_size*self.args.batch_size]
            clip_level_inputs = clip_level_inputs.view(-1, self.args.batch_size, C, self.args.num_rppg, H, W)
            #! TTA for each CLIP
            for clip_idx, clip_input in enumerate(clip_level_inputs): # 0 -> T//160, clip_input \in [B, 160, C, H, W]
                ## ! calcualte the augmentated samples, rPPGs, psds
                augment_inputs = argumation(clip_input)
                augment_psds = [] # [N: [B, 140]]
                augment_rppgs = [] # [N : [B, T] ]
                for augment_input in augment_inputs:
                    output_rppg = self.rppg_estimator_teacher({'input_clip' : augment_input})['rPPG']
                    output_psd_all_batch = []
                    for batch_idx in range(augment_input.shape[0]):
                        output_psd = cal_psd_hr(output_rppg[batch_idx], self.frame_rate, return_type='psd')
                        output_psd_all_batch.append(output_psd) # output_psd : [140,]
                    # output_psds [B, 140]
                    output_psds = torch.stack(output_psd_all_batch, 0)
                    augment_psds.append(output_psds)
                    augment_rppgs.append(output_rppg)
                augment_rppgs = torch.stack(augment_rppgs, 0) # [N, B, T]
                augment_psds = torch.stack(augment_psds, 0) # [N, B, 140]
                
                
                origional_rppg = self.rppg_estimator_stu({'input_clip' : clip_input})['rPPG']
                origional_psds = []
                for batch_idx in range(clip_input.shape[0]):
                    output_psd = cal_psd_hr(origional_rppg[batch_idx], self.frame_rate, return_type='psd')
                    origional_psds.append(output_psd) # output_psd : [140,]
                origional_psds = torch.stack(origional_psds, 0) # [B, 140]
                
                ## ! pesudo label obtained by rppg signal uncertainty
                all_batch_rppg_uncertainty = []
                for batch_idx in range(clip_input.shape[0]):
                    cur_batch_psd = augment_psds[:, batch_idx, :]
                    cur_batch_rppg = augment_rppgs[:, batch_idx, :] # [N, 140]
                    cur_batch_rppg = cur_batch_rppg / torch.norm(cur_batch_rppg, p=2, dim=1).unsqueeze(1) # [N, 140]
                    cur_batch_origional_rppg = origional_rppg[batch_idx, :]
                    cur_batch_origional_rppg = cur_batch_origional_rppg / torch.norm(cur_batch_origional_rppg, p=2)
                    cur_batch_rppg_diff = cur_batch_rppg - cur_batch_origional_rppg
                    cur_batch_rppg_uncertainty = torch.exp(cur_batch_rppg_diff.mean(1)) # [N,]
                    all_batch_rppg_uncertainty.append(cur_batch_rppg_uncertainty)
                all_batch_rppg_uncertainty = torch.stack(all_batch_rppg_uncertainty, 0) # [B, N]
                
                ## ! pesudo label obtained by psd signal uncertainty
                pesudo_label_psd_hr = []
                all_batch_psd_uncertainty = []
                for batch_idx in range(clip_input.shape[0]):
                    cur_batch_psd = augment_psds[:, batch_idx, :] # [N, 140]
                    cur_batch_psd = cur_batch_psd / torch.norm(cur_batch_psd, p=2, dim=1).unsqueeze(1) # [N, 140]
                    cur_batch_origional_psd = origional_psds[batch_idx, :]
                    cur_batch_origional_psd = cur_batch_origional_psd / torch.norm(cur_batch_origional_psd, p=2)
                    cur_batch_psd_diff = cur_batch_psd - cur_batch_origional_psd
                    cur_batch_psd_uncertainty = torch.exp(cur_batch_psd_diff.mean(1)) # [N,]
                    all_batch_psd_uncertainty.append(cur_batch_psd_uncertainty)
                    pesudo_label_psd_per_batch = cur_batch_psd_uncertainty.unsqueeze(1) * cur_batch_psd # [N, 140]
                    pesudo_label_psd_hr.append(pesudo_label_psd_per_batch.mean(0).max(0)[1] + 40)
                all_batch_psd_uncertainty = torch.stack(all_batch_psd_uncertainty, 0) # [B, N]
                pesudo_label_hr = torch.stack(pesudo_label_psd_hr, 0)
                                    
                # ## ! store the original params. dict
                origional_params = {k: v.clone() for k, v in self.rppg_estimator_stu.named_parameters()}
                
                # # ## ! calculate the gradient of uncertantity to get FIM matrix I
                            
                uncertainty = all_batch_psd_uncertainty + all_batch_rppg_uncertainty # [B, N]
                self.optimizer.zero_grad() #! zero the grad
                uncertainty.mean(0).mean(0).backward() #! backward to get the gradient of uncertainty
                with torch.no_grad():
                    fisher_dict = {} # ! 就是在不确定度回传梯度之后所有参数的梯度
                    for nm, m  in self.rppg_estimator_stu.named_modules():  ## previously used model, but now using self.model
                        for npp, p in m.named_parameters():
                            if npp in ['weight', 'bias'] and p.requires_grad:
                                fisher_dict[f"{nm}.{npp}"] = p.grad.data.clone().view(-1)
                    fisher_grads_name = list(fisher_dict.keys()) #! 每个参数的名字 len(M)
                    fisher_grads = [fisher_dict[key] for key in fisher_grads_name] #! 每个参数的梯度 len(M)
                    fim_matrix = torch.from_numpy(np.zeros((len(fisher_grads), len(fisher_grads)))) #! 初始化的FIM矩阵 MxM
                    for i in range(len(fisher_grads)): #! 计算FIM矩阵，对角线表示模块的敏感度，非对角线表示模块间的关联性
                        for j in range(len(fisher_grads)):
                            param_i, param_j = fisher_grads[i], fisher_grads[j]
                            sim_score = torch.sum(param_i.mean() * param_j.mean())
                            fim_matrix[i, j] = sim_score.item()
                    need_to_update = {}
                    diag_fim = torch.diag(fim_matrix) #! 对角线 [M,]
                    save_ratio = 0.8 #! 敏感度位于前1-save_ratio的更新，保留a%
                    related_save_ratio = 0.2 #! 对于每个模块，选择关联度位于前related_save_ratio的不更新，保留a%
                    threshold = torch.sort(diag_fim, descending=True)[0][int(diag_fim.shape[0] * (1 - save_ratio))] # 敏感度高于的a%的更新，保留1-a% 
                    for i in range(diag_fim.shape[0]):
                        if diag_fim[i] > threshold:
                            need_to_update[f"{fisher_grads_name[i]}"] = origional_params[f"{fisher_grads_name[i]}"]
                            for j in range(diag_fim.shape[0]):
                                realted_threshold = torch.sort(fim_matrix[i], descending=True)[0][int(diag_fim.shape[0] * related_save_ratio)]
                                if fim_matrix[i, j] > realted_threshold and j != i and fisher_grads_name[j] in need_to_update:
                                    need_to_update.pop(f"{fisher_grads_name[j]}")
                
                ## ! Look for future, get the future_grad for the augmentated inputs
                self.optimizer.zero_grad()
                K = 4
                for i in range(K):
                    selected_inputs = random.choice(augment_inputs)
                    selected_rppg = self.rppg_estimator_stu({'input_clip' : selected_inputs})['rPPG']
                    fre_loss, kl_loss, train_mae = self.loss_funcs['ce_loss'](selected_rppg, pesudo_label_hr.detach())  # calculate the loss of KL divergence
                    total_loss = fre_loss + kl_loss
                    total_loss.backward()
                future_grads = {}
                for nm, m  in self.rppg_estimator_stu.named_modules():  ## previously used model, but now using self.model
                    for npp, p in m.named_parameters():
                        if npp in ['weight', 'bias'] and p.requires_grad:
                            future_grads[f"{nm}.{npp}"] = p.grad.data.clone().view(-1)
                    
                
                ## ! for rppg_estimator_stu / first_eval & optimize: [!!! both the pesudo_label loss and prior loss !!!]
                self.optimizer.zero_grad()
                rPPG = self.rppg_estimator_stu({'input_clip': clip_input})['rPPG']            
                fre_loss, kl_loss, train_mae = self.loss_funcs['ce_loss'](rPPG, pesudo_label_hr.detach())  # calculate the loss of KL divergence
                total_loss = fre_loss + kl_loss
                total_loss.backward()
                for nm, m  in self.rppg_estimator_stu.named_modules():  ## previously used model, but now using self.model
                    for npp, p in m.named_parameters():
                        if npp in ['weight', 'bias'] and p.requires_grad and f"{nm}.{npp}" in need_to_update:
                            #! calculate the cos value 
                            current_grad = p.grad.data.clone().view(-1)
                            cos_value = torch.dot(current_grad, future_grads[f"{nm}.{npp}"]) /\
                                        (torch.norm(current_grad) * torch.norm(future_grads[f"{nm}.{npp}"]))
                            if cos_value > 0: #? denotes the angle is smaller than 90 degree, modify the gradient
                                # ? future 到 current 的 投影模长
                                weight = torch.norm(future_grads[f"{nm}.{npp}"]) * (cos_value - math.sqrt(2)/2) / torch.norm(current_grad)
                                weight_exp = 1 / (1 + torch.exp(-weight))
                                p.grad.data = p.grad.data * weight_exp
                            else: #? denotes the angle is larger than 90 degree, refuse to update
                                need_to_update.pop(f"{nm}.{npp}")
                self.optimizer.step()
                
                ## ! reset the params. dict according to the original params. dict & threshold
                for nm, m  in self.rppg_estimator_stu.named_modules():
                    for npp, p in m.named_parameters():
                        if npp in ['weight', 'bias'] and p.requires_grad and f"{nm}.{npp}" not in need_to_update:
                            mask_fish = torch.ones_like(origional_params[f"{nm}.{npp}"])  # restore的比例 1就是100%回溯
                            mask = mask_fish
                            with torch.no_grad():
                                p.data = origional_params[f"{nm}.{npp}"] * mask + p * (1.-mask)
                
                ## ! ema update the teacher model
                alpha = 0.99
                for param_teacher, param_student in zip(self.rppg_estimator_teacher.parameters(), self.rppg_estimator_stu.parameters()):
                    param_teacher.data = alpha * param_teacher.data + (1 - alpha) * param_student.data
            
            #! for rppg_estimator_stu / final inference:
            torch.save(self.rppg_estimator_stu.state_dict(), os.path.join(self.save_ckpt_path, f'rppg_estimator_stu_dataset_{dataset_idx}_tmp.pth'))
            
            with torch.no_grad():
                num_clip = 3
                input_len = inputs.shape[2]
                input_len = input_len - input_len % (num_clip * 4)
                clip_len = input_len // num_clip
                inputs = inputs[:, :, :input_len, :, :]
                ecg = ecg[:, :input_len]

                new_args = deepcopy(self.args)
                new_args.num_rppg = clip_len
                val_rppg_estimator = build_model(new_args).to(self.device)
                val_rppg_estimator.load_state_dict(torch.load(f'{self.save_ckpt_path}/rppg_estimator_stu_dataset_{dataset_idx}_tmp.pth', map_location=self.device))
                val_rppg_estimator.eval()
                psd_gt_total = 0
                psd_pred_total = 0
                for idx in range(num_clip):

                    inputs_iter = inputs[:, :, idx*clip_len : (idx+1)*clip_len, :, :]
                    ecg_iter = ecg[:, idx*clip_len : (idx+1)*clip_len]

                    psd_gt = cal_psd_hr(ecg_iter, self.frame_rate, return_type='psd')
                    psd_gt_total += psd_gt.view(-1).max(0)[1].cpu() + 40

                    ## for rppg_estimator:
                    all_inputs = {
                        'input_clip': inputs_iter,
                    }
                    outputs = val_rppg_estimator(all_inputs)
                    rPPG = outputs['rPPG']

                    psd_pred = cal_psd_hr(rPPG[0], self.frame_rate, return_type='psd')
                    psd_pred_total += psd_pred.view(-1).max(0)[1].cpu() + 40

                hr_pred.append(psd_pred_total / num_clip)
                hr_gt.append(psd_gt_total / num_clip)
            
        ## save the results
        cur_mae, cur_rmse, cur_sd, cur_r =  self.update_best(-1, hr_pred, hr_gt, val_type='clip')
        
        torch.save(self.rppg_estimator_stu.state_dict(), os.path.join(self.save_ckpt_path, f'rppg_estimator_stu_dataset_{dataset_idx}.pth'))
        torch.save(self.rppg_estimator_teacher.state_dict(), os.path.join(self.save_ckpt_path, f'rppg_estimator_teacher_dataset_{dataset_idx}.pth'))
        
        return cur_mae, cur_rmse, cur_sd, cur_r
                
    def train(self, start_dataset_idx, continue_log=''):
        self.prepare_train(start_dataset_idx, continue_log)
        self.logger.info(f'================================== Current Log Time : {self.current_date} ================================== \n'\
            f'prepare train, load ckpt and block gradient, start_dataset_idx: {start_dataset_idx}, gpu: {self.gpu_list}.\n'\
            f'dataset: {self.args.datasets}, num_rppg: {self.args.num_rppg}, model: {self.args.model}, loss: {self.loss_funcs_weight}.\n'\
            f'batch_size: {self.actual_batch_size}, lr: {self.args.lr}, optim: {self.args.optim}, scheduler: {self.args.scheduler}.')
        if start_dataset_idx == 0:
            self.logger.info(f'===== Training at the dataset : {self.all_datasets[0]} =====')
            self.initial_train(dataset_idx=0)
        mean_mae, mean_rmse, mean_sd, mean_r = [], [], [], []
        for i in range(max(1, start_dataset_idx), len(self.all_datasets)):
            self.logger.info(f'===== TTA at the dataset : {self.all_datasets[i]} =====')
            cur_mae, cur_rmse, cur_sd, cur_r = self.continue_tta(dataset_idx=i)
            mean_mae.append(cur_mae)
            mean_rmse.append(cur_rmse)
            mean_sd.append(cur_sd)
            mean_r.append(cur_r)
        self.logger.info(f'===== MEAN RESULTs at all datasets =====\n'\
                f'MAE: {np.mean(mean_mae)}, RMSE: {np.mean(mean_rmse)}, SD: {np.mean(mean_sd)}, R: {np.mean(mean_r)}')
        
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## ! general params.
    parser.add_argument('--num_rppg', type=int, default=160, help='the number of rPPG')
    parser.add_argument('--datasets', type=str, default='VIPL_UBFC_UBFCA_PURE_PUREA_BUAA_BUAAA', help='dataset')
    parser.add_argument('--vipl_fold', type=int, default=-1, help='the fold of VIPL dataset, not used')
    parser.add_argument('--save_path', type=str, default='path/to/your/save_dir', help='the path to save the model [ckpt, code, visulization]')
    parser.add_argument('--save_mode', type=str, default='all', help='save mode [all, best]')

    ## ! train params.
    parser.add_argument('--gpu', type=str, default="2", help='gpu id list')
    parser.add_argument('--img_size', type=int, default=128, help='the length of clip')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size per gpu')
    parser.add_argument('--eval_step', type=int, default=1, help='the number of **epochs** to eval')
    parser.add_argument('--epochs', type=int, default=20, help='the number of epochs to train')
    parser.add_argument('--echo_batches', type=int, default=500, help='the number of **mini-batches** to print the loss')
    ### loss
    parser.add_argument('--loss', type=str, default='["np_loss", "ce_loss"]', help='loss = [np_loss, ce_loss]')
    parser.add_argument('--loss_weight', type=str, default='[0, 1]', help='loss_weight = [1, 1]')
    
    ## ! model params.
    parser.add_argument('--model', type=str, default='ResNet3D', help='model')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    ### optim
    parser.add_argument('--optim', type=str, default='adam', help='optimizer = [adam, sgd]')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight decay for optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd')
    ### scheduler
    parser.add_argument('--scheduler', type=str, default='step', help='scheduler = [step]')
    parser.add_argument('--step_size', type=int, default=50, help='learning rate decay step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay')

    args = parser.parse_args()

    set_seed(92)

    rppg_estimator_trainer = RppgEstimatorTrainer(args)

    rppg_estimator_trainer.train(start_dataset_idx=0, continue_log='') # NOTE: WHETHER TO CONTINUE TRAINING
