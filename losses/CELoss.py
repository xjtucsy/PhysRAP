import math
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

# std = 2
def normal_sampling(mean, label_k, std):
    return math.exp(-(label_k-mean) ** 2 / (2*std**2) ) / (math.sqrt(2 * math.pi) * std)

def kl_loss(inputs : torch.Tensor, labels : torch.Tensor):
    criterion = nn.KLDivLoss(reduction='none')
    outputs = torch.log(inputs)
    loss = criterion(outputs, labels)
    loss = loss.sum()
    return loss

class CrossEntropyKL(object):
    def __init__(self):
        super(CrossEntropyKL, self).__init__()
        
    def compute_complex_absolute_given_k(self, output : torch.Tensor, k : torch.Tensor, N : int):
        device = output.device
        two_pi_n_over_N = Variable(2 * math.pi * torch.arange(0, N, dtype=torch.float), requires_grad=True) / N
        hanning = Variable(torch.from_numpy(np.hanning(N)).type(torch.FloatTensor), requires_grad=True).view(1, -1)

        k = k.type(torch.FloatTensor).to(device)
        two_pi_n_over_N = two_pi_n_over_N.to(device)
        hanning = hanning.to(device)

        output = output.view(1, -1) * hanning
        output = output.view(1, 1, -1).type(torch.cuda.FloatTensor)
        k = k.view(1, -1, 1)
        two_pi_n_over_N = two_pi_n_over_N.view(1, 1, -1)
        return (
            torch.sum(output * torch.sin(k * two_pi_n_over_N), dim=-1) ** 2
            + torch.sum(output * torch.cos(k * two_pi_n_over_N), dim=-1) ** 2
        )

    def complex_absolute(self, output : torch.Tensor, Fs : float, bpm_range=None):
        output = output.view(1, -1)

        N = output.size()[1]

        unit_per_hz = Fs / N
        feasible_bpm = bpm_range / 60.0 # 转换到频率
        k = feasible_bpm / unit_per_hz
        
        # only calculate feasible PSD range [0.7,4]Hz
        complex_absolute = self.compute_complex_absolute_given_k(output, k, N)

        return (1.0 / complex_absolute.sum()) * complex_absolute	# Analogous Softmax operator

    
    def cross_entropy_power_spectrum_DLDL_softmax2(self, inputs : torch.Tensor, target : torch.Tensor, Fs, std):
        device = inputs.device

        # 生成目标的心率分布
        target_distribution = [normal_sampling(int(target), i, std) for i in range(140)]
        target_distribution = [max(i, 1e-15) for i in target_distribution]
        target_distribution = torch.Tensor(target_distribution).to(device)

        inputs = inputs.view(1, -1)
        target = target.view(1, -1)

        bpm_range = torch.arange(40, 180, dtype=torch.float).to(device)

        complex_absolute = self.complex_absolute(inputs, Fs, bpm_range)    # 计算pred的功率谱密度

        fre_distribution = F.softmax(complex_absolute.view(-1), dim=0)   # 计算pred的心率分布
        loss_distribution_kl = kl_loss(fre_distribution, target_distribution)   # 计算两个分布之间的距离

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)
        
        return loss_distribution_kl, F.cross_entropy(complex_absolute, target.view((1)).type(torch.long)), abs(target[0].item() - whole_max_idx.item())
    
    def __call__(self, rPPG, clip_average_HR):
        batch_size = rPPG.shape[0]
        frame_rate = 30
        fre_loss, kl_loss, train_mae = 0, 0, 0
        for batch_idx in range(batch_size):
            kl_loss_temp, fre_loss_temp, train_mae_temp = \
                    self.cross_entropy_power_spectrum_DLDL_softmax2(rPPG[batch_idx],\
                                                clip_average_HR[batch_idx] - 40, frame_rate, 1.0)
            fre_loss = fre_loss + fre_loss_temp
            kl_loss = kl_loss + kl_loss_temp
            train_mae = train_mae + train_mae_temp
        return fre_loss / batch_size, kl_loss / batch_size, train_mae / batch_size
