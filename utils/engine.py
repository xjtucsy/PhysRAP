import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets.rppg_datasets import VIPL, UBFC, PURE, BUAA, UBFCA, PUREA, BUAAA
from losses.NPLoss import Neg_Pearson
from losses.CELoss import CrossEntropyKL
from archs.ResNet3D import generate_model

dataset_path_map = {
    'PURE': '/data/your_path/pure',
    'PUREA': '/data/your_path/pure',
}

def _init_fn(seed=92):
    np.random.seed(seed)
    
def build_one_dataset(dataset_name, args, mode):
    mode_num_rppg = {
        'train': args.num_rppg,
        'train_all': args.num_rppg,
        'test_all': -1,
        'test' : -1,
    }

    num_rppg = mode_num_rppg[mode]
    train_mode = mode

    if dataset_name == 'PURE':
        dataset = PURE(data_dir=dataset_path_map[dataset_name], T=num_rppg,
                       train=train_mode, w=args.img_size, h=args.img_size)
    # -------------------------------------------------------
    ## PURE-A dataset
    elif dataset_name == 'PUREA':
        dataset = PUREA(data_dir=dataset_path_map[dataset_name], T=num_rppg,
                       train=train_mode, w=args.img_size, h=args.img_size)
    else:
        raise NotImplementedError

    return dataset

def build_dataset(args, mode, batch_size):
    shuffle_flag = mode == 'train'
    
    datasets = args.datasets.split('_')
    

    all_dataloader = []
    
    for dataset_name in datasets:
        dataset = build_one_dataset(dataset_name, args, mode)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=4,
            worker_init_fn=_init_fn,
        )
        all_dataloader.append(dataloader)
        
    return all_dataloader
    
def build_optimizer(args, model):
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        raise NotImplementedError
    return optimizer

def build_scheduler(args, optimizer):
    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.gamma,
        )
    elif args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.factor,
            patience=args.patience,
            verbose=True,
            threshold=args.threshold,
            threshold_mode='rel',
            cooldown=0,
            min_lr=args.min_lr,
            eps=args.eps,
        )
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.T_max,
            eta_min=args.eta_min,
        )
    else:
        raise NotImplementedError
    return scheduler

def build_criterion(args):
    def _build_criterion(criterion_name):
        if criterion_name == 'np_loss':
            return Neg_Pearson()
        elif criterion_name == 'ce_loss':
            return CrossEntropyKL()
        else:
            raise NotImplementedError
    criterion_list = eval(args.loss)
    criterion_dict = {}
    for criterion_name in criterion_list:
        criterion_dict[criterion_name] = _build_criterion(criterion_name)
    return criterion_dict

def build_model(args):
    if args.model == 'ResNet3D':
        model = generate_model(model_depth=18, num_frames=args.num_rppg)
    else:
        raise NotImplementedError
    return model  


        
        