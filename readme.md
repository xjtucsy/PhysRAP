### PhysRAP

---

#### Introduction

Main code of [**To Remember, To Adapt, To Preempt: A Continual Test-Time Adaptation Framework for Remote Physiological Measurement in Dynamic Domain Shifts**](https://arxiv.org/html/2510.01282v1).

![main_framework_rebuttal_version](readme/main_framework_rebuttal_version.png)

---

#### Reproduce our method

##### 1 video preprocess

- For any video dataset, we need crop the face firstly, use:

```shell
python ./datasets/video_preprocess.py --dataset {dataset_name}
```

##### 2 dataset preprocess

- For any video dataset, we need prepare the dataset for training, use:

```shell
python ./datasets/dataset_preprocess.py --dataset_dir {dataset_dir} --dataset {dataset_name}
```

##### 3 test-time adaptation (PhysRAP)

- Perform contionual test-time adaptation with ([Pretrained Model](https://drive.google.com/file/d/18LgvH-crx_dGpP6eP5eatNmi3VD-9Z91/view?usp=sharing)):

```shell
python ./trainer.py --save_path /path/to/save_dir/ --dataset {dataset_name} --dataset_dir /path/to/dataset
```

---

#### Cite our method

```
@inproceedings{10.1145/3746027.3754751,
author = {Chu, Shuyang and Shi, Jingang and Cheng, Xu and Chen, Haoyu and Liu, Xin and Xu, Jian and Zhao, Guoying},
title = {To Remember, To Adapt, To Preempt: A Stable Continual Test-Time Adaptation Framework for Remote Physiological Measurement in Dynamic Domain Shifts},
year = {2025},
doi = {10.1145/3746027.3754751},
booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
pages = {7307–7316}
}
```




