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

- Perform contionual test-time adaptation with:

```shell
python ./trainer.py --save_path /path/to/save_dir/ --dataset {dataset_name} --dataset_dir /path/to/dataset
```

---

#### Cite our method

```
@inproceedings{chu2025remember,
    author = {Chu, Shuyang and Shi, Jingang and Cheng, Xu and Chen, Haoyu and Liu, Xin and Xu, Jian and Zhao, Guoying},
    title = {To Remember, To Adapt, To Preempt: A Stable Continual Test-Time Adaptation Framework for Remote Physiological Measurement in Dynamic Domain Shifts},
    booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia (MM '25)},
    year = {2025},
    isbn = {979-8-4007-2035-2},
    doi = {10.1145/3746027.3754751},
}
```



