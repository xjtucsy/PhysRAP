### PhysRAP

---

#### Introduction

Main code of **To Remember, To Adapt, To Preempt: A Continual Test-Time Adaptation Framework for Remote Physiological Measurement in Dynamic Domain Shifts**.

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

```

