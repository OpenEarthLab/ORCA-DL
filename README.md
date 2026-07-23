# Data-driven Global Ocean Modeling for Seasonal to Decadal Prediction [Accepted by *Science Advances*]

<!-- <div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2405.15412-b31b1b.svg)](https://arxiv.org/abs/2405.15412)

</div> -->

Paper Link: [https://www.science.org/doi/full/10.1126/sciadv.adu2488](https://www.science.org/doi/full/10.1126/sciadv.adu2488)

**This repository contains the official implementation of ORCA-DL**
---------------------------------------------------------------

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/OpenEarthLab/ORCA-DL.git
cd ORCA-DL
conda create -n orca python=3.9.17
conda activate orca
pip install -r requirements.txt
```

### Resources Download

All the predictions, trained model weights and data can be found in:
https://1drv.ms/f/c/49d761d10f0b201d/Emi9scIyaWBCrNTgRo6t12oBLnF2qGDRGj0M7-g0ekRM1A

Model weights and statistics can also be found in the huggingface repo for quick access:
https://huggingface.co/datasets/JayKuo/ORCA-DL-data/tree/main

> **Note**
> The data in the `train_data` and `valid_test_data` directories have been interpolated and normalized using the mean and standard deviation provided in the `stat` directory. Therefore, they can be directly fed into the model, primarily by concatenating them in the order of the variables.

### Quick Demo

See [demo.ipynb](https://github.com/OpenEarthLab/ORCA-DL/blob/main/demo.ipynb)

Please note that ORCA-DL initially uses [GODAS](https://psl.noaa.gov/data/gridded/data.godas.html) data as its starting point. Initialization with other data is also feasible, but it is necessary to ensure that the data is interpolated to the correct longitude and latitude range and resolution, as in [example_data](https://github.com/OpenEarthLab/ORCA-DL/blob/main/example_data). Here is an example of how to interpolate the original data using [CDO](https://code.mpimet.mpg.de/projects/cdo) (recommended):

```bash
wget https://downloads.psl.noaa.gov/Datasets/godas/sshg.1980.nc -O sshg-1980.nc   # download a 2D data from GODAS
cdo -b f64 remapbil,grid sshg-1980.nc sshg-1980-processed.nc

wget https://downloads.psl.noaa.gov/Datasets/godas/salt.1980.nc -O salt-1980.nc   # download a 3D data from GODAS
cdo -b f64 remapbil,grid salt-1980.nc tmp1.nc
cdo intlevel,10,15,30,50,75,100,125,150,200,250,300,400,500,600,800,1000 tmp1.nc tmp2.nc
cdo setzaxis,zaxis.txt tmp2.nc salt-1980-processed.nc
rm tmp1.nc tmp2.nc
```

After the data interpolation is completed, you can refer to the [demo.ipynb](https://github.com/OpenEarthLab/ORCA-DL/blob/main/demo.ipynb) to run ORCA-DL.

> **Importantly**
> You need to unify the units before using our statistics to normalize the data. See [demo.ipynb](https://github.com/OpenEarthLab/ORCA-DL/blob/main/demo.ipynb).

### Train a new model

First, download and organize the training, validation (optional), testing (optional) data, as shown below.

```
YOUR_CMIP_DATA_DIR/  # for training
    BCC-CSM2-MR/
        so/
            1850_1.npy
            ...
        thetao/
            ...
        ...
    CAS-ESM2-0/
        ...

YOUR_SODA2_DATA_DIR/  # for validation
    so/
        1850_1.npy
        ...
    thetao/
        ...
    ...

YOUR_ORAS5_DATA_DIR/  # for validation
    same as SODA2

YOUR_GODAS_DATA_DIR/  # for testing
    same as SODA2
```

Then, replace your corresponding dir path in `train.sh` and run `bash train.sh` in the command line.

After training, you can run `bash predict.sh` to make ensemble prediction using GODAS data. You can also refer to `demo.ipynb` for a more straightforward way to make predictions.

> **Note**
> We use Fully Sharded Data Parallel (FSDP) to accelerate training. With four NVIDIA A100 GPUs, the training process consumes approximately 36 GB of GPU memory per GPU. As the number of GPUs increases, the memory required per GPU decreases, and conversely, fewer GPUs result in higher memory usage per GPU. The testing process consumes approximately 12 GB of GPU memory on a single GPU.
> Training takes approximately 12 hours, while testing takes about 10 minutes (only saving tos).

## 📋 Updates

- **2025-07-28:** Model predictions (initialized with GODAS) and training data and code are released.
- **2025-03-21:** Data preprocessing processes are released.
- **2025-03-04:** Model weights and demo code are released.

## 📄 Citation

**If you find this work useful, please cite our paper:**

```
@article{guo2025data,
  title={Data-driven global ocean modeling for seasonal to decadal prediction},
  author={Guo, Zijie and Lyu, Pumeng and Ling, Fenghua and Bai, Lei and Luo, Jing-Jia and Boers, Niklas and Yamagata, Toshio and Izumo, Takeshi and Cravatte, Sophie and Capotondi, Antonietta and Ouyang, Wanli},
  journal={Science Advances},
  volume={11},
  number={33},
  pages={eadu2488},
  year={2025},
  publisher={American Association for the Advancement of Science}
}
```
