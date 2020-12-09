# Eagle: Efficient and Agile Performance Estimator and Dataset

## Introduction
We release a tool (Eagle) for measuring and predicting performance of neural networks running on hardware platforms.
A dataset is also released to include latency of NAS-Bench-201 models running on a broad range of devices in the desktop, embedded and mobile domains.
The tool and dataset aim to provide reproducibility and comparability in hardware-aware NAS and to ameliorate the need for researchers to have access to these devices.

In this README, we provide:
- [How to use the dataset](#Latency-dataset)
- [How to measure performance of neural networks on devices](#Device-measurement)
- [How to use predictor to estimate performance](#Performance-predictor)

**If you have any questions, please open an issue or email us.**

## How to Use

Note: please use `Python >= 3.6.0` and `PyTorch >= 1.4.0` and `Tensorflow >= 1.13.0`.

You can type `pip install eagle` to install our tool.

### Latency dataset

The latest dataset can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1lywFtYt6Y89nYbeDfqobU8TZAdFdZIbd?usp=sharing).

This archive contains the following files:
1. desktop-cpu-core-i7-7820x-fp32.pickle
2. desktop-gpu-gtx-1080-ti-fp32.pickle  
3. embedded-gpu-jetson-nono-fp16.pickle 
4. embedded-gpu-jetson-nono-fp32.pickle 
5. embeeded-tpu-edgetpu-int8.pickle    
6. mobile-cpu-snapdragon-450-contex-a53-int8.pickle
7. mobile-cpu-snapdragon-675-kryo-460-int8.pickle
8. mobile-cpu-snapdragon-855-kryo-485-int8.pickle
9. mobile-dsp-snapdragon-675-hexagon-685-int8.pickle
10. mobile-dsp-snapdragon-855-hexagon-690-int8.pickle
11. mobile-gpu-snapdragon-450-adreno-506-int8.pickle 
12. mobile-gpu-snapdragon-675-adren0-612-int8.pickle        
13. mobile-gpu-snapdragon-855-adren0-640-int8.pickle        

The latency benchmarks are stored as dictionary in the Pickle files:

```
{(arch_vector): latency value in seconds}
```

`(arch_vector)` is a tuple representing an architecture in the NAS-Bench-201 search space. It can be converted to/from the arch_str format accepted by NAS-Bench-201 API using the utility functions.

```
import pickle
from model.nasbench201 import utils

#Convert arch_str in NAS-Bench-201 to arch_vector in the pickle files
arch_vector = utils.get_arch_vector_from_arch_str(arch_str) 

latency_data = pickle.load(open('desktop-gpu-core-i7-7820x.pickle','rb'))
latency_query = latency_data[arch_vector]

#We can also convert arch_vector back to NAS-Bench-201 arch_str
arch_str = utils.get_arch_str_from_arch_vector(arch_vector) 
```

### Device measurement

Use `--model` to specify the type of neural models to benchmark. Use `--device` to specify the target device of the performance predictor and provide the configuration via `--cfg` argument. Type `--help` for more details.

Here is an example. At project root:
```
python3 -m eagle.device_runner --model nasbench201 --device desktop --cfg configs/measurements/desktop_nasbench201_local.yaml
```

After the measurement is finished, you can use the following tools to process the measurement files:
1. combine.py - merge multiple pickle files (each contains measurements of 1000 models) into one file
2. calc_stats.py - for each model, remove outliers among the measurements and store the averaged value

### Performance predictor

There are two types of predictor - Graph Convolutional Networks (GCN) and Multilayer Perceptron (MLP). You can specify which one to use by the `--predictor` argument. Use `--model` to specify the type of neural models targetted by the performance predictor. Use `--device` to specify the target device of the performance predictor and provide the configuration via `--cfg` argument. Use `--measurement` to provide the measured performance file for training the predictor. Type `--help` for more details.

Here is an example. At project root:
```
python3 -m eagle.predictors --predictor gcn --model nasbench201 --device desktop --cfg configs/predictors/desktop_gen_nasbench201.yaml -- measurement results/nasbench201/latency/desktop/results.pickle
```

## Citation

If you find that Eagle helps your research, please consider citing it:
```
@misc{2020brpnas,
    title={BRP-NAS: Prediction-based NAS using GCNs},
    author={Lukasz Dudziak and Thomas Chau and Mohamed S. Abdelfattah and Royson Lee and Hyeji Kim and Nicholas D. Lane},
    year={2020},
    eprint{2007.08668},
    archivePredix={arXiv},
    primaryClass={cs.LG}
}
```
