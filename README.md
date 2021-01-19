# Eagle: Efficient and Agile Performance Estimator and Dataset

## Introduction
We release a tool (Eagle) for measuring and predicting performance of neural networks running on hardware platforms.
A dataset is also released to include latency of NAS-Bench-201 models running on a broad range of devices in the desktop, embedded and mobile domains.
The tool and dataset aim to provide reproducibility and comparability in hardware-aware NAS and to ameliorate the need for researchers to have access to these devices.

In this README, we provide:
- [How to use the dataset](#Latency-dataset)
- [How to measure performance of neural networks on devices](#Device-measurement)
- [How to use predictor to estimate performance](#Performance-predictor)
- [Reproducing NAS results from the paper](#Reproducing-NAS-results-from-the-paper)

**If you have any questions, please open an issue or email us.** (last update: 13.01.2021)

## How to Use

Note: please use `Python >= 3.6.0` and `PyTorch >= 1.4.0` and `Tensorflow >= 1.13.0 < 2.0`.

You can type `pip install eagle` to install our tool (optionally with the `-e` argument for in-place installation).
We recommend using [`pipenv`](https://pypi.org/project/pipenv/) (or some other virtual environment management tool) to avoid problems with TF/Pytorch versions.

> **Note:** Although TF is currently being installed automatically by the `setup.py` script, it is only used when performing latency measurements (predictor training is done in Pytorch). Therefore, in case TF versioning is a problem and running models on-device is not required, it should be possible to remove the TF dependency from the setup script and still be able to train and evaluate predictors using our provided measurements dataset.

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

```python
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

Use `--model` to specify the type of neural models to benchmark. Use `--device` to specify the target device and `--metric` to specify what metric to measure, extra argument are provided through a configuration file passed via `--cfg` argument. Type `--help` for more details. To find relevant values:
 - `--model` should match a submodule under `eagle.models`, e.g. `nasbench201`
 - `--device` should match a submodule under `eagle.device_runner`, e.g. `desktop`
 - `--metric` should be one of the metrics supported by the device, those are reported by the submodule's top-level function called `get_supported_metrics`, e.g. for `desktop` it is defined in `eagle/device_runner/desktop/runner.py` and currently contains `"latency"` only.

The extra configuration passed through a config file depends on the implementation details of a specific model family/device runner etc. The values are usually related to the arguments of the relevant constructors and functions. For example, for the `desktop` device, the arguments in the config file, under the `device:` entry, will be passed directly to the related `DesktopRunner` class.

Here is an example. At project root:
```
python3 -m eagle.device_runner --model nasbench201 --device desktop --metric latency --cfg configs/measurements/desktop_nasbench201_local.yaml
```

After the measurement is finished, you can use the following tools to process the measurement files:
1. `combine.py` - merge multiple pickle files (each contains measurements of 1000 models) into one file
2. `calc_stats.py` - for each model, remove outliers among the measurements and store the averaged value
3. optionally, you can normalize values in your dataset - `normalize.py` does a simple linear normalization

### Performance predictor

There are two types of predictor - Graph Convolutional Networks (GCN) and Multilayer Perceptron (MLP). You can specify which one to use by the `--predictor` argument. Use `--model` to specify the type of neural models targeted by the performance predictor. Use `--device` and `--metric` to specify the target device and related metric of the performance predictor and provide the configuration via `--cfg` argument. Use `--measurement` to provide the files containing measured performance for training the predictor. Type `--help` for more details. Similarity to measuring on-device metrics, the `--model` argument should match a model family under `eagle/models` and the `--predictor` argument should match an implementation under `eagle/predictors`. Extra arguments to the predictor class can be passed under the `predictor:` entry in the config file.

> **Note:** Although `--device` and `--metric` are required just like when performing measurements, when training a predictor their values are only used to determine the output folder for storing/loading the predictor and/or evaluation results. Therefore, it is possible to pass any values (as long as they can be used to create directories in the underlying file system) not only those which match relevant implementations under `eagle/device_runner`. For example, when training an accuracy predictor we often pass arguments like `--device none --metric accuracy`.

Here is an example. At project root:
```
python3 -m eagle.predictors --predictor gcn --model nasbench201 --device desktop --metric latency --cfg configs/predictors/desktop_gen_nasbench201.yaml --measurement results/nasbench201/latency/desktop/results.pickle
```

Add `--save <name>` if you want to save your trained predictor.

## Reproducing NAS results from the [paper](https://arxiv.org/pdf/2007.08668.pdf)
First, make sure everything is set up correctly as described in the sections above.
You will also need accuracy dataset (in a format similar to latency datasets) - it can be downloaded from the same [Google Drive]() as others.
The accuracy datasets are all subsets of relevant full datasets (e.g., NAS-Bench-201) adapted to make them compatible with our training pipeline (common for latency and accuracy prediction).
Additionally, if you are interested in latency-constrained NAS, make sure that relevant latency measurements and a pre-trained predictor are also present.

### NAS-Bench-201
Look for `nasbench201_valid_acc_cifar100*` in the google drive (when running our experiments, we used normalized dataset, but from our observations this does not change results noticeably). To train a binary predictor with iterative sampling (our best result) run:

```bash
python3 -m eagle.predictors --predictor gcn --model nasbench201 --device none --metric accuracy --measurement <your_path>/nasbench201_valid_acc_cifar100_norm.pickle --cfg configs/predictors/acc_gcn_nasbench201_bin.yaml --iter 5 --sample_best2 --log
```

You can replace binary predictor with a standard GCN predictor by changing the config file to `acc_gcn_nasbench201.yaml`.
There are also relevant files for sigmoid activation (hard and soft), and trainings with a maximum of 50 and 25 models (all postfixed accordingly).

You can turn iterative data selection on and off (and also control the number of iterations) by changing or removing the `--iter` argument.
The `--sample_best2` argument turns on sampling of the best models after each iteration (alpha=0.5), without it models are sampled randomly (alpha=0).

> **Note:** There is also `--sample_best` option available which does sampling of the best models while discarding low-performing ones but from our observations this does not yield good results.

`--log` and `--exp` arguments are explained [later](#Reading-the-log-file).

#### Working with a latency predictor

To perform transfer learning from a latency predictor, use a command line similar to:
```bash
python3 -m eagle.predictors --predictor gcn --model nasbench201 --device none --metric accuracy --cfg configs/predictors/acc_gcn_nasbench201.yaml --measurement <your_path>nasbench201_valid_acc_cifar100_norm.pickle --iter 5 --sample_best2 --transfer results/nasbench201/latency/desktop_gpu/gcn/predictor_latencies_desktop_gpu_900pts_best.pt --warmup 50 --log --exp gcn_acc_from_desktop_gpu
```

> **Note:** although it is technically possible to perform transfer learning from a unary predictor to binary, from our observations this does not help the binary predictor much.

To run a latency-constrained search, use the following:
```bash
python3 -m eagle.predictors --predictor gcn --model nasbench201 --device none --metric accuracy --cfg configs/predictors/acc_gcn_nasbench201_bin.yaml --measurement ~/data/eagle/nasbench201_valid_acc_cifar100_norm.pickle --iter 5 --sample_best2 --transfer results/nasbench201/latency/desktop_gpu/gcn/predictor_latencies_desktop_gpu_900pts_best.pt --warmup 50 --log --exp gcn_acc_from_desktop_gpu_lat_lim_5ms --lat_limit 0.005
```

> **Note:** if your latency predictor outputs normalized values, you should handle denormalization yourself - the predictor is used in the constructor of `EagleDataset` in `eagle/dataset.py` (at the moment of writing at line 82).

#### Reading the log file

The `--log` argument saves the final evaluation to a log file which can be later used to simulate NAS, the log will be saved in a directory: \
`results/<model>/<metric>/<device>/<predictor>/log.txt`.
 \
You can add `--exp <name>` argument to the command line to have your results saved using the provided experiment name rather than the generic `log.txt`.

The content of the log file is split into three parts, divided by a single line containing `---`:
 - the first part contains information about ground-truth and predicted rankings; each row contains information about a single model and there is as many rows as there are models in the search space. Specifically, each row contains three fields: `GT_accuracy predicted_accuracy model_id`.
 > **Note:** for binary predictor, predicted accuracy is replaced with a model's position in the predicted ranking, where `1` is the least accurate model (to maintain compatibility with accuracy, i.e. higher number == better model).

 > **Note:** the rows are not sorted in any meaningful way.
 - the second part is a list of model identifiers (one per line) which were used to train the predictor, these are stored according to how early during the training process they were selected (mostly meaningful for iterative training)
 - the third part contains some information about how well a predictor did during the final evaluation - its form depends on the predictor's type, e.g. for a binary predictor it contains information about how many predictions performed while doing the final sorting were correct. Although potentially interesting, this part is unused when running NAS.

#### Obtaining final NAS results

After a log file has been created, a final step we do in order to "run" NAS is to read its content and use it to extract a sequence of models that would have been trained. To do that, we: *1)* sort the first part of the log file by the predicted accuracy of models (larger numbers first), *2)* read the second part of the log file and remove the models listed within it from the sorted ranking, *3)* combine the two sequences, starting with the one from step 2.
This results is a "search trace" which we later feed into our generic NAS toolkit, the toolkit selects models according to the provided search trace, querying a standard NAS-Bench-201 dataset to obtain accuracy values. If latency predictor was used, that's also the moment when we check the actual latency of each model and filter out false positives.

> **Note:** when training a predictor we use a fixed 1:1 mapping between architectures and their accuracy (e.g., see values stored in `nasbench201_valid_acc_cifar100_norm.pickle`), however, after a search trace is obtained, the final evaluation is done using multiple SGD seeds present in a relevant NAS benchmark. It is important to realize that because of that a search trace alone is not enough to assess the final NAS performance (although, it might be a good approximation). Also, we did not study how training a predictor on data affected by SGD noise would impact the final results - however, we anticipate that they would remain close to the current ones, please let us know if you find this to be incorrect.


### NAS-Bench-101
Coming soon

### DARTS
Coming soon

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
