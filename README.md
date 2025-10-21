\usepackage{ulem}[//]: # (<p align="center">)

[//]: # (<img width="300" src="https://raw.githubusercontent.com/Raminghorbanii/DQE/master/docs/DQE_logo.png"/>)

[//]: # (</p>)


<h1 align="center">DQE: A Dual-Quality Evaluation Metric for Time Series Anomaly Detection</h1>

[//]: # (<p align="center">)

[//]: # (  <a href="https://kdd.org/kdd2024/">)

[//]: # (    <img src="https://img.shields.io/badge/ACM%20KDD%202024-Accepted-blue.svg" alt="ACM KDD 2024 Accepted">)

[//]: # (  </a>)

[//]: # (  <a href="https://arxiv.org/abs/2405.12096">)

[//]: # (    <img src="https://img.shields.io/badge/Preprint version-Arxiv-green.svg" alt="Preprint Version">)

[//]: # (  </a>)

[//]: # (</p>)

This repository contains the code for DQE (Dual-Quality Evaluation Metric), a novel evaluation metric for assessing anomaly detection in time series data. 
DQE is designed around the principles of locality and
integrality, which holistically synthesizes true positive quality (TQ) and true positive quality (FQ), enabling a fine-grained evaluation of timely, early and late detections.
The methodology is detailed in our paper, demonstrating that DQE provides more reliable and
discriminative evaluations through experiments with both synthetic and real-world datasets.


[//]: # (## Quick Start)

### Installation
Install DQE for immediate use in your projects:

```bash
pip install requirements.txt
```

## How to use DQE? 

Begin by importing the DQE module in your Python script:


```bash
from meata import meata_auc_pr
from config.meata_config import parameter_dict

```

Prepare your input as arrays of anomaly scores (continues or binary) and binary labels. DQE allows for comprehensive customization of parameters. 

Please refer to the main code documentation for a full list of configurable options.

Example usage of DQE:

```bash
    final_dqe, dqe, dqe_w_gt, dqe_w_near_ngt, dqe_w_distant_ngt, dqe_w_ngt = meata_auc_pr(labels,output=pred,parameter_dict=parameter_dict,cal_mode="proportion")
```

### Basic Example

```python 
import numpy as np
from meata import meata_auc_pr
from config.meata_config import parameter_dict

# Example data setup
labels = np.array([0, 1, 0, 1, 0])
scores = np.array([0.1, 0.8, 0.1, 0.9, 0.2])

# Compute DQE

final_dqe, dqe, dqe_w_gt, dqe_w_near_ngt, dqe_w_distant_ngt, dqe_w_ngt = meata_auc_pr(labels,output=scores,parameter_dict=parameter_dict,cal_mode="proportion")

print(dqe)
```

---

## Advanced Setup and Experiments
For researchers interested in reproducing the experiments or exploring the evaluation metric further with various data sets:


### Environment Setup
To use DQE, start by creating and activating a new Conda environment using the following commands:

```bash
conda create --name dqe_env python=3.9
conda activate dqe_env
```

[//]: # (### Install Dependencies)

[//]: # (Install the required Python packages via:)

[//]: # ()
[//]: # (```bash)

[//]: # (git clone https://github.com/raminghorbanii/DQE)

[//]: # (cd DQE)

[//]: # (pip install -r synthetic_exp_requirements.txt)

[//]: # (```)

## Conducting Experiments

### with Synthetic Data

To run experiments on synthetic data, execute the Python script main_synthetic_data_exp.py.
This script allows for the modification of various scenarios, comparing DQE against other established metrics.


```bash
python main_synthetic_data_exp.py
```

[//]: # (Example of how you use DQE using synthetic data &#40;Binary detector&#41;:)

[//]: # (```python)

[//]: # ()
[//]: # (from utils_Synthetic_exp import evaluate_all_metrics, synthetic_generator)

[//]: # ()
[//]: # (label_anomaly_ranges = [[40,59]] # You can selec multiple ranges for anomaly. Here we selected one range with the size of 20 points &#40;A_k&#41; )

[//]: # (predicted_ranges = [[30, 49]]  # You can selec multiple ranges for predictions. Here we selected the range the same as Scenario 2, proposed in the original paper. )

[//]: # (vus_zone_size = e_buffer = d_buffer = 20 )

[//]: # ()
[//]: # (experiment_results = synthetic_generator&#40;label_anomaly_ranges, predicted_ranges, vus_zone_size, e_buffer, d_buffer&#41;)

[//]: # (predicted_array = experiment_results["predicted_array"])

[//]: # (label_array = experiment_results["label_array"])

[//]: # ()
[//]: # ()
[//]: # (score_list_simple = evaluate_all_metrics&#40;predicted_array, label_array, vus_zone_size, e_buffer, d_buffer&#41;)

[//]: # (print&#40;score_list_simple&#41;)

[//]: # ()
[//]: # ()
[//]: # (```)


[//]: # (```bash)

[//]: # ()
[//]: # (Output:)

[//]: # ()
[//]: # ('original_F1Score': 0.5,)

[//]: # ('pa_precision': 0.67,)

[//]: # ('pa_recall': 1.0,)

[//]: # ('pa_f_score': 0.8,)

[//]: # ('Rbased_precision': 0.6,)

[//]: # ('Rbased_recall': 0.6,)

[//]: # ('Rbased_f1score': 0.6,)

[//]: # ('eTaPR_precision': 0.75,)

[//]: # ('eTaPR_recall': 0.75,)

[//]: # ('eTaPR_f1_score': 0.75,)

[//]: # ('Affiliation precision': 0.97,)

[//]: # ('Affiliation recall': 0.99,)

[//]: # ('Affliation F1score': 0.98,)

[//]: # ('VUS_ROC': 0.79,)

[//]: # ('VUS_PR': 0.72,)

[//]: # ('AUC': 0.74,)

[//]: # ('AUC_PR': 0.51,)

[//]: # ()
[//]: # ('DQE': 0.76,)

[//]: # ('DQE-F1': 0.75})

[//]: # ()
[//]: # (```)

[//]: # (### with Real-World Data)

[//]: # (For real-world data experiments, ensure all additional required packages are installed.)

[//]: # ()
[//]: # (```bash)

[//]: # (pip install -r Real_exp_requirements.txt)

[//]: # (```)

#### Download the Dataset
The real-world datasets for experiments can be downloaded from the following link:

Dataset Link: https://www.thedatum.org/datasets/TSB-AD-U.zip 

Ref: This dataset is made available through the GitHub page of the project "The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark (TSB-AD)": https://github.com/TheDatumOrg/TSB-AD

#### Running the Experiments

After downloading, place the unzipped dataset in the same directory. If you store the data in a different location, ensure you update the directory paths in the code to match.

[//]: # (Navigate to the experiments/RealWorld_Data_Experiments directory to run an experiment. )

[//]: # (Navigate to the experiments/RealWorld_Data_Experiments directory to run an experiment. )
Execute the Python script real_data_exp_case.py for producing case results in paper by entering the following command:)

[//]: # (Execute one of the example Python scripts by entering the following command:)

```bash
python real_data_exp_case.py
```

Execute the Python script real_data_exp_all_dataset.py for producing average results in paper:


```bash
python real_data_exp_all_dataset.py
```

[//]: # (Two different examples are provided. These examples allow for modifications and customizations, enabling detailed exploration of various data aspects.)


[//]: # (---)

[//]: # ()
[//]: # (## Setting Buffer Size in DQE)

[//]: # ()
[//]: # (Given the context of time series data, selecting a buffer size for a fair evaluation of anomaly detectors' performance is unavoidable. The buffer parameter of DQE can be set using the following strategies:)

[//]: # ()
[//]: # (- *Expert Knowledge*: Best suited for customized, specific, and real-world applications where expert knowledge is available, or when one has enough experience with the data at hand. Experts can directly specify buffer sizes that are optimized for the particular use case.)

[//]: # ()
[//]: # (- *ACF Analysis*: Automatically determines the optimal buffer size by analyzing the autocorrelation within the data. This function is available in DQE_utils.py.)

[//]: # ()
[//]: # (- *Range of Buffer Sizes*: DQE is flexible and can evaluate performance across all combinations of pre and post buffer sizes, allowing for a comprehensive assessment without expert input. One can start with a maximum buffer size, and DQE automatically divides it into a specified number of ranges &#40;determined by the user&#41;.)

[//]: # ()
[//]: # (- *Default Setting*: Utilizes the input window size of the anomaly detector, a standard, practical buffer size that aligns with the general scale of the data being analyzed. This option is useful when no specific adjustments are needed or when minimal configuration is desired.)

[//]: # ()
[//]: # (This guidance ensures that you can effectively implement these buffer size selection strategies in DQE for optimal results.)

[//]: # ()
[//]: # ()
[//]: # (---)

[//]: # (## Citation)

[//]: # (If you find our work is useful in your research, please consider raising a star  :star:  and citing:)

[//]: # ()
[//]: # (```)

[//]: # (@article{ghorbani2024DQE,)

[//]: # (  title={DQE: Proximity-Aware Time series anomaly Evaluation},)

[//]: # (  author={Ghorbani, Ramin and Reinders, Marcel JT and Tax, David MJ},)

[//]: # (  journal={arXiv preprint arXiv:2405.12096},)

[//]: # (  year={2024})

[//]: # (})

[//]: # (```)

