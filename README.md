[//]: # (\usepackage{ulem}[//]: # &#40;<p align="center">&#41;)

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

[//]: # ()
[//]: # (### Python version )

[//]: # (Python 3.9.)

## Environment

### Environment Setup
To use DQE, start by creating and activating a new Conda environment using the following commands:

```bash
conda create --name dqe_env python=3.9
conda activate dqe_env
```

[//]: # (### Installation)

[//]: # (Install DQE for immediate use in your projects:)

[//]: # ()
[//]: # (```bash)

[//]: # (pip install -r requirements.txt)

[//]: # (```)


### Install Dependencies

Install the required Python packages via:

```bash
pip install -r requirements.txt
```



## How to use DQE? 

Begin by importing the DQE module in your Python script:


```bash
from dqe.dqe_metric import DQE
```

Prepare your input as arrays of anomaly scores (continues or binary) and binary labels. 

Example usage of DQE:

```bash
dqe = DQE(labels, scores)
```

### Basic Example

```python 
import numpy as np
from dqe.dqe_metric import DQE

# Example data setup
labels = np.array([0, 1, 0, 1, 0])
scores = np.array([0.1, 0.8, 0.1, 0.9, 0.2])

# Compute DQE

dqe = DQE(labels, scores)

print(dqe)
```

DQE allows for comprehensive customization of parameters by parameter `parameter_dict`.
Please refer to the main code documentation for a full list of configurable options.

---

## Experiments
For researchers interested in reproducing the experiments or exploring the evaluation metric further with various data sets:


[//]: # (```bash)

[//]: # ()
[//]: # (git clone https://github.com/raminghorbanii/DQE)

[//]: # ()
[//]: # (cd DQE)

[//]: # ()
[//]: # (pip install -r requirements.txt)

[//]: # ()
[//]: # (```)

[//]: # (## Conducting Experiments)

### with Synthetic Data

To run experiments on synthetic data, navigate to the `experiments` directory and execute the Python script `synthetic_data_exp.py`.
This script allows for the modification of various scenarios, comparing DQE against other established metrics.


```bash
python synthetic_data_exp.py --exp_name "over-counting tp"
```

The parameter `exp_name` can be set to one of the following values: 
["over-counting tp", "tp timeliness", "fp proximity", "fp insensitivity or overevaluation of duration", "fp duration overevaluation (af)"].


### with Real-World Data

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

After downloading, place the unzipped dataset in the directory `dataset`. If you store the data in a different location, ensure you update the directory paths in the code to match.

[//]: # (Navigate to the experiments/RealWorld_Data_Experiments directory to run an experiment. )

[//]: # (Navigate to the experiments/RealWorld_Data_Experiments directory to run an experiment. )
 Navigate to the `experiments` directory and execute the Python script `get_algorithms_outputs.py` for producing algorithms' outputs by entering the following command:

```bash
python get_algorithms_outputs.py
```

Execute the Python script `real_data_exp_case.py` for producing case results in paper by entering the following command:

[//]: # (Execute one of the example Python scripts by entering the following command:)

```bash
python real_data_exp_case.py --exp_name "YAHOO case"
```
The parameter `exp_name` can be set to one of the following values: ["YAHOO case", "WSD case", "partition strategy", "detection rate", "weighting strategy"].

Execute the Python script `real_data_exp_all_dataset.py` for producing average results in paper by entering the following command:


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

