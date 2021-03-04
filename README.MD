# Regression for Blast Loads

### TODO

- [x] Repo setup
- [x] Draft model implementation
- [ ] Evaluation metrics implementation
- [ ] Code test
- [ ] Initial Result
- [ ] Tuning Parameters



### Introduction
This is an implementation of regressing blast loads behind walls based on Fully Connected layer.  You can specify arbitrary number of hidden layers and  neurons without modifying the model directly.



### Installation

The code was tested with Anaconda and Python 3.7. After installing the Anaconda environment:

0. Clone the repo:
    ```Shell
    git clone https://github.com/angsa-robotics/angsa-avoidance.git
    cd angsa-avoidance
    ```

1. Install dependencies:

    For PyTorch dependency, see [pytorch.org](https://pytorch.org/) for more details.

    For custom dependencies:
    ```Shell
    pip install tensorboardX
    ```



### Dataset Preparation

Raw data is provided from [Google Drive](https://drive.google.com/file/d/1jx7buZuncWzjPvButWMLGXrbaZXRnH7F/view?usp=sharing). One should download it as CSV form to raw_data folder. Then, using the following command to transform the raw data to the desired form, which is basically removing unnecessary cells and saving them as ***.txt** file

```shell
python scripts/data_transform.py
```

After execution, you'll get two transformed data file in raw_data folder, ended with "_transformed.txt" (**Already done!**)

### Training

Follow steps below to train your model:

0. Configure your train/test argument  in **scripts/train.sh** 

1. Arguments explanation

2. Usage
    ```Shell
    sh scripts/train.sh 
    
    ```
