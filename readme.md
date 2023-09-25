# Torch Trainer
Torch Trainer is a tool that will help you to train you PyTorch Model faster without having to write all the training steps one by one. It also come s with a colored progress bar for better visualization.




# Getting Started
###  == Prerequisites ==
Python 3.6 - 3.11
tqdm
Torch

 1. Clone this repository (install [git](https://git-scm.com/downloads) if haven't installed it yet):
 `git clone https://github.com/Nahid180/PyTorch_Assets.git`
 2. Move `TorchTrainer.py` to your project directory.
 3. Install tqdm (if you don't have it)
	 

    pip install tqdm



### Arguments

|Argument  |Type  |Description	|	
|--|--|--|
|  model|   required| The model you created (CNN) |
|dataset|required|Train dataset|
|  device|   required| Can be either "cuda or "cpu" |
|  epochs|   required| number of times you want to run the training loop |
|  test_dataset|   optional| Test dataset (good if you use) default is "None"|
|  Optimizer|   optional| Optimizer algorithm. Default is "SGD" |
|  learning_rate|   optional| Default is "0.01" |




