# Welcome to the IMAGINE repository

-----------------

## Requirements

The dependencies are listed in the requirements.txt file. Our conda environment can be cloned with:
```
conda env create -f environment.yml
```

## Notebook


We propose a [Google Colab Notebook](https://colab.research.google.com/drive/1G9LmvhbvR40XJ-cysgP6zynBnq_fHY63#scrollTo=HmGFArOeXvps) to walk you through the IMAGINE learning algorithm. 


## Demo

1. Run pre-trained weights

The demo script is /src/imagine/experiments/play.py. It can be used as such:

```python play.py```

## RL training


1. Running the algorithm

The main running script is /src/imagine/experiments/train.py. It can be used as such:

```
python train.py --num_cpu=6 --architecture=modular_attention --imagination_method=CGH --reward_function=learned_lstm  --goal_invention=from_epoch_10 --n_epochs=167
```

Note that the number of cpu is an important parameter. Changing it is **not** equivalent to reducing/increasing training time. One epoch is 600 episodes. Other parameters can be
 found in train.py. The config.py file contains all parameters and is overriden by parameters defined in train.py.
 
 Logs and results are saved in /src/data/expe/PlaygroundNavigation-v1/trial_id/. It contains policy and reward function checkpoints, raw logs (log.txt), a csv containing main metrics (progress.csv) and a json file with the parameters (params.json).
 
 2. Plotting results
 
 Results for one run can be plotted using the script /src/analyses/new_plot.py
