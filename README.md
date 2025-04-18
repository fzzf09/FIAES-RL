# FIAES-RL:Toward Fairness in Information Access on Social Networks via Edge Suggestion
This repository hosts the implementation code for the "A Reinforcement Learning Approach to Edge Suggestion for Fair
Information Access on Social Networks" as introduced in our research paper. 

## Prerequisites

### Software
Install dependencies by running
```
pip install -r requirements.txt
```
The code was tested on 
```
cuda 11.2
tensorflow==2.9.1
keras==2.9.0
```

### Hardware
We ran on a Nvidia RTX 3090 with a 14-core Intel Xeon(R) Platinum 8362 processor @3.6GHz, 64GB RAM. 

### Dataset
Our repository contains four publicly available network datasets: the synthetic Antelope Valley network, three real-world networks,
Email-Eu, Epinions, and Facebook, downloaded from the SNAP collection https://snap.stanford.edu/data/

## Experiments
The run_experiments.py generates all results for: 
1. FIAES-RL
2. Baseline method

On each of the outputted graphs, we run 1,000 Monte-Carlo simulations under the IC model and evaluate our main two criteria:

1. Influence Boost Ratio
2. Equity Score

Train the FIAES-RL model and evaluate it alongside other baselines across different datasets:

```python run_experiments.py```
