# A Reinforcement Learning Approach to Edge Suggestion for Fair Information Access on Social Networks
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
### Dataset
Our repository contains four publicly available network datasets: the synthetic Antelope Valley network, three real-world networks,
Email-Eu, Epinions, and Facebook, downloaded from the SNAP collection https://snap.stanford.edu/data/

## Experiments
The run_experiments.py generates all results for FIAES-RL

On each of the outputted graphs, we run 1,000 Monte-Carlo simulations under the IC model and evaluate our main two criteria:

1. Influence Boost Ratio
2. Equity Score

Train the FIAES-RL model and evaluate it across different datasets:
```python run_experiments.py```
