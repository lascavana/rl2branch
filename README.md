# Learning to branch with Tree MDPs

Lara Scavuzzo, Feng Yang Chen, Didier Chételat, Maxime Gasse, Andrea Lodi, Neil Yorke-Smith, Karen Aardal

Official implementation of the paper *Learning to branch with Tree MDPs*.

## Installation

See installation instructions [here](INSTALL.md).

## Running the experiments

For a given TYPE in {setcover, cauctions, indset, ufacilities, mknapsack}, run the following to reproduce the experiments
```
# Generate MILP instances
python 01_generate_instances.py $TYPE

# Get train instance solutions
python 02_get_instance_solutions.py $TYPE -j 8    # number of parallel threads

# Generate supervised learning datasets
python 03_generate_il_samples.py $TYPE -j 8  # number of parallel threads

# Training supervised learning model
python 04_train_il.py $TYPE -g 0    # GPU id

# Training reinforcement learning learning models
python 05_train_rl.py $TYPE mdp -g 0    
python 05_train_rl.py $TYPE tmdp+DFS -g 0
python 05_train_rl.py $TYPE tmdp+ObjLim -g 0

# Evaluation
python evaluate.py $TYPE -g 0
```
Optional: run steps 4 and 5 with flag `--wandb` to log the training metrics using wandb. This requires a wandb installation, an account and the appropriate projects.

## Questions / Bugs
Please feel free to submit a Github issue if you have any questions or find any bugs. We do not guarantee any support, but will do our best if we can help.
