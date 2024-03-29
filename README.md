### TorchWorks

- Easily code and run pytorch-based experiments with your own models!
- Torchworks is a framework for simplify training, validation, and testing processes with many networks and losses.
- It was designed to reduce code mess and keep works as experiments.

#### TODO List
- [ ] Show network structure
- [ ] Generalize dual optimizer training to multiple optimizers
- [ ] Improve test function
- [ ] Save/Load/Summarize experiments (not network checkpoints!)

#### Example Architecture For an Experiment

An experiment is a simple folder that holds classes in seperate files. Please see experiments/experiment1.


#### Run an Experiment

To create and run an experiment use following example.

```python
from torchworks import Experiment
exp1 = Experiment(exp_path='experiments/experiment1')
exp1.train()
exp1.plot_loss_hist()
```

#### New experiments
Torchworks experiment initializer is a cli tool designed to make it easy to create a new experiment. After installing the package, you can create a new experiment with torchworks template.

```bash
init_torchworks_experiment --exp_name some_experiment --model_name BestModel --dataset_name FancyDataset --loss_name TWLoss
```
