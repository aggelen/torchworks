### TorchWorks

- Easily code and run pytorch-based experiments with your own models!
- Torchworks is a framework for simplify training, validation, and testing processes with many networks and losses.
- It was designed to reduce code mess and keep works as experiments.

####Javascript　

```python
class Experiment:
    def __init__(self, network, experiment_params):
        self.network = network
        
        self.no_epoch = experiment_params['no_epoch']

        #Optimizer
        if experiment_params['optimizer'] == 'default':
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-4, betas=(0.9,0.999))
        else:
            self.optimizer = experiment_params['optimizer']

        self.loss = experiment_params['loss']
        self.data_loader = experiment_params['data_loader']

```