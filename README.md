### TorchWorks

- Easily code and run pytorch-based experiments with your own models!
- Torchworks is a framework for simplify training, validation, and testing processes with many networks and losses.
- It was designed to reduce code mess and keep works as experiments.

#### TODO List
- [x] Show network structure
- [ ] Generalize dual optimizer training to multiple optimizers
- [ ] Improve test function
- [ ] Save/Load/Summarize experiments (not network checkpoints!)

#### Example Architecture For an Experiment

All required classes can be collected into a single file.

```python
class ExampleDataset(torch.utils.data.Dataset):
    def __init__(self, params):
        print('>> Preparing Dataset ...')
        self.data_len = params['data_len']

        print('>> Dataset created!')

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):

        item = None
        return item

    def create_data_loader(self, batch_size):
        self.data_loader = torch.utils.data.DataLoader(self,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 # num_workers=8,
                                                 # pin_memory=True,
                                                 )
```

```python
#%% Network
class ExampleNetwork(nn.Module):
    def __init__(self):
        super(ExampleNetwork, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, 3),
                                   nn.BatchNorm2d(64),
                                   nn.PReLU())

    def forward(self, x):
        y = self.conv1(x)

        # The forward method should return a dict.
        return {'y': y}
```
```python
#%% Loss
class ExampleLoss(nn.Module):
    def __init__(self):
        super(ExampleLoss, self).__init__()

    def forward(self, estimated, target):
        # The forward method takes estimated and target parameters.
        loss = None
        return loss
```

#### Run an Experiment

To create and run an experiment use following example.

```python
from TorchWorks import Experiment
from Networks.ExampleNetwork import ExampleNetwork, ExampleDataset, ExampleLoss
dataset = ExampleLoss(path='path_to_data', data_len=3200, batch_size=8)

model = ExampleNetwork()

experiment_params = {'no_epoch': 2,
                     'optimizer': 'default',
                     'loss': ExampleLoss(),
                     'data_loader': dataset.data_loader}

exp0 = Experiment(network=model, experiment_params=experiment_params)

exp0.train()
```