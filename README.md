# FedCEO-Collaborate-with-Each-Other

This repository contains the code for our paper:

Clients Collaborate: Flexible Differentially Private Federated Learning with Guaranteed Improvement of Utility-Privacy Trade-off


## Dependence

To install the dependencies: `pip install -r requirements.txt`.

## Data

The EMNIST and CIFAR10 datasets are downloaded automatically by the `torchvision` package.

## Usage

We provide scripts that has been tested to produce the results stated in our paper (utility experiments and privacy experiments).
Please find them under the file: `train.sh`.

In the following, we explain several important options.

### Explanation of flags
- FL related

  - `args.epochs`: The number of communication rounds.
  - `args.num_users`: The number of total clients, denoted by $N$.
  - `args.frac`: The sampling rate of clients, denoted by $p$.
  - `args.lr`: The learning rate of local round on the clients, denoted by $\eta$.
  - `args.privacy`: Adopt the DP Gaussian mechanism or not.
  - `args.noise_multiplier`: The ratio of the standard deviation of the Gaussian noise to the L2-sensitivity of the function to which the noise is added.
  - `args.flag`: Using our low-rank processing or not.
- FedCEO related

    - `args.lamb`: The weight of regularization term, denoted by $\lambda$.
    - `args.interval`: The smoothing interval to adopt, denoted by $I$.
    - `args.flag`: The common ratio of the geometric series, denoted by $\vartheta$.
- Model related

  - `args.model`: MLP or LeNet.
- Experiment setting related

  - `args.dataset`: cifar10 or emnist.
  - `args.index`: The index for leaking images on Dataset.





