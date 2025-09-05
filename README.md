# Clients Collaborate: Flexible Differentially Private Federated Learning with Guaranteed Improvement of Utility-Privacy Trade-off (FedCEO at ICML 2025, CCF A, CORE A*)

<x update>
  
> 📣 02/08/25: Update the _Slide_ and _Video_ in ICML 2025!

> 📣 01/05/25: This paper has been accepted to **ICML 2025**!

The official implementation of our paper:

[Clients Collaborate: Flexible Differentially Private Federated Learning with Guaranteed Improvement of Utility-Privacy Trade-off](https://arxiv.org/pdf/2402.07002) (**FedCEO**)

[[OpenReview](https://openreview.net/forum?id=C7dmhyTDrx&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICML.cc%2F2025%2FConference%2FAuthors%23your-submissions))] [[Slide&Video](https://icml.cc/virtual/2025/poster/46080)]


## Dependence

To install the dependencies: `pip install -r requirements.txt`.

## Data

The EMNIST and CIFAR10 datasets are downloaded automatically by the `torchvision` package.

## Usage

We provide scripts that have been tested to produce the results stated in our paper (utility experiments and privacy experiments).
Please find them in the file: `train.sh`.

## Flags
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

## Citation  

```BibTex
@article{li2024clients,
  title={Clients collaborate: Flexible differentially private federated learning with guaranteed improvement of utility-privacy trade-off},
  author={Li, Yuecheng and Wang, Tong and Chen, Chuan and Lou, Jian and Chen, Bin and Yang, Lei and Zheng, Zibin},
  journal={arXiv preprint arXiv:2402.07002},
  year={2024}
}
```



