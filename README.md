# FedCEO: *"Clients Collaborate: Flexible Differentially Private Federated Learning with Guaranteed Improvement of Utility-Privacy Trade-off"* (ICML 2025)

<ml heart paper weekly gzh>

<mit tech>

> 📣 17/12/25: Honored to have our work featured by [PaperWeekly](https://zhuanlan.zhihu.com/paperweekly) [[博客](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247712832&idx=2&sn=6d35c7d8fca09214367ca2b63f99bfd4&chksm=975656b330afd9317064c6ea068543b42436aec686d2dbeb2c322fe7fd8044746b638f8c2d8c&mpshare=1&scene=24&srcid=1122sjo91pjKnm1C5zeNE9PR&sharer_shareinfo=dd55ea0fd95260f0f70261dc41dd6818&sharer_shareinfo_first=dd55ea0fd95260f0f70261dc41dd6818#rd)] and [Data Science Collective](https://medium.com/data-science-collective) [[Blog](https://medium.com/data-science-collective/icml-2025-the-art-of-balance-in-federated-learning-the-fedceo-framework-cracks-the-dilemma-of-326c9d8bd5fe)]！

> 📣 06/09/25: Check out the intro to our paper on [Twitter](https://x.com/Yuecheng_Lee/status/1964308641055723827)! Likes and Reposts are welcome!
  
> 📣 02/08/25: Update the _Slide_ and _Video_ in ICML 2025!

> 📣 01/05/25: This paper has been accepted to **ICML 2025**!

The official implementation of our paper:

[Clients Collaborate: Flexible Differentially Private Federated Learning with Guaranteed Improvement of Utility-Privacy Trade-off](https://arxiv.org/pdf/2402.07002) (**FedCEO**)

[[ArXiv](https://arxiv.org/abs/2402.07002)] [[OpenReview](https://openreview.net/forum?id=C7dmhyTDrx&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICML.cc%2F2025%2FConference%2FAuthors%23your-submissions))] [[Slide&Video](https://icml.cc/virtual/2025/poster/46080)] [[X](https://x.com/Yuecheng_Lee/status/1964308641055723827)] [[中文博客](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247712832&idx=2&sn=6d35c7d8fca09214367ca2b63f99bfd4&chksm=975656b330afd9317064c6ea068543b42436aec686d2dbeb2c322fe7fd8044746b638f8c2d8c&mpshare=1&scene=24&srcid=1122sjo91pjKnm1C5zeNE9PR&sharer_shareinfo=dd55ea0fd95260f0f70261dc41dd6818&sharer_shareinfo_first=dd55ea0fd95260f0f70261dc41dd6818#rd)] [[Blog](https://medium.com/data-science-collective/icml-2025-the-art-of-balance-in-federated-learning-the-fedceo-framework-cracks-the-dilemma-of-326c9d8bd5fe)]

## Abstract

To defend against privacy leakage of user data, differential privacy is widely used in federated learning, but it is not free. The addition of noise randomly disrupts the semantic integrity of the model and this disturbance accumulates with increased communication rounds. In this paper, we introduce a novel federated learning framework with rigorous privacy guarantees, named **FedCEO**, designed to strike a trade-off between model utility and user privacy by letting clients "*Collaborate with Each Other*". Specifically, we perform efficient tensor low-rank proximal optimization on stacked local model parameters at the server, demonstrating its capability to *flexibly* truncate high-frequency components in spectral space. This capability implies that our FedCEO can effectively recover the disrupted semantic information by smoothing the global semantic space for different privacy settings and continuous training processes. Moreover, we improve the SOTA utility-privacy trade-off bound by order of `\sqrt{d}`, where `d` is the input dimension. We illustrate our theoretical results with experiments on representative datasets and observe significant performance improvements and strict privacy guarantees under different privacy settings. 

<img width="1608" height="1010" alt="image" src="https://github.com/user-attachments/assets/f03c01d5-4654-4174-9230-1826dc65ee7f" />

## Lay Summary

Protecting user privacy in collaborative AI training (federated learning) requires adding carefully designed noise. However, this noise can unevenly disrupt different parts of each device's learned knowledge over time – like obscuring facial features in one device's animal recognition model while blurring limb details in another's.We introduce FedCEO, a new approach where devices "Collaborate with Each Other" under server coordination. FedCEO intelligently combines the complementary knowledge from all devices. When one device's understanding of a concept is disrupted by privacy protection, others help fill those gaps.This CEO-like coordination gradually enhances semantic smoothness across devices as training progresses. The server blends the partial understandings into a coherent whole, allowing the global model to recover disrupted patterns while maintaining privacy. The result is significantly improved AI performance across diverse privacy settings and extended training periods.

## Dependence

To install the dependencies: 

```sh
pip install -r requirements.txt
```

## Data

The EMNIST and CIFAR10 datasets are downloaded automatically by the `torchvision` package.

## Usage

We provide scripts that have been tested to produce the results stated in our paper (utility experiments and privacy experiments).
Please find them in the file: `train.sh`.

For example,

```bash
# CIFAR-10
nohup python -u FedCEO.py --privacy True --noise_multiplier 2.0 --flag True --dataset "cifar10" --model "cnn" --lamb 0.6 --r 1.04 --interval 10 > ./logs/log_fedceo_noise=2.0_cifar10_LeNet.log 2>&1 &
```


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

## 🧪 Experimental Results and Commands

We have conducted extensive experiments to validate the utility improvement, privacy protection, and utility-privacy trade-off of **FedCEO**. Below we provide partial results and the commands to reproduce them.

**Explore more experiments in the Appendices!** [[🔗Paper](https://arxiv.org/abs/2402.07002)]

### 1. Utility Experiments

We compare FedCEO with baseline methods under different privacy settings (controlled by `σ_g`). FedCEO consistently achieves the highest test accuracy.

#### Table: Test Accuracy (%) on EMNIST and CIFAR-10

| Dataset    | Model         | σ_g  | UDP-FedAvg | PPSGD  | CENTAUR | FedCEO (ϑ=1) | FedCEO (ϑ>1) |
|------------|---------------|------|-------------|---------|----------|---------------|---------------|
|      |   | 1.0  | 76.59%      | 77.01%  | 77.26%   | 77.14%        | **78.05%**    |
|  EMNIST          |    MLP-2-Layers           | 1.5  | 69.91%      | 70.78%  | 71.86%   | 71.56%        | **72.44%**    |
|            |               | 2.0  | 60.32%      | 61.51%  | 62.12%   | 63.38%        | **64.20%**    |
|    |        | 1.0  | 43.87%      | 49.24%  | 50.14%   | 50.09%        | **54.16%**    |
|    CIFAR-10        |      LeNet-5         | 1.5  | 34.34%      | 47.56%  | 46.90%   | 48.89%        | **50.00%**    |
|            |               | 2.0  | 26.88%      | 34.61%  | 36.70%   | 37.39%        | **45.35%**    |

#### Command to Run Utility Experiment:
```bash
# CIFAR-10
nohup python -u FedCEO.py --privacy True --noise_multiplier 2.0 --flag True --dataset "cifar10" --model "cnn" --lamb 0.6 --r 1.04 --interval 10 > ./logs/log_fedceo_noise=2.0_cifar10_LeNet.log 2>&1 &
```

---

### 2. Privacy Experiments

We use the **DLG attack** to evaluate privacy leakage. Lower PSNR indicates better privacy protection.

#### Figure: Privacy Attack Results on CIFAR-10 (PSNR in dB, lower is better)

<img width="482" height="345" alt="image" src="https://github.com/user-attachments/assets/5075a67e-272d-425f-899f-1d0c3937ff5b" />

#### Command to Run Privacy Experiment:
```bash
# privacy exps
nohup python -u attack_FedCEO.py --privacy True --noise_multiplier 2.0 --flag True --dataset "cifar10" --model "cnn" --index 100 --gpu "" > ./logs/log_attack_fedceo_noise=1.0_cifar10_LeNet.log 2>&1 &
```

---

### 3. Utility-Privacy Trade-off Experiments
We visualize the trade-off between utility (test accuracy) and privacy (ε_p) on CIFAR-10. FedCEO achieves the best balance.

<img width="402" height="352" alt="image" src="https://github.com/user-attachments/assets/9f1d1411-14ea-4765-852c-81abddbaeda8" />

## Citation  

```BibTex
@inproceedings{
li2025clients,
title={Clients Collaborate: Flexible Differentially Private Federated Learning with Guaranteed Improvement of Utility-Privacy Trade-off},
author={Yuecheng Li and Lele Fu and Tong Wang and Jian Lou and Bin Chen and Lei Yang and Jian Shen and Zibin Zheng and Chuan Chen},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=C7dmhyTDrx}
}
```



