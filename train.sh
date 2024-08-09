# demo

# utility exps
# FedAvg
nohup python -u LDP_FedAvg.py --privacy "" --flag "" --dataset cifar10 --model "cnn" > ./logs/log_fedavg_cifar10_LeNet.log 2>&1 &

# LDP-FedAvg
nohup python -u LDP_FedAvg.py --privacy True --flag "" --noise_multiplier 2.0 --dataset "cifar10" --model "cnn" > ./logs/log_ldp_fedavg_noise=2.0_cifar10_LeNet.log 2>&1 &

# Our FedCEO
# CIFAR-10
nohup python -u FedCEO.py --privacy True --noise_multiplier 2.0 --flag True --dataset "cifar10" --model "cnn" --lamb 0.6 --r 1.04 --interval 10 > ./logs/log_fedceo_noise=2.0_cifar10_LeNet.log 2>&1 &

# EMNIST
nohup python -u FedCEO.py --privacy True --noise_multiplier 2.0 --flag True --dataset "emnist" --num_classes 37 --model "mlp" --lamb 0.03 --r 1.06 --interval 20 > ./logs/log_fedceo_noise=2.0_emnist_MLP.log 2>&1 &

# privacy exps
nohup python -u attack_FedCEO.py --privacy True --noise_multiplier 2.0 --flag True --dataset "cifar10" --model "cnn" --index 100 --gpu "" > ./logs/log_attack_fedceo_noise=1.0_cifar10_LeNet.log 2>&1 &
