import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)

    parser.add_argument('--epochs', type=int, default=300,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100,
                        help="total number of users: N")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: p')
    parser.add_argument('--local_ep', type=int, default=30,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--privacy', type=bool, default=True, help='Adopt the DP Gaussian mechanism or not.')
    parser.add_argument('--noise_multiplier', type=float, default=1.0, help='The ratio of the standard deviation of the Gaussian noise to the L2-sensitivity of the function to which the noise is added (How much noise to add)')
    parser.add_argument('--flag', type=bool, default=True, help="Using our low-rank processing or not.")
    
    # TNN argments
    parser.add_argument('--eps', type=float, default=1e-10, help="The Control of Convergence!")
    parser.add_argument('--lamb', type=float, default=55, help="The weight of regularization term")
    parser.add_argument('--interval', type=int, default=5, help='The smoothing interval to adopt')
    parser.add_argument('--r', type=float, default=1.00, help='The common ratio of the geometric series')

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='mlp or cnn')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help="emnist or cifar10")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=True, help="Use gup or not.")
    parser.add_argument('--gpu-id', type=int, default=1, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=2023, help='random seed')

    # attack
    parser.add_argument('--index', type=int, default="25",
                    help='the index for leaking images on Dataset.')
    parser.add_argument('--image', type=str, default="",
                    help='the path to customized image.')


    args = parser.parse_args()
    return args
