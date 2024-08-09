import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNCifar
from utils import get_dataset, average_weights, exp_details
from TensorLR import TFedProx_module_TNN
from random_seed import setup_seed

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
print(torch.__version__, torchvision.__version__)

from utils import label_to_onehot, cross_entropy_for_onehot

import argparse
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
from vision import LeNet, weights_init

if __name__ == '__main__':
    start_time = time.time()


    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)
    setup_seed(args.seed)

    if args.gpu:
        torch.cuda.set_device(args.gpu_id)
    device = 'cuda' if args.gpu else 'cpu'

    train_dataset, test_dataset, user_groups = get_dataset(args)

    if args.model == 'cnn':
        global_model = LeNet()
        local_model = LeNet()

    elif args.model == 'mlp':
        
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
            local_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    
    global_model.to(device)
    global_model.train()
    print(global_model)

    
    global_weights = global_model.state_dict()

    
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0
    best_test_acc = 0

    local_weights = np.array([None] * args.num_users)
    TSVD = TFedProx_module_TNN(args)
    for epoch in tqdm(range(args.epochs)):
        local_losses = []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        local_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=True)
        print('The choiced users and its account: ', idxs_users, len(idxs_users))

        for i, idx in enumerate(idxs_users):
            local_update = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            if epoch == 0:
                w, loss = local_update.update_weights(
                    model=copy.deepcopy(global_model), global_round=epoch, args=args)
            elif epoch % args.interval == 0: 
                if local_weights[idx] == None:
                    local_model = global_model
                else:
                    new_state_dict = {}
                    for key, value in local_weights[idx].items():
                        if key.startswith('_module.'):
                            new_key = key.replace('_module.', '')
                            new_state_dict[new_key] = value
                        else:
                            new_state_dict[key] = value

                    local_model.load_state_dict(new_state_dict)
                    w, loss = local_update.update_weights(
                    model=copy.deepcopy(local_model), global_round=epoch, args=args)
            else:
                w, loss = local_update.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch, args=args)
            
            local_weights[idx] = copy.deepcopy(w)
            local_losses.append(copy.deepcopy(loss))

        if args.flag == True:
            if (epoch + 1) % args.interval == 0:
                print('semantic smooth epoch: ', epoch + 1)
                for key in local_weights[idxs_users[0]].keys():
                    weight_tensor = []
                    for idx in idxs_users:
                        weight_tensor.append(copy.deepcopy(local_weights)[idx][key].cpu().numpy())
                    L, E = TSVD.T_TSVD(np.array(weight_tensor).transpose(), epoch)
                    for i, idx in enumerate(idxs_users):
                        local_weights[idx][key] = torch.tensor(L.transpose()[i].squeeze())
            
                if epoch == 10*args.interval - 1:
                    device = "cpu"
                    # if torch.cuda.is_available():
                    #     device = "cuda"
                    print("Running on %s" % device)

                    if args.dataset == 'cifar10':
                        dst = datasets.CIFAR10("../data/cifar10", download=True)
                    tp = transforms.ToTensor()
                    tt = transforms.ToPILImage()

                    img_index = args.index
                    gt_data = tp(dst[img_index][0]).to(device)

                    if len(args.image) > 1:
                        gt_data = Image.open(args.image)
                        gt_data = tp(gt_data).to(device)


                    gt_data = gt_data.view(1, *gt_data.size())
                    gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
                    gt_label = gt_label.view(1, )
                    gt_onehot_label = label_to_onehot(gt_label)

                    # privacy image
                    plt.imshow(tt(gt_data[0].cpu()))
                    if args.privacy:
                        plt.savefig('./leakage_images_Ours/origin_data.svg', format="svg", bbox_inches="tight")
                        print('Defend Attack!')
                    else:
                        plt.savefig('./leakage_images/origin_data.svg', format="svg", bbox_inches="tight")

                    net = local_model
                    torch.manual_seed(1234)
                    criterion = cross_entropy_for_onehot

                    device = 'cpu'
                    pred = net(gt_data.to(device))
                    y = criterion(pred, gt_onehot_label)
                    dy_dx = torch.autograd.grad(y, net.parameters())

                    original_dy_dx = (list(_.detach().clone() for _ in dy_dx))

                    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
                    dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)

                    plt.imshow(tt(dummy_data[0].cpu()))
                    if args.privacy:
                        plt.savefig('./leakage_images_Ours/init_dummy_data.svg', format="svg", bbox_inches="tight")
                    else:
                        plt.savefig('./leakage_images/init_dummy_data.svg', format="svg", bbox_inches="tight")

                    optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=0.01)

                    history = []
                    for iters in range(1200):
                        def closure():
                            optimizer.zero_grad()

                            dummy_pred = net(dummy_data) 
                            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
                            dummy_loss = criterion(dummy_pred, dummy_onehot_label) 
                            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
                            
                            grad_diff = 0
                            for gx, gy in zip(dummy_dy_dx, original_dy_dx): 
                                grad_diff += ((gx - gy) ** 2).sum()
                            grad_diff.backward()
                            
                            return grad_diff
                        
                        optimizer.step(closure)
                        if (iters+1) % 60 == 0: 
                            current_loss = closure()
                            print(iters, "%.4f" % current_loss.item())
                            history.append(tt(dummy_data[0].cpu()))

                            plt.imshow(tt(dummy_data[0].cpu()))
                            if args.privacy:
                                plt.savefig('./leakage_images_Ours/dummy_data{}.svg'.format((iters+1)), format="svg", bbox_inches="tight")
                            else:
                                plt.savefig('./leakage_images/dummy_data{}.svg'.format((iters+1)), format="svg", bbox_inches="tight")

                    plt.figure(figsize=(12, 8))
                    for i in range(20):
                        plt.subplot(2, 10, i + 1)
                        plt.imshow(history[i])
                        if args.privacy:
                            plt.savefig('./leakage_images_Ours/dummy_image_{}.svg'.format((i+1) * 60), format="svg", bbox_inches="tight")
                        else:
                            plt.savefig('./leakage_images/dummy_image_{}.svg'.format((i+1) * 60), format="svg", bbox_inches="tight")
                        plt.title("iter=%d" % ((i+1) * 60))
                        plt.axis('off')
                    print('Ours privacy attack done!')


        global_weights = average_weights(local_weights[idxs_users])
        
        new_state_dict = {}
        for key, value in global_weights.items():
            if key.startswith('_module.'):
                new_key = key.replace('_module.', '')
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        global_model.load_state_dict(new_state_dict)
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        
        list_acc, list_loss = [], []
        global_model.eval()
        for idx in idxs_users:
            local_update = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            acc, loss, _ = local_update.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

    
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        if test_acc > best_test_acc:
            best_test_acc = test_acc

    file_name = './save/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

