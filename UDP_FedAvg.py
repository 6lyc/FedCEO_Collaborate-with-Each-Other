# UDP-FedAvg
import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNCifar
from utils import get_dataset, average_weights, exp_details
from random_seed import setup_seed

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

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

    print(args.dataset)
    train_dataset, test_dataset, user_groups = get_dataset(args)

    if args.model == 'cnn':
        global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
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

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses, local_weights_values = [], [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch, args=args)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            
        global_weights = average_weights(local_weights)

        
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
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            acc, loss, _ = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'ClientAvg Local Training Loss : {np.mean(np.array(train_loss))}')
            print('ClientAvg Local Test Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

    
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        print(f' \n Results after {epoch} global rounds of training:')
        print("|---- Test Loss: {:.2f}".format(test_loss))
        print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
        print("|---- Best Test Accuracy: {:.2f}%".format(100*best_test_acc))

    
    file_name = './save/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
