import os
os.environ["KMP_DUPLICATE_LIB_OK"]= "TRUE"
import argparse

parser = argparse.ArgumentParser(
        description="CO3Net for Palmprint Recognition - Only Save Best Weight"
    )

parser.add_argument("--batch_size", type=int, default=128, help="Optimal batch size for 8GB VRAM")
parser.add_argument("--epoch_num", type=int, default=3, help="Local epochs")
parser.add_argument("--com", type=int, default=15, help="Total communication rounds")
parser.add_argument("--temp", type=float, default=0.07)
parser.add_argument("--weight1", type=float, default=0.7)
parser.add_argument("--weight2", type=float, default=0.15)
parser.add_argument("--weight3", type=float, default=1)
parser.add_argument("--mu", type=float, default=1e-2)

parser.add_argument("--id_num",type=int, default = 100, help = "IITD: 460 KTU: 145 Tongji: 600 REST: 358 XJTU: 200 POLYU 378 Multi-Spec 500")
parser.add_argument("--gpu_id",type=str, default='0')
parser.add_argument("--lr",type=float, default=0.001)
parser.add_argument("--redstep",type=int, default=30)
parser.add_argument("--mode", type=str, default='fedavg', help="fedavg|fedprox|fedpdf|fedavgM|fednova")
## Federated learning aggregation hyperparameters
parser.add_argument("--server_lr", type=float, default=1.0, help="Server-side learning rate, for FedAvgM")
parser.add_argument("--server_momentum", type=float, default=0.9, help="Server-side momentum factor, for FedAvgM")
parser.add_argument("--save_interval",type=int,default = 200)

##Training Path
parser.add_argument("--train_set_file",type=str,default='./data/train_all_server.txt')
parser.add_argument("--test_set_file",type=str,default='./data/test_server.txt')

##Store Path
parser.add_argument("--des_path",type=str,default='./weightps/checkpoint/')
parser.add_argument("--path_rst",type=str,default='./weightps/rst_test/')
parser.add_argument("--save_path",type=str,default='./cross-db-checkpoint/PolyU_1')
parser.add_argument("--seed",type=int,default=42)
args = parser.parse_args()


import time
import sys
import copy
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

from loss import SupConLoss
import matplotlib.pyplot as plt

from utils.util import saveLossACC
plt.switch_backend('agg')

from models import MyDataset
from models.compnet import co3net

torch.set_num_threads(8)

def set_seed(seed):
    """for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

server_momentum_buffer = {}

def communication(args, server_model, models, client_weights):
    global server_momentum_buffer
    with torch.no_grad():
        mode = args.mode.lower()
        server_sd = server_model.state_dict()

        # 1. Aggregation logic
        for key in server_sd.keys():
            if 'num_batches_tracked' in key:
                server_sd[key].data.copy_(models[0].state_dict()[key])
                continue

            is_bn = 'bn' in key
            is_fc = 'fc' in key

            if (mode == 'fedbn' and is_bn) or (mode == 'fedper' and is_fc):
                continue

            if mode == 'fedavgm':
                delta_w = torch.zeros_like(server_sd[key], dtype=torch.float32)
                for i, client_model in enumerate(models):
                    delta_w += client_weights[i] * (client_model.state_dict()[key] - server_sd[key])

                if key not in server_momentum_buffer:
                    server_momentum_buffer[key] = torch.zeros_like(delta_w)

                server_momentum_buffer[key].mul_(args.server_momentum).add_(delta_w)
                server_sd[key].data.add_(server_momentum_buffer[key], alpha=args.server_lr)
            else:
                temp = torch.zeros_like(server_sd[key], dtype=torch.float32)
                for i, client_model in enumerate(models):
                    temp += client_weights[i] * client_model.state_dict()[key]
                server_sd[key].data.copy_(temp)

        # 2. Sync distribution
        for i, client_model in enumerate(models):
            client_sd = client_model.state_dict()
            for key in server_sd.keys():
                if mode == 'fedbn' and 'bn' in key: continue
                if mode == 'fedper' and 'fc' in key: continue
                client_sd[key].data.copy_(server_sd[key])

    return server_model, models


def communication_sub(s_model, models):
    """Auxiliary aggregation logic"""
    with torch.no_grad():
        for key in s_model.state_dict().keys():
            if 'num_batches_tracked' in key:
                s_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
            else:
                temp = torch.zeros_like(s_model.state_dict()[key])
                for client_idx in range(len(models)):
                    temp += (1.0 / len(models)) * models[client_idx].state_dict()[key]
                s_model.state_dict()[key].data.copy_(temp)
    return s_model


def fit(epoch, model, data_loader, optimize=None, server_model=None, mode='fedavg', aux_model=None, phase='training'):
    if phase not in ['training', 'testing']:
        raise TypeError('input error!')

    if phase == 'training':
        model.train()
        aux_model.train()
        server_model.train()
    else:
        model.eval()
        aux_model.eval()
        server_model.eval()

    running_loss = 0
    entro_loss = 0
    supcon_loss = 0
    prox_loss = 0
    mse_loss = 0
    running_correct = 0

    cri_mse = nn.MSELoss().cuda()

    for batch_id, (datas, target) in enumerate(data_loader):
        data = datas[0].cuda()
        data_con = datas[1].cuda()
        target = target.cuda()

        if phase == 'training':
            optimize.zero_grad()
            output, fe1 = model(data, target)
            _, fe2 = model(data_con, target)
            _, fe3 = aux_model(data, target)
            fe = torch.cat([fe1.unsqueeze(1), fe2.unsqueeze(1)], dim=1)
        else:
            with torch.no_grad():
                output, fe1 = model(data, None)
                _, fe2 = model(data_con, None)
                _, fe3 = aux_model(data, None)
                fe = torch.cat([fe1.unsqueeze(1), fe2.unsqueeze(1)], dim=1)

        ce = criterion(output, target)
        ce2 = con_criterion(fe, target)
        fe1 = F.normalize(fe1, p=2, dim=-1)
        fe3 = F.normalize(fe3, p=2, dim=-1)
        ce3 = cri_mse(fe1, fe3.detach())

        # Proximal loss calculation
        def calc_proximal(m1, m2, mu_val, dev):
            diff = torch.tensor(0., device=dev)
            for p1, p2 in zip(m1.parameters(), m2.parameters()):
                diff += torch.sum(torch.pow(p1 - p2, 2))
            return (mu_val / 2.0) * torch.sqrt(diff + 1e-8)

        device = data.device
        loss_prox_server = calc_proximal(server_model, model, mu, device)
        loss_prox_aux = calc_proximal(aux_model, model, mu, device)
        loss2 = loss_prox_server + loss_prox_aux

        loss = weight1 * ce + weight2 * ce2 + weight3 * ce3 + loss2

        running_loss += loss.item()
        entro_loss += ce.item()
        supcon_loss += ce2.item()
        prox_loss += loss2.item() * weight3
        mse_loss += ce3.item()

        preds = output.data.max(dim=1, keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum().item()

        if phase == 'training':
            loss.backward()
            optimize.step()

    total = len(data_loader.dataset)
    loss = running_loss / total
    entroloss = entro_loss / total
    supconloss = supcon_loss / total
    proxloss = prox_loss / total
    mseloss = (mse_loss * weight3) / total
    accuracy = (100.0 * running_correct) / total

    # Simplified output
    if epoch % 5 == 0 or epoch == args.epoch_num - 1:
        print('epoch %d: \t%s loss is \t%7.5f ;\t%s accuracy is \t%d/%d \t%7.3f%%' % (
        epoch, phase, loss, phase, running_correct, total, accuracy))

    return loss, accuracy

def Dataset():
    src_dataset_1 = DataLoader(MyDataset(txt='./datacasia/train_NIR.txt', transforms=None, train=True, imside=128, outchannels=1), batch_size=batch_size, num_workers=0, shuffle=True)
    src_dataset_2 = DataLoader(MyDataset(txt='./datacasia/train_RED.txt', transforms=None, train=True, imside=128, outchannels=1), batch_size=batch_size, num_workers=0, shuffle=True)
    src_dataset_3 = DataLoader(MyDataset(txt='./datacasia/train_BLUE.txt', transforms=None, train=True, imside=128, outchannels=1), batch_size=batch_size, num_workers=0, shuffle=True)
    src_dataset_4 = DataLoader(MyDataset(txt='./datacasia/train_GREEN.txt', transforms=None, train=True, imside=128, outchannels=1), batch_size=batch_size, num_workers=0, shuffle=True)
    return[src_dataset_1, src_dataset_2, src_dataset_3, src_dataset_4]


if __name__== "__main__" :

    set_seed(args.seed)
    batch_size = args.batch_size
    epoch_num = args.epoch_num
    num_classes = args.id_num
    weight1 = args.weight1
    weight2 = args.weight2
    weight3 = args.weight3
    mu = args.mu
    communications = args.com

    print('seed:',args.seed)
    print('batch_size:', batch_size)
    print('local_epochs:', epoch_num)
    print('weight of cross:', weight1)
    print('weight of contra:', weight2)
    print('weight of mse:', weight3)

    des_path = args.des_path
    if not os.path.exists(des_path):
        os.makedirs(des_path)

    path_rst = args.path_rst
    if not os.path.exists(path_rst):
        os.makedirs(path_rst)

    names = ['red','green','blue','nir']
    print('%s' % (time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))

    train_datas = Dataset()

    # Initialize models
    server_model = co3net(num_classes=num_classes).cuda()
    best_net = co3net(num_classes=num_classes).cuda()

    visib_net = co3net(num_classes=num_classes).cuda()   ### Green & Blue
    invis_net = co3net(num_classes=num_classes).cuda()

    models = [copy.deepcopy(server_model) for _ in range(4)]
    optimizers =[torch.optim.Adam(models[idx].parameters(), lr=args.lr) for idx in range(4)]
    schedulers = [lr_scheduler.StepLR(optimizers[idx], step_size=args.redstep, gamma=0.8) for idx in range(4)]
    client_weights =[1 / 4 for _ in range(4)]

    criterion = nn.CrossEntropyLoss()
    con_criterion = SupConLoss(temperature=args.temp, base_temperature=args.temp)

    train_losses, train_accuracy = [], []
    val_losses, val_accuracy = [],[]
    bestacc = 0

    print("\n--- Start FL Training ---")
    for com in range(communications):
        temp_val_acc = []
        visb_models =[]
        invis_models =[]

        print(f"\n========== Communication {com}/{communications-1} ==========")

        for idx in range(4):
            print(f"--- Client {idx}: {names[idx].upper()} ---")
            for epoch in range(epoch_num):

                # Training process
                if idx == 1 or idx == 2:   ### visible (green & blue)
                    epoch_loss, epoch_accuracy = fit(epoch, models[idx], train_datas[idx], optimize=optimizers[idx], server_model=server_model, mode=args.mode.lower(), aux_model=invis_net, phase='training')
                else:                      ### invisible (red & nir)
                    epoch_loss, epoch_accuracy = fit(epoch, models[idx], train_datas[idx], optimize=optimizers[idx], server_model=server_model, mode=args.mode.lower(), aux_model=visib_net, phase='training')

                # Local validation process
                if idx == 1 or idx == 2:
                    val_epoch_loss, val_epoch_accuracy = fit(epoch, models[idx], train_datas[idx], server_model=server_model, aux_model=invis_net, phase='testing')
                else:
                    val_epoch_loss, val_epoch_accuracy = fit(epoch, models[idx], train_datas[idx], server_model=server_model, aux_model=visib_net, phase='testing')

                schedulers[idx].step()

                # Logging
                train_losses.append(epoch_loss)
                train_accuracy.append(epoch_accuracy)
                val_losses.append(val_epoch_loss)
                val_accuracy.append(val_epoch_accuracy)
                temp_val_acc.append(val_epoch_accuracy)

            if idx == 1 or idx == 2:
                visb_models.append(models[idx])
            else:
                invis_models.append(models[idx])

        # Auxiliary model aggregation
        visib_net = communication_sub(visib_net, visb_models)
        invis_net = communication_sub(invis_net, invis_models)

        # Server global model aggregation
        server_model, models = communication(args, server_model, models, client_weights)

        # Calculate average accuracy
        avg_val_acc = sum(temp_val_acc) / len(temp_val_acc)
        print(f"--> [Server] Global Communication {com} Avg Val Accuracy: {avg_val_acc:.3f}%")

        # Core: only keep and save the best weights
        if avg_val_acc >= bestacc:
            bestacc = avg_val_acc
            torch.save(server_model.state_dict(), des_path + 'net_params_best.pth')
            print(f"   [!] New Best Model Saved! Acc: {bestacc:.3f}%")

        # Periodically save regular weights or plot
        if com % 10 == 0 or com == (communications - 1):
            torch.save(server_model.state_dict(), des_path + 'net_params.pth')
            try:
                saveLossACC(train_losses, val_losses, train_accuracy, val_accuracy, bestacc, path_rst)
            except Exception:
                pass

        if com % args.save_interval == 0 and com != 0:
            torch.save(server_model.state_dict(), des_path + 'com_' + str(com) + '_net_params.pth')

    print('\nTraining Finished! The best model weight has been saved to:', des_path + 'net_params_best.pth')