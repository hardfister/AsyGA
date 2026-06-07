import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import time
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
import random
import matplotlib.pyplot as plt

from utils.util import plotLossACC, saveLossACC
plt.switch_backend('agg')

from models import MyDataset
from models.ccnet1 import ccnet as co3net

torch.set_num_threads(8)

parser = argparse.ArgumentParser(
    description="DPFed-Palm for Palmprint Recognition"
)

# Basic hyperparameters
parser.add_argument("--batch_size", type=int, default=128, help="Optimal batch size for 8GB VRAM")
parser.add_argument("--epoch_num", type=int, default=3, help="Local epochs, recommended 1~5")
parser.add_argument("--com", type=int, default=50, help="Total communication rounds")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--redstep", type=int, default=30)
parser.add_argument("--seed", type=int, default=42)

# DPFed-Palm additional hyperparameters (corresponding to the Aggregation setup in the paper)
parser.add_argument("--k", type=int, default=10, help="First k rounds use FedAvg, then switch to dynamic PFL")
parser.add_argument("--m_step", type=float, default=0.1, help="Step size for dynamic PFL m search")

# Paths and other settings
parser.add_argument("--mode", type=str, default='fedavg', help="fedavg|fedprox|fedpdf|fedavgM|fednova")
parser.add_argument("--des_path", type=str, default='./weightdp/checkpoint/')
parser.add_argument("--path_rst", type=str, default='./weightdp/rst_test/')

args = parser.parse_args()


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


# ========================== Core Loss Functions from the Paper ==========================

class SupConLoss(nn.Module):
    """Eq.(4): Supervised Contrastive Loss"""
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Compute cosine similarity and divide by temperature parameter tau
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)

        # Subtract max for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Remove self-vs-self contrast (zero out diagonal)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        # Compute log(exp(z_i * z_p / tau) / sum(exp(z_i * z_a / tau)))
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # Average log_prob over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1e-6)
        
        loss = - mean_log_prob_pos.mean()
        return loss

def batch_hard_triplet_loss(labels, embeddings, margin=1.0):
    """Eq.(5): Triplet Loss (based on Batch Hard strategy)"""
    dot_product = torch.matmul(embeddings, embeddings.t())
    square_norm = torch.diag(dot_product)
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
    distances = torch.max(distances, torch.zeros_like(distances))
    distances = torch.sqrt(distances + 1e-16)

    mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    mask_neg = (labels.unsqueeze(0) != labels.unsqueeze(1)).float()

    # Find the hardest positive for each anchor (max distance)
    hardest_positive_dist = (distances * mask_pos).max(dim=1)[0]

    # Find the hardest negative for each anchor (min distance)
    max_dist = distances.max(dim=1, keepdim=True)[0]
    hardest_negative_dist = (distances + max_dist * mask_pos).min(dim=1)[0]

    # max(0, d(a,p) - d(a,n) + alpha)
    loss = F.relu(hardest_positive_dist - hardest_negative_dist + margin)
    return loss.mean()

# ========================== Federated Learning Core Logic ==========================

def communication_fedavg_only(server_model, models, client_weights):
    """Compute FedAvg global model only, without direct distribution (for PFL base model calculation)"""
    server_sd = server_model.state_dict()
    with torch.no_grad():
        for key in server_sd.keys():
            if 'num_batches_tracked' in key:
                server_sd[key].data.copy_(models[0].state_dict()[key])
                continue
            
            # Check shape compatibility
            shape_mismatch = False
            for client_model in models:
                if key not in client_model.state_dict() or client_model.state_dict()[key].shape != server_sd[key].shape:
                    shape_mismatch = True
                    break
            if shape_mismatch: continue

            temp = torch.zeros_like(server_sd[key], dtype=torch.float32)
            for i, client_model in enumerate(models):
                temp += client_weights[i] * client_model.state_dict()[key]
            server_sd[key].data.copy_(temp)
    return server_model

def distribute_model(server_model, models):
    """Distribute global model to each client (standard FedAvg procedure)"""
    server_sd = server_model.state_dict()
    with torch.no_grad():
        for i, client_model in enumerate(models):
            client_sd = client_model.state_dict()
            for key in server_sd.keys():
                if key in client_sd and client_sd[key].shape == server_sd[key].shape:
                    client_sd[key].data.copy_(server_sd[key])
    return models

def fit(epoch, model, data_loader, optimize=None, phase='training', verbose=True):
    if phase == 'training':
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_correct = 0
    
    # Initialize loss function components
    criterion_ce = nn.CrossEntropyLoss().cuda()
    criterion_supcon = SupConLoss(temperature=0.07).cuda()

    for batch_id, (datas, target) in enumerate(data_loader):
        data = datas[0].cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        if phase == 'training':
            optimize.zero_grad()
            # Modified ccnet returns logits and normalized features
            output, features = model(data, target)

            # --- Compute Four Loss Components ---
            # 1. Cross-Entropy / ArcFace loss (ArcFace logits are fed directly to CE)
            loss_ce_arc = criterion_ce(output, target)

            # 2. Supervised Contrastive Loss
            loss_con = criterion_supcon(features, target)

            # 3. Triplet Loss (default alpha=1.0 per paper)
            loss_triplet = batch_hard_triplet_loss(target, features, margin=1.0)

            # 4. Center Loss (retrieved from modified ccnet)
            loss_center = model.get_center_loss(features, target)

            # --- Combined Loss Function ---
            # L_1 = 0.8 * CE + 0.2 * Con
            # L_2 = Triplet
            # L_3 = ArcFace_CE + Center
            # Since our CE already includes ArcFace_CE logic, the total weighted aggregation is:
            total_loss = (1.8 * loss_ce_arc) + (0.2 * loss_con) + loss_triplet + loss_center
            
            total_loss.backward()
            optimize.step()
            loss_item = total_loss.item()
            
        else:
            with torch.no_grad():
                # During testing, ArcFace margin is not needed (pass y=None), and complex losses are not computed
                output, _ = model(data, None)
                loss_ce_arc = criterion_ce(output, target)
                loss_item = loss_ce_arc.item()

        running_loss += loss_item
        preds = output.data.max(dim=1, keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum().item()

    total = len(data_loader.dataset)
    loss_val = running_loss / total
    accuracy = (100.0 * running_correct) / total

    if verbose:
        print('epoch %d: \t%s loss is \t%7.5f ;\t%s accuracy is \t%d/%d \t%7.3f%%' % (
            epoch, phase, loss_val, phase, running_correct, total, accuracy))

    return loss_val, accuracy

def cross_client_evaluate(models, test_loaders):
    """
    DPFed-Palm cross-client validation:
    Client i's personalized model is validated on all Client j's test sets.
    """
    total_acc = 0.0
    eval_count = 0
    
    for i, model in enumerate(models):
        for j, loader in enumerate(test_loaders):
            _, acc = fit(0, model, loader, phase='testing', verbose=False)
            total_acc += acc
            eval_count += 1
            
    return total_acc / eval_count

def load_data(batch_size):
    workers = 4
    train_loaders = []
    test_loaders = []
    for i in range(1, 5):
        train_loader = DataLoader(MyDataset(txt=f'./datacasia/train_{i}.txt', transforms=None, train=True, imside=128, outchannels=1), 
                                  batch_size=batch_size, num_workers=workers, pin_memory=True, shuffle=True)
        test_loader = DataLoader(MyDataset(txt=f'./datacasia/test_{i}.txt', transforms=None, train=False, imside=128, outchannels=1), 
                                 batch_size=batch_size, num_workers=workers, pin_memory=True, shuffle=False)
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    return train_loaders, test_loaders


# ========================== Main Program ==========================
if __name__== "__main__" :
    set_seed(args.seed)

    # Assume homogeneous multi-spectral tasks with the same number of client classes
    client_id_nums = [100, 100, 100, 100]

    print('==> Initiating DPFed-Palm Configuration...')
    print(f'Com rounds: {args.com} | k (FedAvg rounds): {args.k} | m_step: {args.m_step}')

    os.makedirs(args.des_path, exist_ok=True)
    os.makedirs(args.path_rst, exist_ok=True)

    # Load data
    train_datas, test_datas = load_data(args.batch_size)
    names = ['red','green','blue','nir']

    # Initialize Server and Client models
    server_model = co3net(num_classes=max(client_id_nums)).cuda()
    best_net = co3net(num_classes=max(client_id_nums)).cuda()

    models = []
    for idx in range(4):
        client_model = co3net(num_classes=client_id_nums[idx]).cuda()
        client_sd = client_model.state_dict()
        for key in server_model.state_dict().keys():
            if key in client_sd and client_sd[key].shape == server_model.state_dict()[key].shape:
                client_sd[key].data.copy_(server_model.state_dict()[key])
        models.append(client_model)
        
    optimizers = [torch.optim.Adam(models[idx].parameters(), lr=args.lr) for idx in range(4)]
    schedulers = [lr_scheduler.StepLR(optimizers[idx], step_size=args.redstep, gamma=0.8) for idx in range(4)]
    client_weights = [1/4 for _ in range(4)]
    
    bestacc = 0

    for com in range(args.com):
        print(f"\n========== Communication {com} / {args.com-1} ==========")
        
        # 1. Local Training
        for idx in range(4):
            print(f"--- Client {idx}: {names[idx].upper()} Local Training ---")
            for epoch in range(args.epoch_num):
                fit(epoch, models[idx], train_datas[idx], optimize=optimizers[idx], phase='training')
                schedulers[idx].step()

        # 2. Aggregation Phase
        if com < args.k:
            # Phase A: First k rounds use FedAvg
            print("=> Aggregation Strategy: FedAvg")
            server_model = communication_fedavg_only(server_model, models, client_weights)
            models = distribute_model(server_model, models)

            # Validation on own domain
            current_acc_sum = 0
            for idx in range(4):
                _, acc = fit(0, models[idx], test_datas[idx], phase='testing', verbose=False)
                current_acc_sum += acc
            current_acc_mean = current_acc_sum / 4
            print(f"--> [Server] Global Communication {com} Avg Val Accuracy: {current_acc_mean:.3f}%")

        else:
            # Phase B: Later rounds use dynamic PFL
            print("=> Aggregation Strategy: Dynamic PFL (Cross-Spectral Testing)")

            # (1) Extract the base global model theta_1 for this round
            server_model = communication_fedavg_only(server_model, models, client_weights)
            global_sd = server_model.state_dict()

            best_m = 0.0
            best_cross_acc = 0.0
            best_personalized_sds = [None] * 4

            # (2) Traverse weight m to search for optimal personalized parameters
            for m_val in np.arange(0.0, 1.0 + args.m_step, args.m_step):
                m = round(m_val, 2)  # Prevent floating point precision issues

                # Create temporary personalized models: theta_3 = (1-m) * theta_1 + m * theta_2
                temp_models = []
                for idx in range(4):
                    local_sd = models[idx].state_dict()
                    temp_sd = {}
                    for key in global_sd.keys():
                        if key in local_sd and local_sd[key].shape == global_sd[key].shape:
                            temp_sd[key] = (1 - m) * global_sd[key] + m * local_sd[key]
                        elif key in local_sd:
                            temp_sd[key] = local_sd[key]

                    tm = co3net(num_classes=client_id_nums[idx]).cuda()
                    tm.load_state_dict(temp_sd)
                    temp_models.append(tm)

                # (3) Cross-domain test evaluation
                cross_acc = cross_client_evaluate(temp_models, test_datas)
                print(f"   [Search] m = {m:.1f} | Average Cross-Client Accuracy = {cross_acc:.3f}%")

                if cross_acc >= best_cross_acc:
                    best_cross_acc = cross_acc
                    best_m = m
                    best_personalized_sds = [tm.state_dict() for tm in temp_models]

                # Free VRAM
                del temp_models
                torch.cuda.empty_cache()

            print(f"--> [Dynamic PFL] Optimal m selected: {best_m} with Cross-Acc: {best_cross_acc:.3f}%")
            current_acc_mean = best_cross_acc

            # (4) Load the selected optimal personalized parameters to local clients for next round
            for idx in range(4):
                models[idx].load_state_dict(best_personalized_sds[idx])

        # 3. Model saving logic
        if current_acc_mean >= bestacc:
            bestacc = current_acc_mean
            torch.save(server_model.state_dict(), args.des_path + 'net_params_best.pth')
            print(f"   [!] New Best Model Saved! Acc: {bestacc:.3f}%")

    print('\nTraining Finished!')
    print(f'Best Overall Accuracy: {bestacc:.3f}%')