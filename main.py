import os
import time  # Import time module for resource benchmarking
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.ccnet1 import ccnet as ccnet
from models import MyDataset
from loss import SupConLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

# --- Hook function to freeze BatchNorm statistics ---
def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def load_and_finetune():

    # ==========================================
    # Ablation Study Control Panel (Ablation Study Switches)
    # ==========================================
    use_feature_anchor = True   # 1. Teacher model anchor (controls loss_reg)
    use_hard_freeze    = True    # 2. Backbone hard freeze
    use_bn_freeze      = True   # 3. Freeze BatchNorm statistics
    use_contrastive_loss = True  # 4. Supervised contrastive loss switch
    # ==========================================

    # --- 1. Parameter settings ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_path = r'E:\document\1\PSFed-Palm-main1\weightps\checkpoint\net_params_best.pth'
    save_path = r'E:\document\1\PSFed-Palm-main1\md\checkpoint\bipps0finetuned_best.pth'


    real_id_num = 100
    batch_size = 64

    deep_base_lr = 1e-6
    head_lr = 1e-4
    max_epochs = 200
    patience = 20

    con_weight = 0.5
    reg_weight = 45.0

    # --- 2. Initialize student model ---
    model = ccnet(num_classes=real_id_num).to(device)
    # --- 3. Load pretrained weights ---
    if os.path.exists(pretrained_path):
        print(f"Loading multispectral pretrained weights: {pretrained_path}")
        pretrained_dict = torch.load(pretrained_path, map_location=device)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict and "arclayer_" not in k and "fc" not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("Multispectral feature extraction layers loaded successfully.")
    else:
        print("No pretrained file found!")

    # --- Teacher model initialization ---
    if use_feature_anchor:
        teacher_model = copy.deepcopy(model)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
            param.grad = None
    else:
        teacher_model = None

    # --- 4. Prepare data ---
    train_txt = './ca/test.txt'
    train_dataset = MyDataset(txt=train_txt, train=True, imside=128)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    val_txts = {
        'Blue': './dataca/train_BLUE.txt',
        'Green': './dataca/train_GREEN.txt',
        'Red': './dataca/train_RED.txt',
        'NIR': './dataca/train_NIR.txt'
    }

    val_loaders = {}
    for mod, txt_path in val_txts.items():
        if os.path.exists(txt_path):
            v_dataset = MyDataset(txt=txt_path, train=False, imside=128)
            val_loaders[mod] = DataLoader(v_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

# ==========================================================
    # --- 5. Backbone hard freeze strategy (ultimate solution for massive FC architecture) ---
    # ==========================================================
    head_params = []
    tunable_base_params = []

    frozen_params_cnt = 0
    trainable_params_cnt = 0

    print(f"\n[Strategy] Executing feature extractor freeze strategy...")

    for name, param in model.named_parameters():
        # Extract the layer prefix name (e.g. 'fc.weight' -> 'fc')
        layer_prefix = name.split('.')[0]

        # 1. Precisely identify the massive mapping layers and classification head (keep trainable)
        is_head = layer_prefix in ['fc', 'fc1', 'arclayer']

        if is_head:
            param.requires_grad = True
            head_params.append(param)
            trainable_params_cnt += param.numel()
        else:
            # 2. Belongs to multispectral convolutional backbone (very few parameters, but heavy computation)
            if use_hard_freeze:
                param.requires_grad = False
                frozen_params_cnt += param.numel()
            else:
                param.requires_grad = True
                tunable_base_params.append(param)
                trainable_params_cnt += param.numel()

    # Inject optimizer
    optimizer = torch.optim.Adam([
        {'params': tunable_base_params, 'lr': deep_base_lr},
        {'params': head_params, 'lr': head_lr}
    ], weight_decay=1e-4)

    print("\n" + "="*50)
    print("[Academic Metric 1] Modular Freezing Analysis (Architectural Freezing)")
    print("="*50)
    print(f"  Strategy Status: Hard Freezing = {use_hard_freeze}")
    print(f"  Trainable Parameters (Heads): {trainable_params_cnt / 1e6:.4f} M")
    print(f"  Frozen Parameters (Backbone): {frozen_params_cnt / 1e6:.4f} M (Note: few conv params, but saves VRAM/compute)")
    print("="*50 + "\n")
    # ==========================================================

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    criterion = nn.CrossEntropyLoss()
    con_criterion = SupConLoss(temperature=0.07).to(device)

    # =====================================================================
    # Resource Benchmark: Dimension 1 - Count Trainable Parameters
    # =====================================================================
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print("\n" + "="*50)
    print("[Academic Metric 1] Model Parameter Analysis (Parameter Analysis)")
    print("="*50)
    print(f"  Strategy Status: Hard Freezing = {use_hard_freeze}")
    print(f"  Total Parameters      : {total_params / 1e6:.4f} M")
    print(f"  Trainable Parameters  : {trainable_params / 1e6:.4f} M (Proportion {(trainable_params/total_params)*100:.1f}%)")
    print(f"  Frozen Parameters     : {frozen_params / 1e6:.4f} M")
    print("="*50 + "\n")

    # --- 6. Test original model, establish a clean baseline ---
    print(f"--- Testing original model, establishing clean multispectral baseline ---")
    model.eval()
    baseline_losses = {mod: 0.0 for mod in val_loaders.keys()}

    with torch.no_grad():
        for mod, loader in val_loaders.items():
            val_con_loss_mod = 0.0
            val_batches_mod = 0
            for datas, target in loader:
                data1 = datas[0].to(device)
                target = target.to(device).long()
                _, fe1_val = model(data1, target)
                fe_val_formatted = fe1_val.unsqueeze(1)
                loss_val_con = con_criterion(fe_val_formatted, target)
                val_con_loss_mod += loss_val_con.item()
                val_batches_mod += 1
            if val_batches_mod > 0:
                baseline_losses[mod] = val_con_loss_mod / val_batches_mod

    best_val_loss = sum(baseline_losses.values()) / len(baseline_losses)
    epochs_no_improve = 0

    mod_str = " | ".join([f"{mod}: {baseline_losses[mod]:.4f}" for mod in baseline_losses.keys()])
    print(f"      Original Model Val_Feature_Loss => [{mod_str}] => Global Average (Avg): {best_val_loss:.4f}\n")

    # Reset CUDA peak memory statistics
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    # --- 7. Training loop ---
    for epoch in range(max_epochs):
        model.train()

        if use_bn_freeze:
            model.apply(set_bn_eval)

        train_loss, train_ce, train_con, train_reg = 0.0, 0.0, 0.0, 0.0

        # Record epoch start time (use synchronize to ensure accurate GPU timing)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_start_time = time.time()

        # Use enumerate to get batch_idx
        for batch_idx, (datas, target) in enumerate(train_loader):
            data1, data2 = datas[0].to(device), datas[1].to(device)
            target = target.to(device).long()

            optimizer.zero_grad()

            output, fe1 = model(data1, target)
            _, fe2 = model(data2, target)

            # Basic classification loss
            loss_ce = criterion(output, target)

            # Contrastive learning loss computation
            fe = torch.cat([fe1.unsqueeze(1), fe2.unsqueeze(1)], dim=1)
            loss_con = con_criterion(fe, target)

            # Teacher anchor loss computation
            loss_reg = torch.tensor(0.0).to(device)
            if use_feature_anchor:
                with torch.no_grad():
                    _, fe_teacher = teacher_model(data1, target)
                cos_sim = F.cosine_similarity(fe1, fe_teacher.detach(), dim=1)
                loss_reg = (1.0 - cos_sim).mean()

            # Core loss ablation assembly logic
            loss = loss_ce

            if use_contrastive_loss:
                loss = loss + con_weight * loss_con

            if use_feature_anchor:
                loss = loss + reg_weight * loss_reg

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_ce += loss_ce.item()
            if use_contrastive_loss:
                train_con += loss_con.item()
            if use_feature_anchor:
                train_reg += loss_reg.item()

            # =====================================================================
            # Resource Benchmark: Dimension 2 - Measure Peak VRAM Usage
            # Most accurate after the first batch of the first epoch, as the
            # computation graph and Adam state are fully built at this point
            # =====================================================================
            if epoch == 0 and batch_idx == 0 and torch.cuda.is_available():
                torch.cuda.synchronize()
                peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                print("\n" + "="*50)
                print("[Academic Metric 2] VRAM Space Overhead (VRAM Overhead)")
                print("="*50)
                print(f"  Peak VRAM Usage: {peak_vram:.2f} MB")
                print("  (Note: This metric reflects space used by the backward computation graph and optimizer state)")
                print("="*50 + "\n")

        # =====================================================================
        # Resource Benchmark: Dimension 3 - Measure Throughput & Time
        # =====================================================================
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_end_time = time.time()

        train_epoch_time = epoch_end_time - epoch_start_time
        num_images_processed = len(train_loader) * batch_size
        throughput = num_images_processed / train_epoch_time

        # ---------- Validation phase ----------
        model.eval()
        mod_losses = {mod: 0.0 for mod in val_loaders.keys()}

        with torch.no_grad():
            for mod, loader in val_loaders.items():
                val_con_loss_mod = 0.0
                val_batches_mod = 0
                for datas, target in loader:
                    data1 = datas[0].to(device)
                    target = target.to(device).long()
                    _, fe1_val = model(data1, target)
                    fe_val_formatted = fe1_val.unsqueeze(1)
                    loss_val_con = con_criterion(fe_val_formatted, target)
                    val_con_loss_mod += loss_val_con.item()
                    val_batches_mod += 1
                if val_batches_mod > 0:
                    mod_losses[mod] = val_con_loss_mod / val_batches_mod

        avg_val_con_total = sum(mod_losses.values()) / len(mod_losses)
        scheduler.step(avg_val_con_total)

        # Print log (with timing and throughput info)
        log_str = f"Ep [{epoch+1:03d}/{max_epochs}] Tr_Loss: {train_loss/len(train_loader):.3f} (CE:{train_ce/len(train_loader):.3f}"
        if use_contrastive_loss: log_str += f", CON:{train_con/len(train_loader):.3f}"
        if use_feature_anchor:   log_str += f", REG:{train_reg/len(train_loader):.4f}"
        log_str += ")"
        print(log_str)

        # Print timing overhead log
        print(f"      [Academic Metric 3] Timing Overhead: Training Time = {train_epoch_time:.2f} s/epoch | Throughput = {throughput:.1f} images/s")

        mod_str = " | ".join([f"{mod}: {mod_losses[mod]:.4f}" for mod in mod_losses.keys()])
        print(f"      After Fine-tuning Val_Feature_Loss => [{mod_str}] => Global Average (Avg): {avg_val_con_total:.4f}")

        # Save logic
        if avg_val_con_total < best_val_loss:
            best_val_loss = avg_val_con_total
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  [!] Baseline broken! Best weights saved (Avg Loss: {avg_val_con_total:.4f})\n")
        else:
            epochs_no_improve += 1
            print(f"  [-] Baseline not surpassed, current best: {best_val_loss:.4f} (Early stop: {epochs_no_improve}/{patience})\n")

        if epochs_no_improve >= patience:
            break

if __name__ == "__main__":

    load_and_finetune()