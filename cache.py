import pickle
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# ================= Import Your Models and Datasets =================
from models.ccnet1 import ccnet
from models import MyDataset

# ================= 1. Core Configuration Zone =================
# Output directory for saving evaluation results
SAVE_DIR = r'save\dp'
# SAVE_DIR = r'save\AsyGA'
os.makedirs(SAVE_DIR, exist_ok=True)

# Model configuration (Switch paths/names to evaluate different methods)
MODEL_PATHS = {
    'Base': r'weightdp\checkpoint\net_params_best.pth'
    # ,'ours': r'save\checkpoint\AsyGA\AsyGA.pth'
}

# Paths to the text files containing test split data for the 4 spectral bands
BAND_TXT_FILES = {
    'NIR':   './datapolyu/train_NIR.txt',
    'Red':   './datapolyu/train_RED.txt',
    'Green': './datapolyu/train_GREEN.txt',
    'Blue':  './datapolyu/train_BLUE.txt'
}
# ===================================================================

def calculate_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    try:
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    except Exception:
        idx = np.nanargmin(np.absolute((fpr - fnr)))
        eer = (fpr[idx] + fnr[idx]) / 2.0
    return eer, fpr, tpr

@torch.no_grad()
def extract_features(model, txt_path, device):
    dataset = MyDataset(txt=txt_path, train=False, imside=128)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
    all_features, all_labels = [], []
    for datas, target in loader:
        img = datas[0].to(device) if isinstance(datas, list) else datas.to(device)
        # Invoke the model's feature extraction function
        feature = model.getFeatureCode(img)
        # L2 Normalization
        feature = torch.nn.functional.normalize(feature, p=2, dim=1) 
        all_features.append(feature.cpu().numpy())
        all_labels.append(target.numpy())
    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)


def evaluate_model(model_name, model_path, device):
    print(f"\n[{model_name}] Initializing model loading and feature extraction...")
    model = ccnet(num_classes=500).to(device)
    
    # 1. Load weights state dictionary
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    
    # 2. Replace mismatched key names for compatibility
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace('se_mlp', 'fc') # Resolves key naming discrepancy
        new_state_dict[new_k] = v
        
    # 3. Shape validation: Discard the final layer weights if the class counts mismatch
    model_dict = model.state_dict()
    filtered_dict = {
        k: v for k, v in new_state_dict.items() 
        if k in model_dict and v.shape == model_dict[k].shape
    }
    model.load_state_dict(filtered_dict, strict=False)
    model.eval()

    bands = list(BAND_TXT_FILES.keys())
    features_dict, labels_dict = {}, {}
    
    # Extract features for each individual spectral band
    for band, path in BAND_TXT_FILES.items():
        feat, lbl = extract_features(model, path, device)
        features_dict[band] = feat
        labels_dict[band] = lbl

    num_bands = len(bands)
    eer_matrix = np.zeros((num_bands, num_bands))
    all_y_true, all_y_score = [], []

    print(f"[{model_name}] Computing cross-spectral matching matrix...")
    for i, band1 in enumerate(bands):
        for j, band2 in enumerate(bands):
            f1, l1 = features_dict[band1], labels_dict[band1]
            f2, l2 = features_dict[band2], labels_dict[band2]
            
            sim_matrix = np.dot(f1, f2.T)
            
            if i == j:
                np.fill_diagonal(sim_matrix, -np.inf)
                indices = np.triu_indices(len(l1), k=1) 
                y_score = sim_matrix[indices]
                y_true = (l1[:, None] == l1[None, :])[indices].astype(int)
            else:
                y_score = sim_matrix.flatten()
                y_true = (l1[:, None] == l2[None, :]).flatten().astype(int)
            
            eer, fpr, tpr = calculate_eer(y_true, y_score)
            eer_matrix[i, j] = eer
            
            all_y_true.append(y_true)
            all_y_score.append(y_score)

    df_eer = pd.DataFrame(eer_matrix, index=bands, columns=bands)
    df_eer['Average'] = df_eer.mean(axis=1)
    df_eer.loc['Average'] = df_eer.mean(axis=0)

    final_y_true = np.concatenate(all_y_true)
    final_y_score = np.concatenate(all_y_score)

    return df_eer, final_y_true, final_y_score, features_dict, labels_dict


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Cache file destination path
    CACHE_FILE = os.path.join(SAVE_DIR, 'extracted_features_cache.pkl')
    
    # Set to True if you updated checkpoints and want to force recalculation
    FORCE_RECALCULATE = False 

    # =========================================================
    # Phase 1: Data Retrieval and Console Reporting
    # =========================================================
    if os.path.exists(CACHE_FILE) and not FORCE_RECALCULATE:
        print(f"============== Cache Found. Skipping Model Inference! ==============")
        print(f"Loading cached metrics from: {CACHE_FILE}")
        with open(CACHE_FILE, 'rb') as f:
            results = pickle.load(f)
        
        # Load and print EER matrix directly from cache
        for name, data in results.items():
            print(f"\n=================== [{name}] EER Matrix Results (From Cache) ===================")
            print(data['df'])
            print("================================================================================\n")
    else:
        print(f"============== Executing Model Inference | Device: {device} ==============")
        results = {}
        for name, path in MODEL_PATHS.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Target model checkpoint path not found: {path}")
                
            df, y_true, y_score, feat_dict, lbl_dict = evaluate_model(name, path, device)
            results[name] = {
                'df': df, 'y_true': y_true, 'y_score': y_score,
                'feat_dict': feat_dict, 'lbl_dict': lbl_dict
            }
            
            # Core Requirement 1: Display EER matrix in terminal
            print(f"\n=================== [{name}] EER Matrix Results ===================")
            print(df)
            print("===================================================================\n")
            
            # Core Requirement 2: Export model EER matrix to CSV
            csv_name = 'EER_Matrix_' + name.replace(" ", "_") + '.csv'
            df.to_csv(os.path.join(SAVE_DIR, csv_name))

        # Core Requirement 3: Serialize and save evaluation results to .pkl cache
        print(f"[Caching] Exporting data to {CACHE_FILE} ...")
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(results, f)
        print("Cache exported successfully!")

    print(f"Execution complete! All generated files are located in: {SAVE_DIR}")

if __name__ == "__main__":
    main()