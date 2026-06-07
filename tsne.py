import pickle
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve
from sklearn.manifold import TSNE  # Added: for computing t-SNE
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull  # Please ensure this line is added at the top of the script
# ================= Import models and dataset =================
from models.ccnet1 import ccnet
from models import MyDataset

# ================= 1. Core Configuration =================
# Output data save directory
SAVE_DIR = r'E:\document\1\PSFed-Palm-main1\md\bip\pics'
os.makedirs(SAVE_DIR, exist_ok=True)

# Core customization: enter the absolute paths of the model weights to compare below
MODEL_PATHS = {
    'Baseline':       r'E:\document\1\PSFed-Palm-main1\weightps\checkpoint\net_params_best.pth',
    'Traditional FT': r'E:\document\1\PSFed-Palm-main1\md\checkpoint\bipps1finetuned_best.pth',  # Replace with the actual weight path for AsyGA
    'AsyGA': r'E:\document\1\PSFed-Palm-main1\md\checkpoint\bippsfinetuned_best.pth'
}

# Dataset txt paths for the 4 bands (ensure these are test set data)
BAND_TXT_FILES = {
    'NIR':   './datapom/trainf_NIR.txt',
    'Red':   './datapom/trainf_RED.txt',
    'Green': './datapom/trainf_GREEN.txt',
    'Blue':  './datapom/trainf_BLUE.txt'
}

# Global academic paper plotting style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
# =====================================================

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
        # Call the model's feature extraction method
        feature = model.getFeatureCode(img)
        # L2 normalization
        feature = torch.nn.functional.normalize(feature, p=2, dim=1)
        all_features.append(feature.cpu().numpy())
        all_labels.append(target.numpy())
    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)


def evaluate_model(model_name, model_path, device):
    print(f"\n[{model_name}] Starting to load and extract features...")
    model = ccnet(num_classes=500).to(device)

    # 1. Load the weight dictionary
    state_dict = torch.load(model_path, map_location=device, weights_only=True)

    # 2. Replace mismatched key names in the dictionary
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace('se_mlp', 'fc')
        new_state_dict[new_k] = v

    # 3. Add shape check, discard the last layer weights that don't match the number of classes
    model_dict = model.state_dict()
    filtered_dict = {
        k: v for k, v in new_state_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    model.load_state_dict(filtered_dict, strict=False)
    model.eval()

    bands = list(BAND_TXT_FILES.keys())
    features_dict, labels_dict = {}, {}

    # Extract features for each band
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


# ================= Added core function: t-SNE academic plotting module =================
def plot_tsne_comparison(results):
    """
    Adaptive multi-model comparison t-SNE convex hull shadow plot.
    Supports 2 or 3 models arranged horizontally, dynamically showing progressive optimization of cluster compactness.
    """
    # Dynamically get the model list from current cache or run (maintain the order defined in the config area)
    target_models = [m for m in MODEL_PATHS.keys() if m in results]
    num_models = len(target_models)

    if num_models < 2:
        print(f"\n[-] Note: t-SNE comparison requires data from at least 2 models. Currently only have: {target_models}")
        return

    print(f"\n============== Starting t-SNE dimensionality reduction with convex hull shading for [{num_models} models] ==============")

    # Dynamically adjust canvas width based on model count: width 16 for 2 models, 22 for 3 models
    fig_width = 16 if num_models == 2 else 22
    fig, axes = plt.subplots(1, num_models, figsize=(fig_width, 7.5))

    # If only one subplot (fallback), make axes a list for easy iteration
    if num_models == 1:
        axes = [axes]

    for idx, name in enumerate(target_models):
        ax = axes[idx]
        feat_dict = results[name]['feat_dict']
        lbl_dict = results[name]['lbl_dict']

        # 1. Merge multi-band data
        all_feats = np.concatenate([feat_dict[b] for b in feat_dict.keys()], axis=0)
        all_lbls = np.concatenate([lbl_dict[b] for b in lbl_dict.keys()], axis=0)

        # 2. Select the first 10 classes for display
        unique_labels = np.unique(all_lbls)
        selected_classes = unique_labels[:10]

        mask = np.isin(all_lbls, selected_classes)
        X_selected = all_feats[mask]
        y_selected = all_lbls[mask]

        if len(X_selected) > 2500:
            np.random.seed(42)
            sample_idx = np.random.choice(len(X_selected), 2500, replace=False)
            X_selected = X_selected[sample_idx]
            y_selected = y_selected[sample_idx]

        print(f"  [+] Executing t-SNE manifold computation for [{name}]...")

        # Compatible with latest scikit-learn
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, init='pca')
        X_tsne = tsne.fit_transform(X_selected)

        unique_y = np.unique(y_selected)
        palette = sns.color_palette("tab10", len(unique_y))

        # 3. Draw convex hull shaded regions for each class
        for c_idx, c in enumerate(unique_y):
            c_mask = (y_selected == c)
            X_c = X_tsne[c_mask]

            if len(X_c) >= 3:
                try:
                    from scipy.spatial import ConvexHull
                    hull = ConvexHull(X_c)
                    # Draw semi-transparent filled shadow
                    ax.fill(X_c[hull.vertices, 0], X_c[hull.vertices, 1],
                            color=palette[c_idx], alpha=0.12, zorder=1)
                    # Draw boundary dashed line
                    ax.plot(np.append(X_c[hull.vertices, 0], X_c[hull.vertices[0], 0]),
                            np.append(X_c[hull.vertices, 1], X_c[hull.vertices[0], 1]),
                            color=palette[c_idx], linestyle='--', linewidth=1.2, alpha=0.4, zorder=1)
                except Exception:
                    pass

            # Draw solid scatter points
            ax.scatter(X_c[:, 0], X_c[:, 1],
                       color=palette[c_idx], label=f'ID-{c}',
                       alpha=0.85, edgecolors='black', linewidths=0.5, s=35, zorder=2)

        # 4. Subplot beautification
        ax.set_title(f'Feature Space: {name}', fontsize=14, fontweight='bold', pad=12)
        ax.set_xlabel('t-SNE Dimension 1', fontsize=11, fontweight='bold')
        ax.set_ylabel('t-SNE Dimension 2', fontsize=11, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.5, zorder=0)

        # Dynamically determine: only place the unified legend outside the rightmost subplot
        if idx == num_models - 1:
            ax.legend(title="Palmprint Identities", bbox_to_anchor=(1.04, 1), loc='upper left', fontsize=9, borderaxespad=0.)

    plt.suptitle("Feature Embedding Clustering Comparison (t-SNE)\n"
                 "Shaded Areas Represent Cluster Convex Hulls (Smaller Area = Better Compactness)",
                 fontsize=16, y=0.98, fontweight='bold')

    plt.tight_layout()
    tsne_save_path = os.path.join(SAVE_DIR, f'tSNE_{num_models}Models_Comparison.pdf')
    plt.savefig(tsne_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" [OK] High-resolution comparison chart for {num_models} models exported to: {tsne_save_path}\n")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cache file path
    CACHE_FILE = os.path.join(SAVE_DIR, 'extracted_features_cache.pkl')

    # Set to True if you want to force re-run after updating weight files
    FORCE_RECALCULATE = False

    # =========================================================
    # Stage 1: Data acquisition and terminal output
    # =========================================================
    if os.path.exists(CACHE_FILE) and not FORCE_RECALCULATE:
        print(f"============== Cache file found, skipping model inference! ==============")
        print(f"Loading cache: {CACHE_FILE}")
        with open(CACHE_FILE, 'rb') as f:
            results = pickle.load(f)

        for name, data in results.items():
            print(f"\n=================== [{name}] EER Matrix Results (from cache) ===================")
            print(data['df'])
            print("=================================================================\n")
    else:
        print(f"============== Starting model inference and feature extraction | Using device: {device} ==============")
        results = {}
        for name, path in MODEL_PATHS.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")

            df, y_true, y_score, feat_dict, lbl_dict = evaluate_model(name, path, device)
            results[name] = {
                'df': df, 'y_true': y_true, 'y_score': y_score,
                'feat_dict': feat_dict, 'lbl_dict': lbl_dict
            }

            print(f"\n=================== [{name}] EER Matrix Results ===================")
            print(df)
            print("=========================================================\n")

            csv_name = 'EER_Matrix_' + name.replace(" ", "_") + '.csv'
            df.to_csv(os.path.join(SAVE_DIR, csv_name))

        print(f"[Saving cache] Saving data to {CACHE_FILE} ...")
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(results, f)
        print("Cache saved successfully!")

    # =========================================================
    # Stage 2: New call area -- automatically generate t-SNE 1-row comparison chart
    # =========================================================
    plot_tsne_comparison(results)

    print(f"All done! Data files and charts have been saved to: {SAVE_DIR}")

if __name__ == "__main__":
    main()