import os
import math
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc  # Garbage collection to free memory

# ================= 1. Core Configuration =================

# 1. Fill in the absolute paths to all your PKL files (8 total, each containing only 1 model)
PKL_FILES = [
    r'save\dp\extracted_features_cache.pkl',
    r'save\Asy\extracted_features_cache.pkl'
]

# 2. Fill in the model names corresponding one-to-one with the PKL files above (8 total)
NEW_MODEL_NAMES = [
    'DPFed-Palm',
    'PSFed-Palm-AsyGA'
]

SAVE_DIR = r'save\pic'
os.makedirs(SAVE_DIR, exist_ok=True)

# Global academic paper style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11

# =============================================================

def ultra_fast_roc_auc_tar(y_true, y_score, target_far=1e-6, decimals=4):
    """
    [OOM-Safe Algorithm] For billion-scale massive data, uses high-efficiency histogram statistics
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    
    score_int = np.round(y_score * (10**decimals)).astype(np.int32)
    min_s = np.min(score_int)
    score_int -= min_s
    max_s = np.max(score_int)
    
    hist_gen = np.bincount(score_int, weights=y_true, minlength=max_s+1)
    hist_imp = np.bincount(score_int, weights=(1 - y_true), minlength=max_s+1)
    
    tps = np.cumsum(hist_gen[::-1])[::-1]
    fps = np.cumsum(hist_imp[::-1])[::-1]
    
    tpr = tps / tps[0] if tps[0] > 0 else tps * 0
    fpr = fps / fps[0] if fps[0] > 0 else fps * 0
    
    fpr = fpr[::-1]
    tpr = tpr[::-1]
    
    roc_auc = np.trapz(tpr, fpr)
    
    valid_idx = np.where(fpr <= target_far)[0]
    tar = tpr[valid_idx[-1]] * 100 if len(valid_idx) > 0 else 0.0
    
    return fpr, tpr, roc_auc, tar


def plot_individual_heatmaps(results):
    """2-row x 4-col EER heatmaps for 8 models, each subplot with a distinct color scheme"""
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    axes = axes.flatten()
    model_names = list(results.keys())
    
    # 8 carefully selected distinct academic gradient colormaps
    cmaps = ['Reds', 'Blues', 'Greens', 'Purples', 'Oranges', 'YlGnBu', 'YlOrRd', 'GnBu']
    
    for i, name in enumerate(model_names):
        ax = axes[i]
        sns.heatmap(results[name]['df'] * 100, annot=True, fmt=".2f", cmap=cmaps[i], ax=ax, cbar=True, cbar_kws={'label': 'EER (%)'})
        ax.set_title(name, fontsize=14, fontweight='bold')
    
    plt.suptitle("Cross-Spectral EER (%) Matrix Comparison", fontsize=18, y=0.98, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'EER.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_all_roc(results):
    """Plot all 8 models' ROC curves on a single figure with distinct colors and line styles"""
    plt.figure(figsize=(10, 8))
    model_names = list(results.keys())
    
    # 1. Classic academic journal high-visibility colors
    colors = ['#E64B35FF', '#4DBBD5FF', '#00A087FF', '#3C5488FF', '#F39B7FFF', '#8491B4FF', '#7E6148FF', '#DC0000FF']

    # 2. Carefully designed 8 distinct dashed line styles (supports black-and-white printing)
    # (3, 5) = 3pt solid, 5pt blank; (1, 1) = dense dotted; (5, 5, 1, 5) = classic dash-dot-dot
    linestyles = [
                                  # 1. Solid
        '--',                         # 2. Standard dashed
        ':',                          # 3. Standard dotted
        '-.',                         # 4. Standard dash-dot
        (0, (5, 5)),                  # 5. Sparse dashed (5pt solid, 5pt blank)
        (0, (1, 2)),                  # 6. Dense dotted (1pt solid, 2pt blank)
        (0, (5, 2, 1, 2)),            # 7. Tight dash-dot (5pt long, 2pt blank, 1pt short, 2pt blank)
        (0, (3, 5, 1, 5, 1, 5)),       # 8. Double-dot dashed (3pt, 5pt, 1pt, 5pt, 1pt, 5pt)
        '-'
    ]

    for i, name in enumerate(model_names):
        data = results[name]
        # Apply color and line style, with increased line width (linewidth=2.5) for clearer dash details
        plt.plot(data['fpr'], data['tpr'], 
                 label=f"{name} (AUC={data['auc']:.4f})", 
                 color=colors[i], 
                 linestyle=linestyles[i], 
                 linewidth=2.5)
                
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xscale('log') 
    plt.xlim([1e-7, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Log Scale)', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Global Cross-Spectral ROC Curve Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'ROC.pdf'), dpi=300)
    plt.close()

def plot_all_tar(results, target_far=1e-6):
    """Standalone bar chart: TAR comparison across 8 models under strict FAR"""
    model_names = list(results.keys())
    tars = [results[name]['tar'] for name in model_names]
    
    # 8-color palette synchronized with ROC color scheme
    colors = ['#E64B35FF', '#4DBBD5FF', '#00A087FF', '#3C5488FF', '#F39B7FFF', '#8491B4FF', '#7E6148FF', '#DC0000FF']
    
    plt.figure(figsize=(13, 6))
    bars = plt.bar(model_names, tars, width=0.45, color=colors, edgecolor='black', zorder=3)
    
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                 f'{bar.get_height():.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
                     
    plt.ylabel(f'TAR (%) @ FAR={target_far}', fontweight='bold', fontsize=12)
    plt.title(f'TAR Comparison Under Strict Security (FAR = {target_far})', fontweight='bold', fontsize=14)
    plt.xticks(fontweight='bold', fontsize=10, rotation=15)  # Slight rotation to prevent label overlap
    
    plt.ylim([0, max(tars) + 15 if max(tars) + 15 <= 100 else 105])
    plt.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'TAR.pdf'), dpi=300)
    plt.close()


def plot_all_score_distributions(results):
    """2-row x 4-col matching score distribution plots for 8 models, color scheme matching the heatmaps"""
    model_names = list(results.keys())
    
    # Modified: changed from 2x3 to 2x4 matrix to perfectly accommodate 8 models
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()
    
    # 8 core contrast colors consistent with the heatmap tone
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'teal', 'chocolate', 'crimson']
    
    for i, name in enumerate(model_names):
        ax = axes[i]
        color = colors[i]
        
        def get_clean_data(label):
            s = results[name]['y_score_sub'][results[name]['y_true_sub'] == label]
            return s[np.isfinite(s)]

        gen, imp = get_clean_data(1), get_clean_data(0)

        def get_kde_boundary(data, ax, col, linestyle, lbl, is_gen):
            if len(data) == 0: return 0
            sns.kdeplot(data, ax=ax, color=col, ls=linestyle, label=lbl, linewidth=2)
            line = ax.get_lines()[-1]
            x, y = line.get_data()
            threshold = y.max() * 0.00005 
            idx = np.where(y >= threshold)[0]
            return x[idx[0]] if is_gen else x[idx[-1]]

        # Plot curves and capture boundary intersection points
        imp_r = get_kde_boundary(imp, ax, color, '-', 'Impostor (Diff)', False)
        gen_l = get_kde_boundary(gen, ax, color, '--', 'Genuine (Same)', True)

        # Draw intersection point markers
        ax.scatter([imp_r, gen_l], [0, 0], color=color, s=50, zorder=6, edgecolors='white', linewidth=1)

        # Annotate intersection point values
        ax.annotate(f'{imp_r:.2f}', xy=(imp_r, 0), xytext=(0, 24), 
                    textcoords='offset points', color=color, ha='center', fontweight='bold', fontsize=9)
        ax.annotate(f'{gen_l:.2f}', xy=(gen_l, 0), xytext=(0, -24), 
                    textcoords='offset points', color=color, ha='center', fontweight='bold', fontsize=9)

        gap = gen_l - imp_r
        info_box = f"Margin Gap: {gap:.4f}"
        
        ax.text(0.05, 0.95, info_box, transform=ax.transAxes, verticalalignment='top',
                fontsize=11, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.7))

        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.axhline(0, color='black', lw=1, zorder=5)
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.legend(loc='upper right')
        
    plt.suptitle("Matching Score Distributions & Margin Gaps Comparison", fontsize=18, y=0.98, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'IMP.pdf'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    print("============== Starting streaming read and metric computation (one model at a time) ==============")
    results = {}
    sub_sample = 50000  # Sample size for distribution plots and reducing memory overhead
    total_files = len(PKL_FILES)
    
    # Strictly follow index order for streaming unpack and naming
    for idx, pkl_path in enumerate(PKL_FILES):
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"File not found: {pkl_path}")
            
        current_name = NEW_MODEL_NAMES[idx]
        print(f" [+] Loading and streaming model [{idx+1}/{total_files}]: {current_name} ...")
        
        with open(pkl_path, 'rb') as f:
            res = pickle.load(f)
        
        # Dynamic compatibility for single-model cache dict reading
        model_key = list(res.keys())[0]
        data = res[model_key]
        
        # Ultra-fast non-sorting matrix computation core
        fpr, tpr, auc_val, tar = ultra_fast_roc_auc_tar(data['y_true'], data['y_score'], target_far=1e-6)
        
        # Cache lightweight core plotting assets
        results[current_name] = {
            'df': data['df'], 
            'fpr': fpr, 'tpr': tpr, 'auc': auc_val, 'tar': tar,
            'y_true_sub': data['y_true'][:sub_sample].copy(),
            'y_score_sub': data['y_score'][:sub_sample].copy()
        }
        
        # Immediately cut large array memory references to prevent memory surge
        del data['y_true']
        del data['y_score']
        del res
        gc.collect() 

    print("\n============== Data streaming complete, generating independent model academic charts ==============")
    
    print("Generating [1/4]: Independent 8-model distinct-color EER heatmap matrix...")
    plot_individual_heatmaps(results)

    print("Generating [2/4]: Unified-coordinate 8-model ROC curve comparison...")
    plot_all_roc(results)
    
    print("Generating [3/4]: Independent model TAR bar chart comparison...")
    plot_all_tar(results, target_far=1e-6)

    print("Generating [4/4]: Independent 8-model matching score distribution & Margin Gap plot...")
    plot_all_score_distributions(results)

    print(f"\n=================== All done! ===================")
    print(f"All high-resolution single-model academic figures saved to:\n{SAVE_DIR}")

if __name__ == "__main__":
    main()