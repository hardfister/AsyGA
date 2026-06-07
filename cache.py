import pickle
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# ================= 导入你的模型和数据集 =================
from models.ccnet1 import ccnet
from models import MyDataset

# ================= 1. 核心配置区 =================
# 输出数据的保存目录
SAVE_DIR = r'save\dp'
#SAVE_DIR = r'save\AsyGA'
os.makedirs(SAVE_DIR, exist_ok=True)

# 只测试这一个模型 (如需测试Base，请更换路径和名称)
MODEL_PATHS = {
    'Base': r'weightdp\checkpoint\net_params_best.pth'
    #,'ours': r'save\checkpoint\AsyGA\AsyGA.pth'
}

# 4个波段对应的数据集 txt 路径 (请确保这些txt是测试集数据)
BAND_TXT_FILES = {
    'NIR':   './datapolyu/train_NIR.txt',
    'Red':   './datapolyu/train_RED.txt',
    'Green': './datapolyu/train_GREEN.txt',
    'Blue':  './datapolyu/train_BLUE.txt'
}
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
        # 调用模型的特征提取方法
        feature = model.getFeatureCode(img)
        # L2 归一化
        feature = torch.nn.functional.normalize(feature, p=2, dim=1) 
        all_features.append(feature.cpu().numpy())
        all_labels.append(target.numpy())
    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)


def evaluate_model(model_name, model_path, device):
    print(f"\n[{model_name}] 开始加载并提取特征...")
    model = ccnet(num_classes=500).to(device)
    
    # 1. 加载权重字典
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    
    # 2. 替换字典中不匹配的键名
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace('se_mlp', 'fc') # 解决名字不匹配的问题
        new_state_dict[new_k] = v
        
    # 3. 加入 shape 检查，丢弃分类数不一致的最后一层权重
    model_dict = model.state_dict()
    filtered_dict = {
        k: v for k, v in new_state_dict.items() 
        if k in model_dict and v.shape == model_dict[k].shape
    }
    model.load_state_dict(filtered_dict, strict=False)
    model.eval()

    bands = list(BAND_TXT_FILES.keys())
    features_dict, labels_dict = {}, {}
    
    # 提取各波段特征
    for band, path in BAND_TXT_FILES.items():
        feat, lbl = extract_features(model, path, device)
        features_dict[band] = feat
        labels_dict[band] = lbl

    num_bands = len(bands)
    eer_matrix = np.zeros((num_bands, num_bands))
    all_y_true, all_y_score = [], []

    print(f"[{model_name}] 计算跨光谱匹配矩阵...")
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
    
    # 缓存文件路径
    CACHE_FILE = os.path.join(SAVE_DIR, 'extracted_features_cache.pkl')
    
    # 如果更新了权重文件想强制重新运行，请改为 True
    FORCE_RECALCULATE = False 

    # =========================================================
    # 阶段 1：数据获取与终端打印
    # =========================================================
    if os.path.exists(CACHE_FILE) and not FORCE_RECALCULATE:
        print(f"============== 发现缓存文件，直接跳过模型推理！ ==============")
        print(f"正在加载缓存: {CACHE_FILE}")
        with open(CACHE_FILE, 'rb') as f:
            results = pickle.load(f)
        
        # 从缓存中提取并打印 EER
        for name, data in results.items():
            print(f"\n=================== [{name}] EER 矩阵结果 (来自缓存) ===================")
            print(data['df'])
            print("=================================================================\n")
    else:
        print(f"============== 开始模型推理与特征提取 | 采用设备: {device} ==============")
        results = {}
        for name, path in MODEL_PATHS.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"找不到模型文件: {path}")
                
            df, y_true, y_score, feat_dict, lbl_dict = evaluate_model(name, path, device)
            results[name] = {
                'df': df, 'y_true': y_true, 'y_score': y_score,
                'feat_dict': feat_dict, 'lbl_dict': lbl_dict
            }
            
            # 1. 核心要求：将 EER 矩阵直接打印在终端
            print(f"\n=================== [{name}] EER 矩阵结果 ===================")
            print(df)
            print("=========================================================\n")
            
            # 2. 核心要求：保存该模型的 eer csv 文件
            csv_name = 'EER_Matrix_' + name.replace(" ", "_") + '.csv'
            df.to_csv(os.path.join(SAVE_DIR, csv_name))

        # 3. 核心要求：保存该模型的 pkl 缓存文件
        print(f"[保存缓存] 正在将数据保存到 {CACHE_FILE} ...")
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(results, f)
        print("缓存保存成功！")

    print(f"全部完成！数据文件已保存在：{SAVE_DIR}")

if __name__ == "__main__":
    main()