import random

import torch
import numpy as np
from tensorflow.python.framework.random_seed import set_seed
from torch.utils.data import Dataset, DataLoader
from msa_gcn import MSA_GCN
from preprocess import SkeletonPreprocessor
import os
from typing import Dict, Tuple
import torch.nn as nn
import torch.nn.functional as F

class SkeDataset(Dataset):
    def __init__(self, data_dir : str, seqlen : int = 60):
        self.data_dir = data_dir
        self.seqlen = seqlen
        self.preprocessor = SkeletonPreprocessor(seqlen)

        self.samples = []
        for sub in os.listdir(data_dir):
            sub_dir = os.path.join(data_dir, sub)
            if os.path.isdir(sub_dir):
                for seq in os.listdir(sub_dir):
                    seq_dir = os.path.join(sub_dir, seq)
                    if os.path.isdir(seq_dir):
                        jfiles = sorted([
                            os.path.join(seq_dir, f)
                            for f in os.listdir(seq_dir)
                            if f.endswith('.json')
                        ])
                        if jfiles:
                            self.samples.append(jfiles)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx : int) -> torch.Tensor:
        jfiles = self.samples[idx]

        ske_seq = self.preprocessor.sequence(jfiles)
        ske_seq = self.preprocessor.normalize(ske_seq)

        return torch.from_numpy(ske_seq).permute(2, 0, 1).float()

def extract(model : nn.Module,
            dataloader : DataLoader,
            device : torch.device,
            feature_dim : int = 64) -> Dict[str, torch.Tensor]:
    model.eval()
    proto = {'global' : None}
    total_sam = 0

    with torch.no_grad():
        for inputs in dataloader:
            if inputs.nelement() == 0:
                continue

            inputs = inputs.to(device)

            features = model.ini[0](inputs.squeeze(-1))
            for gcn in model.asst_gcn[:1]:
                features = gcn(features)

            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.squeeze(-1).squeeze(-1)

            if features.shape[-1] == feature_dim:
                bat_mean = features.mean(dim = 0)
                if proto['global'] is None:
                    proto['global'] = bat_mean
                    total_sam = features.size(0)
                else:
                    total_sam += features.size(0)
                    proto['global'] = (
                        proto['global'] * (total_sam - features.size(0))
                        + bat_mean * features.size(0)
                    ) / total_sam

    if proto['global'] is None:
        proto['global'] = torch.zeros(feature_dim).to(device)
        print("警告：未提取到有效特征，返回零值原型")

    return proto

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(42)
    data_dir = 'data'
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    dataset = SkeDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False)

    model = MSA_GCN(class_num = 4).to(device)

    print("提取骨骼原型...")
    prototypes = extract(model, dataloader, device)

    torch.save(prototypes, 'prototypes/skeleton_prototypes.pth')
    print(f"骨骼原型已保存到 prototypes/skeleton_prototypes.pth")
    print(f"提取的原型特征维度: {prototypes['global'].shape}")


if __name__ == '__main__':
    main()
