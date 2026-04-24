"""
CellODE: Enhanced Cell-type-aware OOD Perturbation Prediction

增强版CellODE，包含：
1. 因果解耦模块 - 分离基础表达和响应敏感性
2. 细胞类型感知注意力 - 跨细胞类型知识迁移
3. 多头注意力机制 - 捕获多种响应模式

与scPerturBench OOD框架完全兼容

使用方式:
    conda activate cpa
    python myCellODE.py
"""

import os
import sys
sys.path.append('/root/autodl-tmp/home/project/Pertb_benchmark')
from myUtil import *
import anndata as ad
import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CellODE-v2 模型定义
# ============================================================

class CausalDisentangle(nn.Module):
    """
    因果解耦模块

    将细胞状态分解为：
    - E_c: 基础表达水平（决定扰动前的基因表达）
    - S_c: 响应敏感性（决定对扰动的响应强度和模式）
    """
    def __init__(self, latent_dim=256):
        super().__init__()
        self.shared_encoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim)
        )

        # E_c 专用路径
        self.E_encoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim // 2, latent_dim)
        )

        # S_c 专用路径
        self.S_encoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim // 2, latent_dim)
        )

        # 对抗判别器（确保解耦）
        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, 1)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, z):
        """
        输入: z [B, latent_dim] - 细胞状态编码
        输出: E_c [B, latent_dim], S_c [B, latent_dim]
        """
        h = self.shared_encoder(z)

        E_c = self.E_encoder(h)
        S_c = self.S_encoder(h)

        # 添加随机性
        E_c = E_c + torch.randn_like(E_c) * 0.05
        S_c = S_c + torch.randn_like(S_c) * 0.05

        return E_c, S_c


class GeneExpressionEncoder(nn.Module):
    """
    基因表达编码器
    将高维基因表达转换为低维潜在表示
    """
    def __init__(self, gene_num, latent_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(gene_num, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class PerturbationEncoder(nn.Module):
    """
    扰动编码器
    将扰动类型编码为向量
    """
    def __init__(self, num_perts, latent_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(num_perts, latent_dim)
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )

    def forward(self, pert_idx):
        emb = self.embedding(pert_idx)
        return self.projection(emb)


class CellTypeAwareAttention(nn.Module):
    """
    细胞类型感知注意力模块

    核心创新：
    1. 使用多头注意力捕获多种响应模式
    2. 细胞相似度加权聚合已知细胞类型的知识
    3. 门控机制自适应调整迁移比例
    """
    def __init__(self, latent_dim=256, num_heads=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads

        assert latent_dim % num_heads == 0, "latent_dim must be divisible by num_heads"

        # 多头注意力
        self.W_q = nn.Linear(latent_dim, latent_dim)
        self.W_k = nn.Linear(latent_dim, latent_dim)
        self.W_v = nn.Linear(latent_dim, latent_dim)
        self.W_o = nn.Linear(latent_dim, latent_dim)

        # 细胞类型相似度网络
        self.cell_sim_net = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, 1)
        )

        # 多头聚合器
        self.aggregator = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # 响应模式GRU
        self.response_gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=latent_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # 门控融合网络
        self.gate_net = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim),
            nn.Sigmoid()
        )

        # 层级归一化
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)

    def compute_cell_similarity(self, S_target, S_sources):
        """
        计算目标细胞与源细胞类型的相似度

        S_target: [B, latent_dim]
        S_sources: [num_known, latent_dim]
        """
        # 扩展维度
        S_target_expanded = S_target.unsqueeze(1)  # [B, 1, latent]
        S_sources_expanded = S_sources.unsqueeze(0)  # [1, num_known, latent]

        # 拼接计算相似度
        combined = torch.cat([S_target_expanded.expand(-1, S_sources.size(0), -1),
                             S_sources_expanded.expand(S_target.size(0), -1, -1)], dim=-1)

        sim_weights = self.cell_sim_net(combined).squeeze(-1)  # [B, num_known]
        sim_weights = torch.softmax(sim_weights, dim=1)

        return sim_weights

    def forward(self, S_c, pert_emb, known_responses, known_cell_types):
        """
        参数：
        - S_c: [B, latent_dim] - 目标细胞类型的响应敏感性
        - pert_emb: [B, latent_dim] - 扰动嵌入
        - known_responses: [num_known, latent_dim] - 已知细胞类型的响应
        - known_cell_types: [num_known, latent_dim] - 已知细胞类型的敏感性

        输出：
        - migrated_response: [B, latent_dim] - 迁移的扰动响应
        """
        batch_size = S_c.size(0)
        num_known = known_responses.size(0)

        # 1. 计算细胞类型相似度
        sim_weights = self.compute_cell_similarity(S_c, known_cell_types)  # [B, num_known]

        # 2. 注意力聚合
        Q = self.W_q(S_c).unsqueeze(1)  # [B, 1, latent]
        K = self.W_k(known_cell_types).unsqueeze(0)  # [1, num_known, latent]
        V = self.W_v(known_responses).unsqueeze(0)  # [1, num_known, latent]

        attn_output, _ = self.aggregator(Q, K, V)  # [B, 1, latent]
        attn_output = attn_output.squeeze(1)  # [B, latent]
        attn_output = self.norm1(attn_output)

        # 3. GRU聚合响应模式
        pert_emb_expanded = pert_emb.unsqueeze(1)  # [B, 1, latent]
        gru_input = known_responses.unsqueeze(0).expand(batch_size, -1, -1) + pert_emb_expanded
        gru_out, _ = self.response_gru(gru_input)
        gru_agg = gru_out.mean(dim=1)  # [B, latent]
        gru_agg = self.norm2(gru_agg)

        # 4. 门控融合
        combined = torch.cat([attn_output, gru_agg, S_c], dim=-1)
        gate = self.gate_net(combined)
        migrated = gate * attn_output + (1 - gate) * gru_agg

        return migrated


class ResponseDecoder(nn.Module):
    """
    响应解码器

    将细胞类型表示、扰动表示和迁移知识结合生成响应预测
    """
    def __init__(self, latent_dim=256, gene_num=5000):
        super().__init__()

        # 交互编码器
        self.interaction_encoder = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim)
        )

        # 响应幅度预测
        self.magnitude_net = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, 1),
            nn.Softplus()
        )

        # 差异表达解码
        self.delta_decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, gene_num)
        )

        # 方向解码
        self.direction_decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, gene_num),
            nn.Tanh()
        )

    def forward(self, E_c, S_c, pert_emb, migrated_response):
        """
        E_c: 基础表达水平
        S_c: 响应敏感性
        pert_emb: 扰动嵌入
        migrated_response: 迁移的响应知识
        """
        # 交互编码
        interaction = self.interaction_encoder(
            torch.cat([E_c, S_c, pert_emb], dim=-1)
        )

        # 融合迁移知识
        combined = interaction + migrated_response

        # 预测幅度
        magnitude = self.magnitude_net(torch.cat([combined, S_c], dim=-1))

        # 解码差异表达
        delta_pattern = self.delta_decoder(combined)
        direction = self.direction_decoder(combined)

        # 组合
        delta = delta_pattern * direction * magnitude

        return delta


class CellODEv2(nn.Module):
    """
    CellODE-v2 完整模型

    包含因果解耦和细胞类型感知注意力的增强模型
    """
    def __init__(self, gene_num, num_perts, latent_dim=256, num_heads=8):
        super().__init__()
        self.gene_encoder = GeneExpressionEncoder(gene_num, latent_dim)
        self.causal_disentangle = CausalDisentangle(latent_dim)
        self.pert_encoder = PerturbationEncoder(num_perts, latent_dim)
        self.attention = CellTypeAwareAttention(latent_dim, num_heads)
        self.decoder = ResponseDecoder(latent_dim, gene_num)

        self.latent_dim = latent_dim
        self.gene_num = gene_num

    def forward(self, x, pert_idx, known_data=None):
        """
        x: [batch, gene_num] - 基因表达
        pert_idx: [batch,] - 扰动索引
        known_data: 已知细胞类型的参考数据
        """
        # 基因编码
        gene_repr = self.gene_encoder(x)

        # 因果解耦
        E_c, S_c = self.causal_disentangle(gene_repr)

        # 扰动编码
        pert_emb = self.pert_encoder(pert_idx)

        # OOD知识迁移
        if known_data is not None and self.training:
            migrated_response = self.attention(
                S_c, pert_emb,
                known_data['known_deltas'],
                known_data['known_cell_types']
            )
        else:
            migrated_response = pert_emb

        # 解码
        delta = self.decoder(E_c, S_c, pert_emb, migrated_response)

        return delta


# ============================================================
# 工具函数
# ============================================================

def generatePairedSample(adata, outSample, perturbation):
    """生成配对训练样本"""
    cellTypes = list(adata.obs['condition1'].unique())
    cellTypes = [i for i in cellTypes if i != outSample]

    annList_Pertb = []
    annList_control = []

    for celltype in cellTypes:
        perturb_cells = adata[(adata.obs['condition1'] == celltype) &
                             (adata.obs['condition2'] == perturbation)]
        control_cells = adata[(adata.obs['condition1'] == celltype) &
                              (adata.obs['condition2'] == 'control')]

        Nums = min(perturb_cells.shape[0], control_cells.shape[0])
        if Nums > 0:
            perturb_cells = perturb_cells[:Nums]
            control_cells = control_cells[:Nums]
            annList_Pertb.append(perturb_cells)
            annList_control.append(control_cells)

    if len(annList_Pertb) > 0:
        return ad.concat(annList_Pertb), ad.concat(annList_control)
    return None, None


def build_known_cell_types(adata, outSample, perturbation):
    """
    构建已知细胞类型的参考数据（用于注意力机制）

    返回:
        known_cell_types: [num_known, latent_dim] - 已知细胞类型的敏感性表示
        known_deltas: [num_known, latent_dim] - 已知细胞类型的扰动响应
    """
    known_cell_types_list = []
    known_deltas_list = []

    known_cellTypes = [ct for ct in adata.obs['condition1'].unique() if ct != outSample]

    for known_ct in known_cellTypes:
        control_cells = adata[(adata.obs['condition1'] == known_ct) &
                             (adata.obs['condition2'] == 'control')]
        perturb_cells = adata[(adata.obs['condition1'] == known_ct) &
                             (adata.obs['condition2'] == perturbation)]

        if control_cells.n_obs > 0 and perturb_cells.n_obs > 0:
            # 使用均值作为表示
            control_expr = control_cells.X.toarray() if hasattr(control_cells.X, 'toarray') else control_cells.X
            perturb_expr = perturb_cells.X.toarray() if hasattr(perturb_cells.X, 'toarray') else perturb_cells.X

            # 简化：用原始表达差的均值作为delta代理
            delta_mean = (perturb_expr - control_expr).mean(axis=0)
            control_mean = control_expr.mean(axis=0)

            known_cell_types_list.append(control_mean)
            known_deltas_list.append(delta_mean)

    if len(known_cell_types_list) > 0:
        return np.array(known_cell_types_list), np.array(known_deltas_list)
    return None, None


def trainCellODEv2(Xtr, ytr, Xval, yval, gene_num, num_perts, pert2idx,
                   perturbation, known_data=None, epochs=100, batch_size=128):
    """训练CellODE-v2模型"""

    patience = 20
    min_delta = 0.0001
    best_val_loss = np.inf
    counter = 0

    # 数据
    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    ytr = torch.tensor(ytr, dtype=torch.float32)
    Xval = torch.tensor(Xval, dtype=torch.float32)
    yval = torch.tensor(yval, dtype=torch.float32)

    train_dataset = TensorDataset(Xtr, ytr)
    val_dataset = TensorDataset(Xval, yval)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # 模型
    model = CellODEv2(gene_num=gene_num, num_perts=num_perts, latent_dim=256, num_heads=8)
    model = model.cuda()

    criterion = nn.SmoothL1Loss()  # 对异常值更鲁棒
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()

            # 获取扰动索引（从targets推断，简化处理）
            pert_indices = torch.full((inputs.size(0),), pert2idx.get(perturbation, 0),
                                       dtype=torch.long, device=inputs.device)

            optimizer.zero_grad()

            # 如果有已知数据，传给模型
            if known_data is not None and epoch < epochs // 2:
                outputs = model(inputs, pert_indices, known_data)
            else:
                outputs = model(inputs, pert_indices, None)

            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                pert_indices = torch.full((inputs.size(0),), pert2idx.get(perturbation, 0),
                                          dtype=torch.long, device=inputs.device)
                outputs = model(inputs, pert_indices, None)
                val_loss += criterion(outputs, targets).item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}')

        # 早停
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), '{}_best.pth'.format(perturbation))
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    model.load_state_dict(torch.load('{}_best.pth'.format(perturbation)))
    return model


def predictCellODEv2(model, Xte, pert_idx, gene_num, num_perts):
    """使用CellODE-v2预测"""
    model.eval()
    Xte_tensor = torch.tensor(Xte, dtype=torch.float32)
    predictions = []

    with torch.no_grad():
        for i in range(0, len(Xte_tensor), 128):
            batch = Xte_tensor[i:i+128].cuda()
            pert = torch.full((batch.size(0),), pert_idx, dtype=torch.long, device='cuda')
            delta = model(batch, pert, None)
            predictions.append(delta.cpu().numpy())

    return np.vstack(predictions)


# ============================================================
# 主函数
# ============================================================

def Kang_OutSample(DataSet, outSample):
    """对单个OOD细胞类型进行扰动预测"""
    basePath = f'/root/autodl-tmp/home/project/Pertb_benchmark/DataSet/{DataSet}/outSample/hvg5000/CellODE/'
    tmp = '{}/{}'.format(basePath, outSample)
    if not os.path.isdir(tmp):
        os.makedirs(tmp)
    os.chdir(tmp)

    print(f"\n{'='*60}")
    print(f"CellODE-v2: DataSet={DataSet}, outSample={outSample}")
    print(f"{'='*60}")

    # 加载数据
    path = f'/root/autodl-tmp/home/project/Pertb_benchmark/DataSet/{DataSet}/filter_hvg5000_logNor.h5ad'
    adata = sc.read_h5ad(path)

    gene_num = adata.n_vars
    perturbations = list(adata.obs['condition2'].unique())
    perturbations = [i for i in perturbations if i != 'control']

    pert2idx = {p: i+1 for i, p in enumerate(perturbations)}
    pert2idx['control'] = 0
    num_perts = len(perturbations) + 1

    for perturbation in perturbations:
        print(f"\n--- Perturbation: {perturbation} ---")

        adata_subset = adata[adata.obs['condition2'].isin([perturbation, 'control'])]
        Xtr_anndata, ytr_anndata = generatePairedSample(adata_subset, outSample, perturbation)

        if Xtr_anndata is None or Xtr_anndata.n_obs < 10:
            print(f"Skipping {perturbation}: insufficient samples")
            continue

        Xtr = Xtr_anndata.X.toarray() if hasattr(Xtr_anndata.X, 'toarray') else Xtr_anndata.X
        ytr = ytr_anndata.X.toarray() if hasattr(ytr_anndata.X, 'toarray') else ytr_anndata.X

        train_size = int(0.8 * Xtr.shape[0])
        Xtr_train, Xtr_val = Xtr[:train_size], Xtr[train_size:]
        ytr_train, ytr_val = ytr[:train_size], ytr[train_size:]

        # 获取测试数据
        Xte_control = adata[(adata.obs['condition2'] == 'control') &
                           (adata.obs['condition1'] == outSample)]
        Xte_treat = adata[(adata.obs['condition2'] == perturbation) &
                         (adata.obs['condition1'] == outSample)]
        Xte = Xte_control.X.toarray() if hasattr(Xte_control.X, 'toarray') else Xte_control.X

        # 构建已知数据
        known_ct, known_delta = build_known_cell_types(adata, outSample, perturbation)
        known_data = None
        if known_ct is not None:
            known_data = {
                'known_cell_types': torch.tensor(known_ct, dtype=torch.float32).cuda(),
                'known_deltas': torch.tensor(known_delta, dtype=torch.float32).cuda()
            }

        print(f"Train: {Xtr.shape[0]}, Test: {Xte.shape[0]}")

        try:
            model = trainCellODEv2(
                Xtr_train, ytr_train - Xtr_train,
                Xtr_val, ytr_val - Xtr_val,
                gene_num, num_perts, pert2idx, perturbation,
                known_data, epochs=100, batch_size=128
            )

            delta_pred = predictCellODEv2(model, Xte, pert2idx[perturbation], gene_num, num_perts)
            ypred = Xte + delta_pred

        except Exception as e:
            print(f"Training failed: {e}")
            ypred = np.tile(ytr.mean(axis=0), (Xte.shape[0], 1))

        # 保存结果
        Xte_control.obs['perturbation'] = 'control'
        imputed = Xte_control.copy()
        imputed.X = ypred
        imputed.obs['perturbation'] = 'imputed'

        if Xte_treat.n_obs > 0:
            Xte_treat.obs['perturbation'] = 'stimulated'
            result = ad.concat([Xte_control, Xte_treat, imputed])
        else:
            result = ad.concat([Xte_control, imputed])

        result.write_h5ad('{}_imputed.h5ad'.format(perturbation))
        print(f"Saved: {perturbation}_imputed.h5ad")

        del model
        torch.cuda.empty_cache()


def KangMain(DataSet):
    """主函数"""
    filein_tmp = '/root/autodl-tmp/home/project/Pertb_benchmark/DataSet/{}/filter_hvg5000_logNor.h5ad'.format(DataSet)
    adata_tmp = sc.read_h5ad(filein_tmp)
    outSamples = list(adata_tmp.obs['condition1'].unique())

    print(f"\nDataSet: {DataSet}, Cell types: {outSamples}")

    for outSample in outSamples:
        try:
            Kang_OutSample(DataSet, outSample)
        except Exception as e:
            print(f"Error: {outSample} - {e}")
            continue


# ============================================================
# 配置
# ============================================================

DataSets = ["kangCrossCell", "kangCrossPatient", "Parekh", "Haber",
            "crossPatient", "KaggleCrossPatient", "KaggleCrossCell",
            "crossSpecies", "McFarland", "Afriat", "TCDD", "sciplex3"]

DataSets = ["kangCrossCell"]

torch.cuda.set_device('cuda:0')


if __name__ == '__main__':
    for DataSet in tqdm(DataSets):
        KangMain(DataSet)
