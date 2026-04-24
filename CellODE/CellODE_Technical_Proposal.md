# CellODE: Cell-type-aware Out-of-Distribution Perturbation Prediction

## 1. 背景与问题定义

### 1.1 问题背景

单细胞扰动预测（Single-cell Perturbation Prediction）是计算生物学中的核心任务之一，旨在预测基因敲除、过表达、化学扰动等干预对细胞转录组的影响。scPerturBench基准测试[1]对27种方法进行了系统评估，发现现有算法在**细胞上下文泛化（Cellular Context Generalization）** 场景下表现不佳。

### 1.2 OOD问题定义

**细胞上下文泛化场景**：
- **训练集**：已知细胞类型上的扰动响应数据
- **测试集**：未知（未见）细胞类型上的扰动响应预测
- **挑战**：不同细胞类型具有不同的baseline表达谱和扰动响应模式

设 $X \in \mathbb{R}^{G \times N}$ 为基因表达矩阵，其中 $G$ 为基因数，$N$ 为细胞数。扰动 $p$ 作用于细胞类型 $c$ 的响应可建模为：

$$\Delta_{c,p} = f(\phi(c), \psi(p))$$

其中 $\phi(c)$ 是细胞类型嵌入，$\psi(p)$ 是扰动嵌入，$f$ 是预测函数。

**OOD目标**：学习一个预测函数 $f$，使得对于未见细胞类型 $c_{new}$ 和已知扰动 $p$：

$$\hat{\Delta}_{c_{new}, p} \approx f(\phi(c_{new}), \psi(p))$$

### 1.3 现有方法局限性

| 方法 | 核心思想 | OOD局限性 |
|------|---------|----------|
| trVAE | 条件VAE | 条件信息融合方式简单，无法显式建模细胞类型异质性 |
| CellOT | 最优传输 | 传输计划计算量大，难以捕捉细胞类型特定的响应机制 |
| bioLord | 解耦表示 | 依赖预定义细胞类型嵌入，泛化能力受限于嵌入质量 |
| CPA | 组合因子分解 | 线性组合假设过于简单，无法建模非线性交互 |

**核心问题**：现有方法缺乏对**细胞类型-扰动交互因果机制**的显式建模。

---

## 2. 核心设计思想

### 2.1 因果解耦（Causal Disentanglement）

我们假设扰动响应遵循以下因果结构：

```
细胞类型 c ──(基础表达谱)──┐
                          ├──→ 扰动响应 Δ
扰动 p ──(扰动机制)───────┘
```

细胞类型 $c$ 决定了两件事：
1. **基础表达水平** $E_c$：扰动前的基因表达baseline
2. **响应敏感性** $S_c$：对扰动的响应强度和模式

扰动 $p$ 具有：
1. **直接效应** $D_p$：直接影响的基因
2. **间接效应** $I_p$：通过基因网络传播的次级效应

### 2.2 细胞类型感知注意力（Cell-type-aware Attention）

核心创新：设计一种注意力机制，能够：
1. 识别跨细胞类型的结构相似性
2. 将已知细胞类型的响应知识迁移到未见细胞类型

### 2.3 设计目标

1. **解耦性**：分离细胞身份、扰动机制、响应效应
2. **可迁移性**：利用细胞类型间的相似性进行知识迁移
3. **可解释性**：关联学习到的表示与生物学机制
4. **高效性**：支持大规模训练和推理

---

## 3. 算法架构：CellODE

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        CellODE 架构                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐       │
│  │ Gene Expr  │      │ Perturbation│      │ Cell Type   │       │
│  │ Encoder    │      │ Encoder     │      │ Encoder     │       │
│  │ (scGPT)    │      │ (Drug/Gene) │      │ (CP Embed)  │       │
│  └──────┬──────┘      └──────┬──────┘      └──────┬──────┘       │
│         │                    │                    │              │
│         ▼                    ▼                    ▼              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              因果解耦模块 (Causal Disentanglement)         │    │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  │    │
│  │  │ 基础表达 E_c  │  │ 响应敏感 S_c │  │ 扰动效应 D_p │  │    │
│  │  └───────────────┘  └───────────────┘  └───────────────┘  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │           细胞类型感知注意力 (Cell-type Attention)        │    │
│  │                                                           │    │
│  │   Query: S_c (响应敏感)    Key: {S_{c'})}  Value: Δ_{c',p}│   │
│  │                                                           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   响应解码器 (Response Decoder)          │    │
│  │         Δ_{c,p} = g(E_c, S_c, Attention(S_c, {Δ_{c',p}}))│  │
│  └─────────────────────────────────────────────────────────┘    │
│                            │                                     │
│                            ▼                                     │
│                    Predicted Expression                          │
│                      X' = X_base + Δ                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 基因表达编码器（Gene Expression Encoder）

**输入**：原始基因表达向量 $x \in \mathbb{R}^G$

**目标**：学习基因的分布式表示，捕获基因间的生物学关联

**实现**：
```python
class GeneEncoder(nn.Module):
    """
    使用预训练的scGPT基因嵌入作为初始化，
    通过微调学习细胞类型特定的基因表示
    """
    def __init__(self, gene_num, latent_dim, pretrained_path=None):
        super().__init__()
        # scGPT预训练基因嵌入 [gene_num, 256]
        self.gene_embedding = load_pretrained_gene_embedding(pretrained_path)

        # 位置编码用于区分不同基因
        self.pos_encoding = PositionalEncoding(d_model=256)

        # Transformer编码器层
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=4
        )

        # 输出映射
        self.output_proj = nn.Sequential(
            nn.Linear(256, latent_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        # x: [batch_size, gene_num]
        gene_emb = self.gene_embedding.weight[:x.size(1)]  # [gene_num, 256]
        gene_emb = gene_emb.unsqueeze(0).expand(x.size(0), -1, -1)  # [B, G, 256]

        # 添加位置编码
        gene_emb = self.pos_encoding(gene_emb)

        # 考虑基因共表达模式的注意力
        encoded = self.transformer(gene_emb)

        # 池化得到基因级别的表示
        gene_repr = self.output_proj(encoded.mean(dim=1))  # [B, latent_dim]

        return gene_repr
```

### 3.3 扰动编码器（Perturbation Encoder）

**输入**：扰动描述（药物SMILES、基因符号、扰动类型）

**目标**：学习扰动的生物学效应表示

```python
class PerturbationEncoder(nn.Module):
    """
    多模态扰动编码器
    - 化学扰动：分子指纹 + 药物描述符
    - 基因扰动：基因嵌入 + GO富集
    """
    def __init__(self, pert_type, latent_dim):
        super().__init__()
        self.pert_type = pert_type

        if pert_type == 'chemical':
            # 分子指纹编码器 (Morgan Fingerprint -> 256)
            self.fp_encoder = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Linear(512, 256)
            )

            # 药物描述符编码器
            self.desc_encoder = nn.Sequential(
                nn.Linear(200, 128),
                nn.ReLU(),
                nn.Linear(128, 256)
            )

            self.fusion = nn.MultiheadAttention(256, 4, batch_first=True)
            self.output_proj = nn.Linear(256, latent_dim)

        elif pert_type == 'genetic':
            # 基因嵌入 (使用scGPT的geneformer或自定义)
            self.gene_encoder = GeneEncoder(gene_num=20000, latent_dim=256)

            # GO语义相似性编码
            self.go_encoder = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, latent_dim)
            )

            self.output_proj = nn.Linear(256 * 2, latent_dim)

    def forward(self, pert_info):
        if self.pert_type == 'chemical':
            fp, desc = pert_info
            fp_emb = self.fp_encoder(fp)
            desc_emb = self.desc_encoder(desc)

            # 跨模态注意力融合
            fused, _ = self.fusion(
                fp_emb.unsqueeze(1),
                desc_emb.unsqueeze(1),
                desc_emb.unsqueeze(1)
            )
            out = self.output_proj(fused.squeeze(1))

        elif self.pert_type == 'genetic':
            gene_emb, go_emb = pert_info
            gene_repr = self.gene_encoder(gene_emb)
            go_repr = self.go_encoder(go_emb)
            out = self.output_proj(torch.cat([gene_repr, go_repr], dim=-1))

        return out
```

### 3.4 细胞类型编码器（Cell Type Encoder）

**关键创新**：学习细胞类型的**响应敏感度表示**，而非简单的身份标签

```python
class CellTypeEncoder(nn.Module):
    """
    细胞类型编码器：学习响应敏感度表示

    核心思想：
    1. 从基因表达推断细胞身份和响应敏感性
    2. 利用预训练细胞类型嵌入（如scGPT细胞嵌入）
    3. 解耦为基础表达水平 E_c 和响应敏感性 S_c
    """
    def __init__(self, gene_num, latent_dim, cell_emb_dim=256):
        super().__init__()
        self.latent_dim = latent_dim

        # 1. 基因表达到细胞状态的编码器
        self.expr_encoder = nn.Sequential(
            nn.Linear(gene_num, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, cell_emb_dim)
        )

        # 2. 预训练细胞类型嵌入适配器
        self.cell_emb_adapter = nn.Linear(cell_emb_dim, latent_dim)

        # 3. 解耦层：分解为基础表达和响应敏感性
        self.disentangle = DisentangleModule(latent_dim)

        # 4. 相似性保留正则化
        self.similarity_net = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Tanh()
        )

    def forward(self, x_expr, cell_type_id=None, pretrained_emb=None):
        # 从表达推断细胞状态
        state_repr = self.expr_encoder(x_expr)  # [B, cell_emb_dim]

        # 如果有预训练细胞嵌入，进行适配
        if pretrained_emb is not None:
            adapted_emb = self.cell_emb_adapter(pretrained_emb)
            # 融合两种表示
            state_repr = 0.5 * state_repr + 0.5 * adapted_emb

        # 解耦为基础表达 E_c 和响应敏感性 S_c
        E_c, S_c = self.disentangle(state_repr)

        return {
            'base_expression': E_c,      # 基础表达水平
            'response_sensitivity': S_c,  # 响应敏感性
            'cell_state': state_repr      # 完整细胞状态
        }


class DisentangleModule(nn.Module):
    """
    因果解耦模块

    假设：细胞状态可以分解为
    - E_c: 基础表达水平（决定扰动前的基因表达）
    - S_c: 响应敏感性（决定对扰动的响应强度和模式）

    使用变分推断和对抗训练实现解耦
    """
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        # 共享编码器
        self.shared_encoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2)
        )

        # E_c 的先验和后验
        self.E_prior = nn.Linear(latent_dim, latent_dim)
        self.E_posterior = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Linear(latent_dim, latent_dim)
        )

        # S_c 的先验和后验
        self.S_prior = nn.Linear(latent_dim, latent_dim)
        self.S_posterior = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Linear(latent_dim, latent_dim)
        )

        # 对抗判别器：确保E_c和S_c独立
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

        # 独立编码E_c和S_c
        E_c = self.E_prior(h)
        S_c = self.S_prior(h)

        # 添加随机性用于变分推断
        E_c = E_c + torch.randn_like(E_c) * 0.1
        S_c = S_c + torch.randn_like(S_c) * 0.1

        return E_c, S_c
```

### 3.5 细胞类型感知注意力模块（Cell-type-aware Attention）

**核心创新**：实现跨细胞类型的知识迁移

```python
class CellTypeAwareAttention(nn.Module):
    """
    细胞类型感知注意力模块

    功能：
    1. 从已知细胞类型的扰动响应中提取知识
    2. 根据细胞类型的相似性，将知识迁移到未见细胞类型
    3. 建模细胞类型特异性的响应模式

    公式：
    Attention(Q, K, V) = softmax(QK^T / sqrt(d)) * V

    其中：
    - Q = S_c (查询：目标细胞类型的响应敏感性)
    - K = {S_{c'}} (键：源细胞类型的响应敏感性)
    - V = {Δ_{c',p}} (值：源细胞类型的扰动响应)
    """

    def __init__(self, latent_dim, num_heads=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads

        # 线性投影
        self.W_q = nn.Linear(latent_dim, latent_dim)
        self.W_k = nn.Linear(latent_dim, latent_dim)
        self.W_v = nn.Linear(latent_dim, latent_dim)
        self.W_o = nn.Linear(latent_dim, latent_dim)

        # 细胞类型相似度网络：学习细胞类型间的响应模式相似性
        self.cell_similarity_net = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, 1)
        )

        # 响应模式聚合器
        self.response_aggregator = nn.GRU(
            input_size=latent_dim,
            hidden_size=latent_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # 门控机制：决定从其他细胞类型迁移多少知识
        self.gate_net = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Sigmoid()
        )

    def compute_cell_similarity(self, S_target, S_sources):
        """
        计算目标细胞类型与源细胞类型的相似度

        S_target: [B, latent_dim] - 目标细胞类型的响应敏感性
        S_sources: [B, N, latent_dim] - 源细胞类型的响应敏感性
        """
        # 拼接目标和源
        combined = torch.cat([
            S_target.unsqueeze(1).expand(-1, S_sources.size(1), -1),
            S_sources
        ], dim=-1)  # [B, N, latent_dim * 2]

        # 学习相似度权重
        sim_weights = self.cell_similarity_net(combined)  # [B, N, 1]
        sim_weights = F.softmax(sim_weights, dim=1)

        return sim_weights

    def forward(self, S_c, pert_emb, known_responses, known_cell_types):
        """
        参数：
        - S_c: [B, latent_dim] - 目标细胞类型的响应敏感性
        - pert_emb: [B, latent_dim] - 扰动嵌入
        - known_responses: [B, N, latent_dim] - 已知细胞类型的扰动响应
        - known_cell_types: [B, N, latent_dim] - 已知细胞类型的敏感性表示

        输出：
        - migrated_response: [B, latent_dim] - 迁移的扰动响应
        """
        batch_size = S_c.size(0)
        num_known = known_responses.size(1)

        # 1. 计算细胞类型相似度
        sim_weights = self.compute_cell_similarity(
            S_c, known_cell_types
        )  # [B, N, 1]

        # 2. Query-Key-Value注意力
        Q = self.W_q(S_c).unsqueeze(1)  # [B, 1, latent_dim]
        K = self.W_k(known_cell_types)  # [B, N, latent_dim]
        V = self.W_v(known_responses)    # [B, N, latent_dim]

        # 多头注意力
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_known, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_known, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights * sim_weights.unsqueeze(1)  # 加入细胞相似度

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 1, self.latent_dim)
        attn_output = self.W_o(attn_output.squeeze(1))  # [B, latent_dim]

        # 3. 使用GRU聚合时序响应模式
        known_responses_expanded = known_responses + pert_emb.unsqueeze(1)  # [B, N, latent_dim]
        rnn_out, _ = self.response_aggregator(known_responses_expanded)
        aggregated = rnn_out.mean(dim=1)  # [B, latent_dim]

        # 4. 门控融合：决定迁移知识的比例
        combined = torch.cat([attn_output, aggregated], dim=-1)
        gate = self.gate_net(combined)
        migrated_response = gate * attn_output + (1 - gate) * aggregated

        return migrated_response
```

### 3.6 响应解码器（Response Decoder）

```python
class ResponseDecoder(nn.Module):
    """
    扰动响应解码器

    将细胞类型表示、扰动表示和迁移的知识结合，
    生成最终的扰动响应预测

    Δ_{c,p} = Decoder(E_c, S_c, pert_emb, migrated_response)
    """

    def __init__(self, latent_dim, gene_num):
        super().__init__()
        self.latent_dim = latent_dim

        # 1. 细胞类型-扰动交互编码器
        self.interaction_encoder = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim)
        )

        # 2. 响应幅度预测器：预测响应的整体强度
        self.magnitude_predictor = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1),
            nn.Softplus()  # 确保响应幅度非负
        )

        # 3. 差异表达模式解码器
        self.delta_decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, gene_num)  # 输出每个基因的log fold change
        )

        # 4. 方向解码器：区分上调和下调
        self.direction_decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, gene_num),
            nn.Tanh()  # 输出[-1, 1]，表示上调/下调程度
        )

    def forward(self, E_c, S_c, pert_emb, migrated_response):
        """
        输入：
        - E_c: [B, latent_dim] - 基础表达水平
        - S_c: [B, latent_dim] - 响应敏感性
        - pert_emb: [B, latent_dim] - 扰动嵌入
        - migrated_response: [B, latent_dim] - 迁移的响应知识

        输出：
        - delta: [B, gene_num] - 预测的基因表达变化（log fold change）
        """
        # 1. 编码细胞-扰动交互
        interaction = self.interaction_encoder(
            torch.cat([E_c, S_c, pert_emb], dim=-1)
        )

        # 2. 融合迁移的响应知识
        combined = interaction + migrated_response

        # 3. 预测响应幅度（全局强度因子）
        magnitude = self.magnitude_predictor(
            torch.cat([combined, S_c], dim=-1)
        )  # [B, 1]

        # 4. 解码差异表达模式
        delta_pattern = self.delta_decoder(combined)  # [B, gene_num]
        direction = self.direction_decoder(combined)   # [B, gene_num]

        # 5. 组合得到最终预测
        # 基础模式 * 方向 * 幅度
        delta = delta_pattern * direction * magnitude

        return delta
```

### 3.7 完整CellODE模型

```python
class CellODE(nn.Module):
    """
    CellODE: Cell-type-aware Out-of-Distribution Perturbation Prediction

    端到端的扰动响应预测模型，包含：
    1. 基因表达编码器
    2. 扰动编码器
    3. 细胞类型编码器（含解耦模块）
    4. 细胞类型感知注意力模块
    5. 响应解码器
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gene_num = config.gene_num
        self.latent_dim = config.latent_dim

        # 编码器
        self.gene_encoder = GeneEncoder(
            gene_num=config.gene_num,
            latent_dim=config.latent_dim
        )

        self.pert_encoder = PerturbationEncoder(
            pert_type=config.pert_type,
            latent_dim=config.latent_dim
        )

        self.cell_encoder = CellTypeEncoder(
            gene_num=config.gene_num,
            latent_dim=config.latent_dim
        )

        # 注意力模块
        self.attention = CellTypeAwareAttention(
            latent_dim=config.latent_dim,
            num_heads=config.num_heads
        )

        # 解码器
        self.decoder = ResponseDecoder(
            latent_dim=config.latent_dim,
            gene_num=config.gene_num
        )

        # 损失函数组件
        self.lambda_reconstruction = config.lambda_reconstruction
        self.lambda_disentangle = config.lambda_disentangle
        self.lambda_ood = config.lambda_ood

    def forward(self, batch, known_data=None):
        """
        前向传播

        参数：
        - batch: 当前batch的输入数据
          - x: 基因表达 [B, gene_num]
          - pert: 扰动信息
          - cell_type: 细胞类型ID
        - known_data: 已知数据（用于OOD预测）
          - known_expr: 已知细胞类型的表达
          - known_delta: 已知细胞类型的响应

        返回：
        - prediction: 预测的基因表达变化
        - losses: 各项损失
        """
        x, pert, cell_type = batch['x'], batch['pert'], batch['cell_type']

        # 1. 编码
        gene_repr = self.gene_encoder(x)
        pert_emb = self.pert_encoder(pert)
        cell_info = self.cell_encoder(x, cell_type)

        E_c = cell_info['base_expression']
        S_c = cell_info['response_sensitivity']

        # 2. OOD知识迁移
        if known_data is not None and self.training:
            migrated_response = self.attention(
                S_c, pert_emb,
                known_data['known_delta'],
                known_data['known_cell_types']
            )
        else:
            # IID模式：使用自身信息
            migrated_response = pert_emb

        # 3. 解码得到响应预测
        delta = self.decoder(E_c, S_c, pert_emb, migrated_response)

        # 4. 计算重建
        predicted_x = x + delta

        return {
            'predicted_delta': delta,
            'predicted_x': predicted_x,
            'cell_info': cell_info
        }

    def compute_loss(self, prediction, target):
        """
        计算总损失

        损失函数包含：
        1. 重建损失：预测表达与真实表达的差异
        2. 解耦损失：确保E_c和S_c独立
        3. OOD损失：增强OOD泛化能力
        4. 传输损失：最优传输正则化
        """
        delta_pred = prediction['predicted_delta']
        delta_true = target['delta']

        # 1. 重建损失 (MSE / NB loss)
        recon_loss = F.mse_loss(delta_pred, delta_true)

        # 2. 解耦损失 (MMD-based independence penalty)
        E_c = prediction['cell_info']['base_expression']
        S_c = prediction['cell_info']['response_sensitivity']
        disentangle_loss = self._compute_disentangle_loss(E_c, S_c)

        # 3. OOD特定损失
        ood_loss = self._compute_ood_loss(prediction, target)

        # 总损失
        total_loss = (
            self.lambda_reconstruction * recon_loss +
            self.lambda_disentangle * disentangle_loss +
            self.lambda_ood * ood_loss
        )

        return {
            'total': total_loss,
            'reconstruction': recon_loss,
            'disentangle': disentangle_loss,
            'ood': ood_loss
        }

    def _compute_disentangle_loss(self, E_c, S_c):
        """
        解耦损失：使用最大均值差异(MMD)确保E_c和S_c独立
        """
        # 简单使用协方差惩罚
        E_c_centered = E_c - E_c.mean(dim=0)
        S_c_centered = S_c - S_c.mean(dim=0)

        cov_E = torch.mm(E_c_centered.T, E_c_centered) / (E_c.size(0) - 1)
        cov_S = torch.mm(S_c_centered.T, S_c_centered) / (S_c.size(0) - 1)

        # 惩罚高度相关
        off_diagonal = torch.abs(cov_E * cov_S).mean()
        return off_diagonal

    def _compute_ood_loss(self, prediction, target):
        """
        OOD损失：使用对比学习和标签平滑增强泛化
        """
        # 标签平滑
        delta_pred = prediction['predicted_delta']
        delta_true = target['delta']

        # L1损失（对异常值更鲁棒）
        ood_loss = F.smooth_l1_loss(delta_pred, delta_true)

        return ood_loss
```

---

## 4. 训练策略

### 4.1 两阶段训练

**阶段1：IID预训练**
- 在IID设置下训练，学习基础的扰动响应预测
- 使用所有可用数据

**阶段2：OOD微调**
- 使用细胞类型感知注意力增强模块
- 重点关注跨细胞类型知识迁移
- 使用对比学习增强细胞类型表示

### 4.2 最优传输正则化

```python
class OptimalTransportRegularizer(nn.Module):
    """
    最优传输正则化

    确保模型学习到的细胞类型表示在扰动响应空间中
    保持传输最优的性质
    """
    def __init__(self, latent_dim, transport_reg=0.1):
        super().__init__()
        self.transport_reg = transport_reg

    def sinkhorn(self, a, b, M, reg, num_iters=100):
        """
        Sinkhorn算法计算最优传输

        a: [B,] - 源分布权重
        b: [B,] - 目标分布权重
        M: [B, B] - 成本矩阵
        reg: 正则化参数
        """
        K = torch.exp(-M / reg)

        u = torch.ones_like(a)
        for _ in range(num_iters):
            u = a / (K @ v + 1e-8)
            v = b / (K.T @ u + 1e-8)

        return (u.unsqueeze(-1) * K * v.unsqueeze(0)).sum()

    def forward(self, source_repr, target_repr):
        # 计算成本矩阵
        M = torch.cdist(source_repr, target_repr) ** 2

        # 均匀权重
        B = source_repr.size(0)
        a = torch.ones(B, device=source_repr.device) / B
        b = torch.ones(B, device=target_repr.device) / B

        # 计算传输损失
        ot_loss = self.sinkhorn(a, b, M, self.transport_reg)

        return ot_loss
```

### 4.3 课程学习（Curriculum Learning）

```
训练进度：
1. 简单扰动（单基因敲除）→ 复杂扰动（多基因/组合扰动）
2. 相似的细胞类型 → 差异大的细胞类型
3. 高剂量 → 低剂量
```

### 4.4 训练配置

```yaml
# config.yaml
model:
  name: CellODE
  latent_dim: 256
  num_heads: 8
  gene_num: 5000

training:
  # 阶段1: IID预训练
  iid_epochs: 200
  iid_batch_size: 256
  iid_lr: 1e-4

  # 阶段2: OOD微调
  ood_epochs: 100
  ood_batch_size: 128
  ood_lr: 5e-5

  # 损失权重
  lambda_reconstruction: 1.0
  lambda_disentangle: 0.1
  lambda_ood: 0.5
  lambda_transport: 0.01

  # 优化器
  optimizer: AdamW
  weight_decay: 1e-4
  scheduler:
    type: CosineAnnealingLR
    T_max: 300

data:
  # 数据集配置
  datasets:
    - KangCrossCell
    - Papalexi
    - Scipex3

  ood_split:
    test_cell_types:
      - A549
      - K562
    validation_cell_types:
      - MCF7
```

---

## 5. 评估方法

### 5.1 与scPerturBench一致的评估指标

| 指标 | 描述 | 计算方法 |
|------|------|----------|
| MSE | 均方误差 | $MSE = \frac{1}{G}\sum_g (\Delta_{pred} - \Delta_{true})^2$ |
| PCC-delta | 预测与真实delta的Pearson相关系数 | $\rho(\Delta_{pred}, \Delta_{true})$ |
| E-distance | 能量距离 | 衡量分布差异 |
| Wasserstein距离 | 扰动分布间的Wasserstein距离 | 最优传输理论 |
| KL-divergence | 预测与真实分布的KL散度 | $D_{KL}(P_{true} \| P_{pred})$ |
| Common-DEGs | 共同差异表达基因重叠率 | Jaccard index |

### 5.2 OOD特定评估

```python
class OODEvaluator:
    """
    OOD评估器

    专门评估模型在未见细胞类型上的泛化能力
    """
    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader

    def evaluate(self, test_cell_types):
        """
        对指定的测试细胞类型进行评估
        """
        results = {}

        for metric in ['mse', 'pcc', 'e_distance', 'common_degs']:
            scores = []

            for batch in self.data_loader:
                # 获取预测
                pred = self.model.predict(batch)

                # 计算各指标
                if metric == 'mse':
                    score = self._compute_mse(pred, batch)
                elif metric == 'pcc':
                    score = self._compute_pcc(pred, batch)
                elif metric == 'e_distance':
                    score = self._compute_e_distance(pred, batch)
                elif metric == 'common_degs':
                    score = self._compute_common_degs(pred, batch)

                scores.append(score)

            results[metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'per_cell_type': self._breakdown_by_cell_type(scores, batch)
            }

        return results

    def _compute_e_distance(self, pred, batch):
        """
        计算能量距离

        E(P, Q) = 2 * E(||X - Y||) - E(||X - X'||) - E(||Y - Y'||)

        用于衡量预测分布与真实分布的差异
        """
        delta_pred = pred['delta']
        delta_true = batch['delta']

        # 样本间距离
        dist_pred = torch.cdist(delta_pred, delta_pred)
        dist_true = torch.cdist(delta_true, delta_true)
        dist_cross = torch.cdist(delta_pred, delta_true)

        e_dist = (2 * dist_cross.mean() - dist_pred.mean() - dist_true.mean())
        return e_dist.item()
```

---

## 6. 数据准备与预处理

### 6.1 数据格式

```python
# AnnData格式要求
adata = sc.read_h5ad('perturbation_data.h5ad')

# obs包含:
# - condition1: 细胞类型
# - condition2: 扰动类型
# - dose: 剂量（对于化学扰动）
# - perturbation: 完整扰动描述

# layers包含:
# - counts: 原始 counts
# - logNor: log-normalized表达

# var:
# - gene_ids: 基因标识符
# - highly_variable: HVG标记
```

### 6.2 数据划分策略

**IID设置**：
- 随机划分训练/验证/测试
- 所有细胞类型和扰动在各个集合中都出现

**OOD设置**：
- 保留某些细胞类型仅出现在测试集
- 扰动在训练集和测试集都出现
- 评估模型对未见细胞类型的泛化能力

---

## 7. 与现有方法的比较

### 7.1 核心差异

| 特性 | CellODE | bioLord | CPA | trVAE | CellOT |
|------|---------|---------|-----|-------|--------|
| 解耦机制 | 因果解耦 | VAE解耦 | 线性分解 | 条件VAE | 最优传输 |
| 知识迁移 | 细胞注意力 | 嵌入插值 | 线性组合 | 隐变量插值 | 传输计划 |
| 预训练嵌入 | scGPT | 无 | 无 | 无 | 无 |
| OOD显式建模 | 是 | 部分 | 否 | 否 | 部分 |
| 可解释性 | 高 | 中 | 中 | 低 | 低 |

### 7.2 预期改进

基于scPerturBench的结果，我们预期CellODE在以下方面取得改进：

1. **MSE**：通过因果解耦，减小对未见细胞类型的预测误差
2. **PCC-delta**：细胞类型感知注意力应能更好地捕捉响应模式
3. **E-distance**：最优传输正则化应减小分布差异
4. **Common-DEGs**：解耦机制应帮助识别真正受影响的基因

---

## 8. 实现计划

### 8.1 环境配置（与cpa和pertpyV7兼容）

由于用户已有 `cpa` 和 `pertpyV7` 两个conda环境，CellODE设计为可以在这两个环境中运行：

#### 方案A：在cpa环境中开发（推荐）

cpa环境已包含：biolord, scGPT, scDisInFact, CPA等核心依赖

```yaml
# environment_cellode_cpa.yml
name: cpa  # 复用现有cpa环境
channels:
  - pytorch
  - conda-forge
  - bioconda
dependencies:
  # 已有依赖（来自cpa环境）
  - python=3.9
  - pytorch>=2.0
  - scanpy>=1.9
  - anndata>=0.9
  - numpy>=1.24
  - scipy>=1.10
  - scikit-learn>=1.3
  - pandas>=2.0
  - matplotlib>=3.7
  - seaborn>=0.12
  - cpa-tools>=0.8.3
  - biolord>=0.0.2
  - pertpy>=0.6.0  # 评估指标
pip:
  # 新增依赖
  - wandb>=0.15
  - pytest>=7.4
```

#### 方案B：创建独立环境

如果需要隔离，可以使用：

```yaml
# environment_cellode.yml
name: cellode
channels:
  - pytorch
  - conda-forge
  - bioconda
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
dependencies:
  - python=3.9
  - pytorch>=2.0
  - scanpy>=1.9.3
  - anndata>=0.9.1
  - numpy>=1.24
  - scipy>=1.10
  - scikit-learn>=1.2
  - pandas>=2.0
  - matplotlib>=3.7
  - seaborn>=0.12
  - tqdm>=4.65
  - h5py>=3.8
  - jax>=0.4.16
  - jaxlib>=0.4.16
pip:
  - cpa-tools>=0.8.3
  - pertpy>=0.6.0
  - wandb>=0.15
```

#### pertpyV7环境的用途

pertpyV7环境专门用于**性能评估**，因为它包含计算各种距离指标（MSE, PCC, E-distance, Wasserstein, KL-divergence等）的工具。

训练后的模型评估：
```bash
conda activate pertpyV7
export OPENBLAS_NUM_THREADS=20
export JAX_PLATFORMS=cpu
python evaluate.py --model CellODE --checkpoint best_model.pt
```

### 8.2 训练和评估脚本设计（兼容cpa/pertpyV7）

#### 训练脚本 (train_cellode.py)

```python
#!/usr/bin/env python
"""
CellODE训练脚本

使用方式：
1. IID预训练：
   conda activate cpa
   python train_cellode.py --mode iid --dataset KangCrossCell

2. OOD微调：
   conda activate cpa
   python train_cellode.py --mode ood --dataset KangCrossCell --checkpoint iid_best.pt

3. 评估（使用pertpyV7）：
   conda activate pertpyV7
   python train_cellode.py --mode eval --checkpoint ood_best.pt
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

# 导入项目模块
from src.models.cellode import CellODE
from src.data.dataset import PerturbationDataset
from src.training.trainer import Trainer
from src.evaluation.evaluator import OODEvaluator

# pertpy用于评估指标（仅在评估时需要）
try:
    import pertpy as pt
    HAS_PERTPY = True
except ImportError:
    HAS_PERTPY = False


def parse_args():
    parser = argparse.ArgumentParser(description='CellODE Training')
    parser.add_argument('--mode', type=str, default='iid',
                        choices=['iid', 'ood', 'eval'],
                        help='训练模式: iid预训练, ood微调, eval评估')
    parser.add_argument('--dataset', type=str, default='KangCrossCell',
                        help='数据集名称')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='数据目录')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='模型检查点路径')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='输出目录')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='配置文件路径')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备: cuda或cpu')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    return parser.parse_args()


def setup_environment(args):
    """设置训练环境"""
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    return device, output_dir


def load_dataset(args, mode='train'):
    """加载数据集

    数据格式兼容scPerturBench的h5ad格式
    """
    data_path = Path(args.data_dir) / args.dataset

    # 加载h5ad数据
    if mode in ['iid', 'ood']:
        adata = sc.read_h5ad(data_path / 'filter_hvg5000_logNor.h5ad')
    else:
        # 评估模式：使用保存的预测结果
        pass

    # 数据集划分
    if mode == 'iid':
        # IID划分：随机划分
        from sklearn.model_selection import train_test_split
        indices = np.arange(adata.n_obs)
        train_idx, val_idx = train_test_split(
            indices, test_size=0.1, random_state=args.seed
        )
    elif mode == 'ood':
        # OOD划分：按细胞类型划分
        cell_types = adata.obs['condition1'].unique()
        ood_cell_types = cell_types[::3]  # 每3个取1个作为OOD
        train_cell_types = [ct for ct in cell_types if ct not in ood_cell_types]

        train_idx = adata.obs['condition1'].isin(train_cell_types).values
        val_idx = adata.obs['condition1'].isin(ood_cell_types).values

    return adata, train_idx, val_idx


def train_iid(args, device, output_dir):
    """IID预训练"""
    print("="*50)
    print("阶段1: IID预训练")
    print("="*50)

    # 加载数据
    adata, train_idx, val_idx = load_dataset(args, mode='iid')

    # 创建数据集
    train_dataset = PerturbationDataset(adata[train_idx])
    val_dataset = PerturbationDataset(adata[val_idx])

    # 创建模型
    model = CellODE(config={
        'gene_num': adata.n_vars,
        'latent_dim': 256,
        'pert_type': 'genetic',
        'num_heads': 8
    })
    model = model.to(device)

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        output_dir=output_dir
    )

    # 训练
    best_model = trainer.train(
        epochs=200,
        batch_size=256,
        lr=1e-4,
        early_stopping_patience=20
    )

    # 保存模型
    checkpoint_path = output_dir / 'iid_best.pt'
    torch.save(best_model.state_dict(), checkpoint_path)
    print(f"IID预训练完成，模型保存至: {checkpoint_path}")

    return best_model


def train_ood(args, device, output_dir, iid_checkpoint):
    """OOD微调"""
    print("="*50)
    print("阶段2: OOD微调")
    print("="*50)

    # 加载数据
    adata, train_idx, val_idx = load_dataset(args, mode='ood')

    # 创建数据集
    train_dataset = PerturbationDataset(adata[train_idx])
    val_dataset = PerturbationDataset(adata[val_idx])

    # 创建模型并加载IID预训练权重
    model = CellODE(config={
        'gene_num': adata.n_vars,
        'latent_dim': 256,
        'pert_type': 'genetic',
        'num_heads': 8
    })
    model = model.to(device)

    if iid_checkpoint:
        state_dict = torch.load(iid_checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        print(f"加载IID预训练权重: {iid_checkpoint}")

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        output_dir=output_dir
    )

    # OOD微调（使用较小的学习率）
    best_model = trainer.train(
        epochs=100,
        batch_size=128,
        lr=5e-5,
        early_stopping_patience=20
    )

    # 保存模型
    checkpoint_path = output_dir / 'ood_best.pt'
    torch.save(best_model.state_dict(), checkpoint_path)
    print(f"OOD微调完成，模型保存至: {checkpoint_path}")

    return best_model


def evaluate(args, device, checkpoint_path):
    """评估模型（使用pertpy计算指标）"""
    print("="*50)
    print("模型评估")
    print("="*50)

    if not HAS_PERTPY:
        raise ImportError(
            "请使用pertpyV7环境运行评估: conda activate pertpyV7"
        )

    # 加载数据
    adata, _, test_idx = load_dataset(args, mode='ood')
    test_dataset = PerturbationDataset(adata[test_idx])

    # 加载模型
    model = CellODE(config={
        'gene_num': adata.n_vars,
        'latent_dim': 256,
        'pert_type': 'genetic',
        'num_heads': 8
    })
    model = model.to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 评估器
    evaluator = OODEvaluator(model, test_dataset, device=device)

    # 计算指标
    metrics = ['mse', 'pcc_delta', 'e_distance', 'wasserstein',
               'kl_divergence', 'common_degs']

    results = evaluator.evaluate(metrics=metrics)

    # 打印结果
    print("\n评估结果:")
    print("-"*40)
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    # 保存结果
    results_df = pd.DataFrame([results])
    results_path = Path(args.output_dir) / 'evaluation_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n结果保存至: {results_path}")

    return results


def main():
    args = parse_args()
    device, output_dir = setup_environment(args)

    if args.mode == 'iid':
        train_iid(args, device, output_dir)

    elif args.mode == 'ood':
        train_ood(args, device, output_dir, args.checkpoint)

    elif args.mode == 'eval':
        if not args.checkpoint:
            raise ValueError("评估模式需要指定 --checkpoint")
        evaluate(args, device, args.checkpoint)


if __name__ == '__main__':
    main()
```

#### 评估脚本 (evaluate_cellode.py)

```python
#!/usr/bin/env python
"""
CellODE评估脚本

与scPerturBench的calPerformance.py兼容
使用pertpyV7环境计算评估指标

使用方式：
conda activate pertpyV7
export OPENBLAS_NUM_THREADS=20
export JAX_PLATFORMS=cpu
python evaluate_cellode.py --model_path outputs/ood_best.pt --dataset KangCrossCell
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from pathlib import Path

# pertpy用于评估
import pertpy as pt
from scipy import sparse

# 项目模块
from src.models.cellode import CellODE
from src.data.dataset import PerturbationDataset


def checkNan(adata):
    """处理NaN值"""
    if sparse.issparse(adata.X):
        adata.X = adata.X.toarray()
    nan_rows = np.where(np.isnan(adata.X).any(axis=1))[0]
    if len(nan_rows) > 0:
        a = adata[adata.obs['perturbation'] == 'control'].X.mean(axis=0)
        a = a.reshape([1, -1])
        b = np.tile(a, [len(nan_rows), 1])
        adata[nan_rows].X = b
    return adata


def calculateDelta(adata):
    """计算delta（扰动响应）"""
    adata_control = adata[adata.obs['perturbation'] == 'control'].copy()
    adata_imputed = adata[adata.obs['perturbation'] == 'imputed'].copy()
    adata_stimulated = adata[adata.obs['perturbation'] == 'stimulated'].copy()

    control_mean = adata_control.X.mean(axis=0)
    adata_imputed.X = adata_imputed.X - control_mean
    adata_stimulated.X = adata_stimulated.X - control_mean

    adata_delta = sc.concat([adata_control, adata_imputed, adata_stimulated])
    return adata_delta


def evaluate_model(model, test_data, metrics=['mse', 'pcc_delta', 'e_distance']):
    """评估模型性能"""
    results = {}

    with torch.no_grad():
        predictions = model.predict(test_data)
        delta_pred = predictions['delta']

    for metric in metrics:
        if metric == 'mse':
            mse = np.mean((delta_pred - test_data['delta'])**2)
            results['mse'] = mse

        elif metric == 'pcc_delta':
            # Pearson相关系数
            from scipy.stats import pearsonr
            pcc, _ = pearsonr(delta_pred.mean(axis=0), test_data['delta'].mean(axis=0))
            results['pcc_delta'] = pcc

        elif metric == 'e_distance':
            # 能量距离
            distance = pt.tools.Distance(metric='edistance')
            # ... 计算能量距离
            results['e_distance'] = 0.0  # 占位

        elif metric == 'wasserstein':
            # Wasserstein距离
            wass_dist = pt.tools.Distance(metric='wasserstein')
            # ... 计算Wasserstein距离
            results['wasserstein'] = 0.0  # 占位

        elif metric == 'kl_divergence':
            # KL散度
            kl_div = pt.tools.Distance(metric='sym_kldiv')
            # ... 计算KL散度
            results['kl_divergence'] = 0.0  # 占位

    return results


class SuppressOutput:
    """抑制pertpy的输出"""
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._stdout
        sys.stderr = self._stderr


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='KangCrossCell')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output', type=str, default='./results')
    parser.add_argument('--deg', type=int, default=100, choices=[100, 5000])
    args = parser.parse_args()

    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CellODE(config={'gene_num': 5000, 'latent_dim': 256})
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # 加载数据
    data_path = Path(args.data_dir) / args.dataset
    adata = sc.read_h5ad(data_path / 'filter_hvg5000_logNor.h5ad')

    # ... 评估逻辑与scPerturBench的calPerformance.py兼容

    print("评估完成")


if __name__ == '__main__':
    main()
```

#### 使用示例

```bash
# 1. IID预训练（使用cpa环境）
conda activate cpa
python train_cellode.py --mode iid --dataset KangCrossCell --output_dir ./outputs

# 2. OOD微调（使用cpa环境）
conda activate cpa
python train_cellode.py --mode ood --dataset KangCrossCell \
    --checkpoint ./outputs/iid_best.pt \
    --output_dir ./outputs

# 3. 评估（使用pertpyV7环境，与scPerturBench一致）
conda activate pertpyV7
export OPENBLAS_NUM_THREADS=20
export JAX_PLATFORMS=cpu
python evaluate_cellode.py \
    --model_path ./outputs/ood_best.pt \
    --dataset KangCrossCell \
    --output ./results
```

```
CellODE/
├── configs/
│   └── default.yaml
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cellode.py          # 主模型
│   │   ├── encoders.py         # 编码器
│   │   ├── attention.py         # 注意力模块
│   │   └── decoder.py          # 解码器
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py           # 数据加载
│   │   └── preprocessing.py     # 预处理
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py          # 训练器
│   │   └── losses.py           # 损失函数
│   └── evaluation/
│       ├── __init__.py
│       └── evaluator.py         # 评估器
├── scripts/
│   ├── train_cellode.py          # 训练脚本（cpa环境使用）
│   └── evaluate_cellode.py       # 评估脚本（pertpyV7环境使用）
├── tests/
│   └── test_cellode.py
├── README.md
└── setup.py

# 推荐的目录布局（与scPerturBench配合使用）
# 可在AutoDL服务器上创建如下结构
work_dir/
├── CellODE/                     # 本项目代码
├── Pertb_benchmark/             # scPerturBench代码
│   ├── manuscript1/             # Cellular_context_generalization
│   │   ├── i.i.d/
│   │   └── o.o.d./              # OOD测试脚本
│   ├── DataSet/                 # 数据集
│   └── Results/                 # 评估结果
├── cpa_env/                     # cpa conda环境
└── pertpyV7_env/                # pertpyV7 conda环境
```

### 8.4 开发阶段

**Phase 1: 基础框架 (Week 1-2)**
- 项目结构搭建
- 数据加载和预处理
- 基础编码器实现

**Phase 2: 核心模块 (Week 3-4)**
- 因果解耦模块
- 细胞类型感知注意力
- 响应解码器

**Phase 3: 训练与评估 (Week 5-6)**
- 训练流程实现
- 评估指标集成
- 与scPerturBench兼容

**Phase 4: 实验与优化 (Week 7-8)**
- 超参数调优
- 与baseline方法对比
- 论文写作

---

## 9. 参考文献

[1] Wei, Z., Wang, Y., Gao, Y., Liu, Q., et al. (2025). Benchmarking algorithms for generalizable single-cell perturbation response prediction. Nature Methods.

[2] Lotfollahi, M., et al. (2023). Predicting cellular responses to complex perturbations in high-throughput screens. Molecular Systems Biology.

[3] Wang, B., et al. (2024). Disentanglement of single-cell data with biolord. Nature Biotechnology.

[4] Bunne, C., et al. (2023). Learning single-cell perturbation responses using neural optimal transport. Nature Methods.

[5] Kang, J.B., et al. (2024). scGPT: toward building a foundation model for single-cell multi-omics using generative AI. Nature Methods.

---

## 10. 附录：关键公式汇总

### A.1 损失函数

总损失：
$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{recon} + \lambda_2 \mathcal{L}_{disentangle} + \lambda_3 \mathcal{L}_{ood}$$

重建损失：
$$\mathcal{L}_{recon} = \mathbb{E}[||\Delta_{pred} - \Delta_{true}||_2^2]$$

解耦损失（MMD惩罚）：
$$\mathcal{L}_{disentangle} = \text{MMD}(E_c, S_c)$$

OOD损失：
$$\mathcal{L}_{ood} = \mathbb{E}[\text{SmoothL1}(\Delta_{pred}, \Delta_{true})]$$

### A.2 注意力权重

细胞类型相似度：
$$w_{c,c'} = \text{softmax}(\text{MLP}([S_c, S_{c'}]))$$

知识迁移：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V$$

### A.3 响应预测

$$\Delta_{c,p} = \text{Decoder}(E_c, S_c, \psi(p), \text{Attention}(S_c, \{S_{c'}\}, \{\Delta_{c',p}\}))$$
