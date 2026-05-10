# 已有的LLM再训练(规范框架，学习理论)

</div>

---

## 训练配置

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.126.09             Driver Version: 580.126.09     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090        On  |   00000000:39:00.0 Off |                  Off |
| 61%   25C    P8             28W /  450W |       1MiB /  49140MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

训练时长：30min


---

## 改进

| 描述 | 结果 |
|------|------|
| 数据扩大 | 语料库扩大到GB级别-mini版本-数据集质量高 |
| 模型架构 |  |
| 训练策略 |  |
|  |  |

</div>

---

## 效果示例

```
用户      ❯ 

模型      ❯

```

```
用户      ❯ 

模型      ❯ 

```

```
用户      ❯ 

模型      ❯

```

---

## 结构特征

**模型结构 - Transformer Decoder（类GPT）：**
- 输入: token序列（如"你好"）
- ↓
- Token Embedding (wte): 将token ID映射为768维向量
- Position Embedding (wpe): 添加位置编码（可学习，最大512位置）
- ↓
- Dropout: 0.1丢弃率防止过拟合
- ↓
- Transformer层 × 12 (config.n_layer=12):
- LayerNorm → CausalSelfAttention (12头，每头64维)
- 残差连接
- LayerNorm → MLP (4倍扩展: 768 → 3072 → 768)
- 残差连接
- ↓
- 最终LayerNorm (ln_f)
- ↓
- LM Head (lm_head): 线性层，768 → 词表大小(21128)
- ↓
- 输出: 词汇表概率分布

## 关键特性：

- 权重共享: wte和lm_head共享权重（tie weights）
- Flash Attention: 自动使用PyTorch 2.0的Flash Attention加速
- 因果掩码: 通过is_causal=True实现自回归
- 参数初始化:
- 线性层: N(0, 0.02)
- 残差投影层: N(0, 0.02/√(2×n_layer))

## 训练配置：

- 参数量: ~124M (124,000,000参数)
- 优化器: AdamW (lr=3e-4, weight_decay=0.01)
- 学习率调度: Warmup (10% steps) + Cosine退火
- 梯度裁剪: max_norm=1.0
- Batch size: 4 (训练) / 4 (验证)
- Block size: 512 tokens
- Epochs: 20
- 损失函数: 交叉熵（ignore_index=-1忽略padding）

## 生成配置：

- 采样策略: Top-k (k=40) + Temperature (0.8)
- 重复惩罚: repetition_penalty=1.15
- 停止条件: max_new_tokens 或 EOS token

---

## 改进方向

**扩大数据集**（最主要）：
- 采用公开真实高质量数据集
- 尝试至少更大的MB级别数据

**数据处理改进**
- 简单的token拼接，没有句子边界意识
- 验证集和训练集使用相同处理方式

**模型架构改进**：
- 124M参数但对于中文小说可能不
- 没有使用更先进的归一化方法
- 位置编码使用可学习而非RoPE/RoFormer

**训练策略改进**：
- Batch size太小（4），梯度不稳定
- 没有使用混合精度训练
- 验证集处理方式与训练集相同
- 没有early stopping

**生成质量改进**：
- 重复惩罚可能导致语义断裂
- 没有使用beam search
- Temperature固定，没有自适应
