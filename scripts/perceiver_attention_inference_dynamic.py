import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def reshape_tensor(t, heads):
    b, n, dim = t.shape
    head_dim = dim // heads
    # 将张量从形状 (b, n, dim) reshape 成 (b, heads, n, head_dim)
    return t.view(b, n, heads, head_dim).permute(0, 2, 1, 3)

class PerceiverAttentionCA(nn.Module):
    def __init__(self, *, dim=3072, dim_head=128, heads=16, kv_dim=2048, gating=True):
        """
        PerceiverAttentionCA 推理模块，采用动态 gating 机制来平衡 id 特征和 latent 特征。

        参数:
            dim (int): 总特征维度。
            dim_head (int): 每个注意力头的特征维度。
            heads (int): 注意力头数。
            kv_dim (int): 用于 key/value 投影的输入维度。
            gating (bool): 是否使用 gating 机制。
        """
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        self.gating = gating
        
        inner_dim = dim_head * heads
        # 分别为 id 特征和 latent 特征设置 LayerNorm。
        self.norm1 = nn.LayerNorm(dim if kv_dim is None else kv_dim)  # 针对 id 特征
        self.norm2 = nn.LayerNorm(dim)  # 针对 latent 特征

        # 使用 latent 特征生成 query; 使用 id 特征生成 key 和 value。
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim if kv_dim is None else kv_dim, inner_dim * 2, bias=False)
        
        # 动态 gating 模块：从 latent 特征中计算一个 gating 系数（范围在 0 到 1）
        if gating:
            self.gate_net = nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.ReLU(),
                nn.Linear(dim // 2, 1),
                nn.Sigmoid()
            )
        
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        前向推理过程，不包含任何训练相关操作。
        
        参数:
            x (torch.Tensor): id 特征，形状 (b, n1, D)。
            latents (torch.Tensor): latent 图像特征，形状 (b, n2, D)。
        
        返回:
            torch.Tensor: 融合交叉注意力后的输出张量。
        """
        # 分别对 x 和 latents 应用 LayerNorm
        x = self.norm1(x)             # 例如形状: (b, n1, D)
        latents = self.norm2(latents) # 例如形状: (b, n2, D)
        b, seq_len, _ = latents.shape

        # 从 latent 特征计算 query
        q = self.to_q(latents)         # 形状: (b, n2, inner_dim)
        # 从 id 特征计算 key 和 value
        kv = self.to_kv(x)             # 形状: (b, n1, inner_dim * 2)
        k, v = kv.chunk(2, dim=-1)     # 分成两部分，每部分形状: (b, n1, inner_dim)

        # 重塑张量以适应多头注意力计算
        q = reshape_tensor(q, self.heads)  # 形状: (b, heads, n2, head_dim)
        k = reshape_tensor(k, self.heads)  # 形状: (b, heads, n1, head_dim)
        v = reshape_tensor(v, self.heads)  # 形状: (b, heads, n1, head_dim)

        # 应用缩放因子以稳定计算
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # 形状: (b, heads, n2, n1)
        weight = F.softmax(weight.float(), dim=-1).type(weight.dtype)
        
        out = weight @ v  # 形状: (b, heads, n2, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(b, seq_len, -1)  # 形状: (b, n2, inner_dim)

        # 动态计算 gating 系数：这里采用对 latent 特征均值的方式计算一个标量 gating 系数
        if self.gating:
            # 这里计算的是每个样本的全局 latent 特征平均值，然后得到一个 gating 系数
            latent_mean = latents.mean(dim=1)  # 形状: (b, D)
            gate_val = self.gate_net(latent_mean)  # 形状: (b, 1)，取值范围 [0, 1]
            # 根据 gate_val 对 attention 输出进行加权
            # 你可以根据具体需求设计更复杂的融合方式，比如同时对 id 分支也进行处理后再组合
            out = gate_val.unsqueeze(1) * out

        return self.to_out(out)

if __name__ == "__main__":
    # 推理示例：生成随机张量模拟 id 特征和 latent 特征
    batch_size = 4
    id_tokens = 32
    latent_tokens = 2304
    D = 3072
    x = torch.randn(batch_size, id_tokens, D)
    latents = torch.randn(batch_size, latent_tokens, D)
    
    # 初始化模型进行推理
    model = PerceiverAttentionCA(dim=D, dim_head=128, heads=16, kv_dim=2048, gating=True)
    output = model(x, latents)
    print("Output shape:", output.shape)