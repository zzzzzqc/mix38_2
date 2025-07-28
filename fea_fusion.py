import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F
from atten import SpatialAttention, ChannelAttention


class PSPModule(nn.Module):
    def __init__(self, sizes=(1, 3, 6, 8), dimension=2):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats):
        n, c, _, _ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        center = torch.cat(priors, -1)
        return center


class CrossAttention_psp(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super(CrossAttention_psp, self).__init__()

        self.key_embed = in_channels // ratio

        self.q_x_s = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1),
            nn.BatchNorm2d(in_channels // ratio),
            nn.ReLU(inplace=True)
        )
        self.q_x_f = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1),
            nn.BatchNorm2d(in_channels // ratio),
            nn.ReLU(inplace=True)
        )

        self.k_x_s = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1),
            nn.BatchNorm2d(in_channels // ratio),
            nn.ReLU(inplace=True)
        )
        self.k_x_f = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1),
            nn.BatchNorm2d(in_channels // ratio),
            nn.ReLU(inplace=True)
        )

        self.v = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels // ratio, kernel_size=1),
            nn.BatchNorm2d(in_channels // ratio),
            nn.ReLU(inplace=True)
        )

        self.Conv_convert = nn.Conv2d(in_channels // ratio, in_channels, kernel_size=1, stride=1)
        self.psp = PSPModule(sizes=(1, 3, 6, 8))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_s, x_f):
        B, C, H, W = x_s.shape

        query_x_s = self.q_x_s(x_s).view(B, self.key_embed, -1).permute(0, 2, 1)
        query_x_f = self.q_x_f(x_f).view(B, self.key_embed, -1).permute(0, 2, 1)

        key_x_s = self.psp(self.k_x_s(x_s))
        key_x_f = self.psp(self.k_x_f(x_f))

        value = self.psp(self.v(torch.cat([x_s, x_f], dim=1))).permute(0, 2, 1)

        score1 = torch.matmul(query_x_s, key_x_f)
        score1 = (self.key_embed ** -0.5) * score1

        score2 = torch.matmul(query_x_f, key_x_s)
        score2 = (self.key_embed ** -0.5) * score2

        score = F.softmax(score1 + score2, dim=-1)  # /2?

        context = torch.matmul(score, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(B, self.key_embed, H, W)

        context = self.Conv_convert(context)

        context = context + x_f + x_s

        return self.relu(context)


class EfficientAttention(nn.Module):
    """
    input  -> x:[B, D, H, W]
    output ->   [B, D, H, W]
    in_channels:    int -> Embedding Dimension
    key_channels:   int -> Key Embedding Dimension,   Best: (in_channels)
    value_channels: int -> Value Embedding Dimension, Best: (in_channels or in_channels//2)
    head_count:     int -> It divides the embedding dimension by the head_count and process each part individually
    Conv2D # of Params:  ((k_h * k_w * C_in) + 1) * C_out)
    """

    def __init__(self, in_channels, key_channels, value_channels, head_count=1):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w = input_.size()

        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))

        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=2)

            query = F.softmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=1)

            value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]

            context = key @ value.transpose(1, 2)  # dk*dv
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)  # n*dv
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values)

        return attention


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super(AttentionPool2d, self).__init__()
        # --------------------------------------------------#
        #  nn.Parameter()的作用为作为nn.Module中的可训练参数使用
        # --------------------------------------------------#
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        # -------------------------------#
        # 通过全连接层来获取以下四个映射量
        # -------------------------------#
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        # ---------------------------------------------------------------#
        # 首先进行张量shape的转变,由 batch_size,c,h,w -> (h*w),batch_size,c
        # ---------------------------------------------------------------#
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)
        # -------------------------------------------#
        # (h*w),batch_size,c -> (h*w+1),batch_size,c
        # -------------------------------------------#
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        # --------------------------------------------------------------------#
        # tensor的shape以及type均不发生改变,所做的只是将位置信息嵌入至原先的tensor中
        # shape:(h*w+1),batch_size,c
        # --------------------------------------------------------------------#
        x = x + self.positional_embedding[:, None, :].to(x.dtype)
        # ---------------------------------------#
        # 将输入的张量pass through 多头注意力机制模块
        # ---------------------------------------#
        x, _ = torch.nn.functional.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )


class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.projection = nn.Linear(dim, dim)

    # hw不同,e相同
    def forward(self, x, x2):
        # shape: b, hw, e,
        value = x
        key = F.softmax(x2.transpose(1, 2), dim=1)
        query = F.softmax(x2, dim=1)

        context = value @ key
        atten = context @ query

        atten = self.projection(atten)

        return atten


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        tx = x.transpose(1, 2).view(B, C, H, W)
        conv_x = self.dwconv(tx)
        return conv_x.flatten(2).transpose(1, 2)


class MixFFN_skip(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.norm1(self.dwconv(self.fc1(x), H, W) + self.fc1(x)))
        out = self.fc2(ax)
        return out


class Cross_Layer_Fusion(nn.Module):
    def __init__(self, dims, size):
        super().__init__()
        self.dims = dims
        self.size = size

        # todo 中间层经过全连接层作为query,低层作为key,velue,middle_linear_shallow
        self.attention_mls = CrossAttention(dims[0])
        # todo 中间层经过全连接层作为query,高层作为key,velue,middle_linear_deep
        self.attention_mld = CrossAttention(dims[2])
        # todo 中间层作为query,低层经过全连接层作为key,velue,middle_shallow_linear
        self.attention_msl = CrossAttention(dims[1])
        # todo 中间层作为query,高层经过全连接层作为key,velue,middle_shallow_deep
        self.attention_mdl = CrossAttention(dims[1])

        # todo 中间层与低层相互转换的全连接层
        # todo shallow--->middle
        self.linear_s2m = nn.Linear(dims[0], dims[1], bias=False)
        # todo middle--->shallow
        self.linear_m2s = nn.Linear(dims[1], dims[0], bias=False)

        # todo 中间层与高层相互转换的全连接层
        # todo deep--->middle
        self.linear_d2m = nn.Linear(dims[2], dims[1], bias=False)
        # todo midele--->deep
        self.linear_m2d = nn.Linear(dims[1], dims[2], bias=False)

        # todo 中间层经过全连接层作为query，低层作为key,velue，交叉注意力，经过全连接层输出
        self.linear_out1 = nn.Linear(dims[0], dims[1], bias=False)
        # todo 中间层经过全连接层作为query，高层作为key,velue，交叉注意力，经过全连接层输出
        self.linear_out2 = nn.Linear(dims[2], dims[1], bias=False)

        self.norm = nn.LayerNorm(dims[1])
        self.mlp = MixFFN_skip(dims[1], dims[1] * 4)

    def forward(self, x_s, x_m, x_d):
        B_s, C_s, H_s, W_s = x_s.shape
        B_m, C_m, H_m, W_m = x_m.shape
        B_d, C_d, H_d, W_d = x_d.shape
        x_s = x_s.reshape(B_s, C_s, -1).permute(0, 2, 1)
        x_m = x_m.reshape(B_m, C_m, -1).permute(0, 2, 1)
        x_d = x_d.reshape(B_d, C_d, -1).permute(0, 2, 1)

        x_s2m = self.linear_s2m(x_s)
        x_m2s = self.linear_m2s(x_m)
        x_d2m = self.linear_d2m(x_d)
        x_m2d = self.linear_m2d(x_m)

        atten1 = self.attention_mls(x_m2s, x_s)
        atten2 = self.attention_msl(x_m, x_s2m)
        atten3 = self.attention_mdl(x_m, x_d2m)
        atten4 = self.attention_mld(x_m2d, x_d)

        atten14 = self.linear_out1(atten1) + self.linear_out2(atten4)

        hyper_atten14 = atten14.sigmoid()
        other = (1 - hyper_atten14) / 2
        atten_final = hyper_atten14.expand_as(atten14) * atten14 + other.expand_as(atten2) * atten2 + other.expand_as(
            atten3) * atten3

        atten = x_m + atten_final
        atten = Rearrange('b (h w) c -> b c h w', h=H_m, w=W_m)(atten)
        # atten_norm = self.norm(atten)
        # print(atten.shape, atten_norm.shape)
        # atten = atten + self.mlp(atten_norm, *self.size)
        return atten


class Cross_Encoder_Fusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.spa_atten = SpatialAttention(in_channels)
        # self.spa_atten = ChannelAttention(in_channels)
        self.cha_atten = ChannelAttention(in_channels)
        # self.cha_atten = SpatialAttention(in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, (1, 1), 1)

    def forward(self, x_spa, x_freq):
        x_spa_1 = self.conv(x_spa)
        x_freq_1 = self.conv(x_freq)
        x_spa_1_sa = self.spa_atten(x_spa_1)
        x_freq_1_ca = self.cha_atten(x_freq_1)
        x_spa_2 = x_spa_1 + x_spa_1 * x_freq_1_ca
        x_freq_2 = x_freq_1 + x_freq_1 * x_spa_1_sa

        out = x_spa_2 + x_freq_2

        return out


# class Cross_Encoder_Fusion_v2(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.cha_atten = SpatialAttention(in_channels)
#         self.spa_atten = ChannelAttention(in_channels)
#         self.conv = nn.Conv2d(in_channels, in_channels, (1, 1), 1)
#
#     def forward(self, x_spa, x_freq):
#         x_spa_1 = self.conv(x_spa)
#         x_freq_1 = self.conv(x_freq)
#         x_spa_1_sa = self.spa_atten(x_spa_1)
#         x_freq_1_ca = self.cha_atten(x_freq_1)
#         x_spa_2 = x_spa_1 + x_spa_1 * x_freq_1_ca
#         x_freq_2 = x_freq_1 + x_freq_1 * x_spa_1_sa
#         out = x_spa_2 + x_freq_2
#         return out


class CrossAttention_v1(nn.Module):
    def __init__(self, in_channels, num_heads=8, dim_head=64):
        """
        交叉注意力模块
        Args:
            in_channels: 输入通道数
            num_heads: 注意力头数
            dim_head: 每个注意力头的维度
        """
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        inner_dim = num_heads * dim_head
        self.scale = dim_head ** -0.5  # 缩放因子

        # 定义Q、K、V的投影层
        self.to_q = nn.Conv2d(in_channels, inner_dim, 1, bias=False)
        self.to_k = nn.Conv2d(in_channels, inner_dim, 1, bias=False)
        self.to_v = nn.Conv2d(in_channels, inner_dim, 1, bias=False)

        # 输出投影层
        self.to_out = nn.Conv2d(inner_dim, in_channels, 1)

    def forward(self, x1, x2):
        """
        前向传播
        Args:
            x1: 查询特征图 (n, c, h, w)
            x2: 键值特征图 (n, c, h, w)
        Returns:
            融合后的特征图 (n, c, h, w)
        """
        batch, _, h, w = x1.shape

        # 生成Q、K、V
        q = self.to_q(x1)  # (n, inner_dim, h, w)
        k = self.to_k(x2)
        v = self.to_v(x2)

        # 多头处理
        q = self._reshape_to_heads(q)  # (n, heads, h*w, dim_head)
        k = self._reshape_to_heads(k)
        v = self._reshape_to_heads(v)

        # 计算注意力权重
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # 注意力加权
        out = torch.matmul(attn, v)  # (n, heads, h*w, dim_head)

        # 合并多头并恢复形状
        out = self._reshape_from_heads(out, h, w)

        # 输出投影和残差连接
        return self.to_out(out) + x1

    def _reshape_to_heads(self, x):
        batch, inner_dim, h, w = x.shape
        x = x.view(batch, self.num_heads, self.dim_head, h * w)
        return x.permute(0, 1, 3, 2)  # (n, heads, h*w, dim_head)

    def _reshape_from_heads(self, x, h, w):
        batch, heads, _ = x.shape[:3]
        x = x.permute(0, 1, 3, 2).reshape(batch, -1, h, w)
        return x


class Cross_Atten_Encoder_Fusion(nn.Module):
    def __init__(self, channels, num_heads=8, dim_head=64):
        """
        双向交叉注意力融合模块
        Args:
            channels: 输入通道数
            num_heads: 注意力头数
            dim_head: 每个头的维度
        """
        super().__init__()
        # 双向交叉注意力
        self.cross_attn_12 = CrossAttention_v1(channels, num_heads, dim_head)
        self.cross_attn_21 = CrossAttention_v1(channels, num_heads, dim_head)

        # 融合卷积
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        # 双向注意力
        out1 = self.cross_attn_12(x1, x2)  # x1作为query
        out2 = self.cross_attn_21(x2, x1)  # x2作为query

        # 特征融合
        fused = torch.cat([out1, out2], dim=1)
        return self.fusion(fused)


if __name__ == '__main__':
    print('##############')
    x1 = torch.randn(1, 128, 128, 128)
    x2 = torch.randn(1, 256, 64, 64)
    x3 = torch.randn(1, 512, 32, 32)
    dims = [128, 256, 512]
    size = 196
    model = Cross_Layer_Fusion(dims, size)
    out = model(x1, x2, x3)
    print('Cross_Layer_Fusion: ', out.shape)

    x4 = torch.randn(1, 64, 4, 4)
    x5 = torch.randn(1, 64, 4, 4)
    model_cef = Cross_Encoder_Fusion(64)
    out = model_cef(x4, x5)
    print('Cross_Encoder_Fusion', out.shape)

    x1 = torch.randn(2, 256, 32, 32)
    x2 = torch.randn(2, 256, 32, 32)

    # 初始化融合模块
    fusion = Cross_Atten_Encoder_Fusion(channels=256)

    # 前向传播
    output = fusion(x1, x2)

    print(f"输入形状: {x1.shape}, {x2.shape}")
    print(f"输出形状: {output.shape}")  # 应该保持 (2, 256, 32, 32)
