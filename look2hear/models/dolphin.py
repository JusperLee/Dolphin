"""
Dolphin Model

This implementation is inspired by and borrows concepts from Sepformer.
The original Sepformer work is licensed under the Apache-2.0 License.

References:
- SepReformer: https://github.com/dmlguq456/SepReformer
- Apache-2.0 License: https://www.apache.org/licenses/LICENSE-2.0

"""

from re import S
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from vector_quantize_pytorch import ResidualVQ
from .video_compoent import * 
from huggingface_hub import PyTorchModelHubMixin

class LayerScale(torch.nn.Module):
    def __init__(self, dims, input_size, Layer_scale_init=1.0e-5):
        super().__init__()
        if dims == 1:
            self.layer_scale = torch.nn.Parameter(torch.ones(input_size)*Layer_scale_init, requires_grad=True)
        elif dims == 2:
            self.layer_scale = torch.nn.Parameter(torch.ones(1,input_size)*Layer_scale_init, requires_grad=True)
        elif dims == 3:
            self.layer_scale = torch.nn.Parameter(torch.ones(1,1,input_size)*Layer_scale_init, requires_grad=True)
    
    def forward(self, x):
        return x*self.layer_scale

class Masking(torch.nn.Module):
    def __init__(self, input_dim):
        super(Masking, self).__init__()
        self.gate_act = torch.nn.ReLU()
            
    def forward(self, x, skip):
        return self.gate_act(x) * skip


class FFN(torch.nn.Module):
    def __init__(self, in_channels, dropout_rate, Layer_scale_init=1.0e-5):
        super().__init__()
        expand_factor = 3
        self.net1 = torch.nn.Sequential(
            torch.nn.LayerNorm(in_channels),
            torch.nn.Linear(in_channels, in_channels * expand_factor))
        self.depthwise = torch.nn.Conv1d(in_channels * expand_factor, in_channels * expand_factor, 3, padding=1, groups=in_channels * expand_factor)
        self.net2 = torch.nn.Sequential(
            torch.nn.GLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(in_channels * expand_factor // 2, in_channels),
            torch.nn.Dropout(dropout_rate))
        self.Layer_scale = LayerScale(dims=3, input_size=in_channels, Layer_scale_init=Layer_scale_init)

    def forward(self, x):
        y = self.net1(x)
        y = y.permute(0, 2, 1).contiguous()
        y = self.depthwise(y)
        y = y.permute(0, 2, 1).contiguous()
        y = self.net2(y)
        return x + self.Layer_scale(y)


class MultiHeadAttention(torch.nn.Module):
    """
    Multi-Head Attention layer.
        :param int n_head: the number of head s
        :param int n_feat: the number of features
        :param float dropout_rate: dropout rate
    """
    def __init__(self, n_head: int, in_channels: int, dropout_rate: float, Layer_scale_init=1.0e-5):
        super().__init__()
        assert in_channels % n_head == 0
        self.d_k = in_channels // n_head # We assume d_v always equals d_k
        self.h = n_head
        self.layer_norm = torch.nn.LayerNorm(in_channels)
        self.linear_q = torch.nn.Linear(in_channels, in_channels)
        self.linear_k = torch.nn.Linear(in_channels, in_channels)
        self.linear_v = torch.nn.Linear(in_channels, in_channels)
        self.linear_out = torch.nn.Linear(in_channels, in_channels)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.Layer_scale = LayerScale(dims=3, input_size=in_channels, Layer_scale_init=Layer_scale_init)
    
    def forward(self, x, pos_k, mask):
        """
        Compute 'Scaled Dot Product Attention'.
            :param torch.Tensor mask: (batch, time1, time2)
            :param torch.nn.Dropout dropout:
            :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
            weighted by the query dot key attention (batch, head, time1, time2)
        """
        n_batch = x.size(0)
        x = self.layer_norm(x)
        q = self.linear_q(x).view(n_batch, -1, self.h, self.d_k)  #(b, t, d)
        k = self.linear_k(x).view(n_batch, -1, self.h, self.d_k)  #(b, t, d)
        v = self.linear_v(x).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)
        A = torch.matmul(q, k.transpose(-2, -1))
        reshape_q = q.contiguous().view(n_batch * self.h, -1, self.d_k).transpose(0,1)
        if pos_k is not None:
            B = torch.matmul(reshape_q, pos_k.transpose(-2, -1))
            B = B.transpose(0, 1).view(n_batch, self.h, pos_k.size(0), pos_k.size(1))
            scores = (A + B) / math.sqrt(self.d_k)
        else:
            scores = A / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            min_value = float(np.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)
        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.Layer_scale(self.dropout(self.linear_out(x)))  # (batch, time1, d_model)

class DU_MHSA(torch.nn.Module):
    def __init__(self, in_channels: int, num_mha_heads: int, dropout_rate: float):
        super().__init__()
        self.block = torch.nn.ModuleDict({
            'self_attn': MultiHeadAttention(
                n_head=num_mha_heads, in_channels=in_channels, dropout_rate=dropout_rate),
            'linear': torch.nn.Sequential(
                torch.nn.LayerNorm(normalized_shape=in_channels), 
                torch.nn.Linear(in_features=in_channels, out_features=in_channels), 
                torch.nn.Sigmoid())
        })
    
    def forward(self, x: torch.Tensor, pos_k: torch.Tensor):
        """
        Compute encoded features.
            :param torch.Tensor x: encoded source features (batch, max_time_in, size)
            :param torch.Tensor mask: mask for x (batch, max_time_in)
            :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        down_len = pos_k.shape[0]
        x_down = torch.nn.functional.adaptive_avg_pool1d(input=x, output_size=down_len)
        x = x.permute([0, 2, 1])
        x_down = x_down.permute([0, 2, 1])
        x_down = self.block['self_attn'](x_down, pos_k, None)
        x_down = x_down.permute([0, 2, 1])
        x_downup = torch.nn.functional.upsample(input=x_down, size=x.shape[1])
        x_downup = x_downup.permute([0, 2, 1])
        x = x + self.block['linear'](x) * x_downup

        return x

class Heat1D(nn.Module):
    """
    1D Heat Equation Adaptation:
    du/dt - k d²u/dx² = 0;
    du/dx_{x=0, x=a} = 0
    =>
    A_n = C(a, n==0) * sum_{0}^{a} { \phi(x) cos(n π / a x) dx }
    core = cos(n π / a x) exp(- (n π / a)^2 k t)
    u_{x, t} = sum_{0}^{\infinite} { core }
    
    Assume a = T; x in [0, T]; n in [0, T]; with some slight changes
    =>
    (\phi(x) = linear(dwconv(input(x))))
    A(n) = DCT1D(\phi(x))
    u(x, t) = IDCT1D(A(n) * exp(- (n π / a)^2 kt))
    """
    def __init__(self, dim=96, hidden_dim=96, **kwargs):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.hidden_dim = hidden_dim
        self.linear = nn.Conv1d(hidden_dim, 2 * hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_linear = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim) 
        self.to_k = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.GELU(),
        )
 
        self.k = nn.Parameter(torch.ones(hidden_dim))
 
    @staticmethod
    def get_cos_map(N=224, device=torch.device("cpu"), dtype=torch.float):
        # cos((x + 0.5) / N * n * π) which is also the form of DCT and IDCT
        # DCT: F(n) = sum( (sqrt(2/N) if n > 0 else sqrt(1/N)) * cos((x + 0.5) / N * n * π) * f(x) )
        # IDCT: f(x) = sum( (sqrt(2/N) if n > 0 else sqrt(1/N)) * cos((x + 0.5) / N * n * π) * F(n) )
        # returns: (Res_n, Res_x)
        weight_x = (torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(1, -1) + 0.5) / N
        weight_n = torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(-1, 1)
        weight = torch.cos(weight_n * weight_x * torch.pi) * math.sqrt(2 / N)
        weight[0, :] = weight[0, :] / math.sqrt(2)
        return weight
 
    @staticmethod
    def get_decay_map(resolution=224, device=torch.device("cpu"), dtype=torch.float):
        # exp(- (n π / T)^2) for 1D
        # returns: (Res_t,)
        res_t = resolution
        weight_n = torch.linspace(0, torch.pi, res_t + 1, device=device, dtype=dtype)[:res_t]
        weight = torch.pow(weight_n, 2)
        weight = torch.exp(-weight)
        return weight
 
    def forward(self, x: torch.Tensor, freq_embed=None):
        B, T, C = x.shape
        x = x.transpose(1, 2)  # [B, T, C] -> [B, C, T]
        x = self.dwconv(x)  # [B, hidden_dim, T]
        
        x = self.linear(x)  # [B, 2 * hidden_dim, T]
        x, z = x.chunk(chunks=2, dim=1)  # [B, hidden_dim, T], [B, hidden_dim, T]
 
        if (T == getattr(self, "__RES__", 0)) and (getattr(self, "__WEIGHT_COSN__", None).device == x.device):
            weight_cosn = getattr(self, "__WEIGHT_COSN__", None)
            weight_exp = getattr(self, "__WEIGHT_EXP__", None)
            assert weight_cosn is not None
            assert weight_exp is not None
        else:
            weight_cosn = self.get_cos_map(T, device=x.device).detach_()
            weight_exp = self.get_decay_map(T, device=x.device).detach_()
            setattr(self, "__RES__", T)
            setattr(self, "__WEIGHT_COSN__", weight_cosn)
            setattr(self, "__WEIGHT_EXP__", weight_exp)
 
        N = weight_cosn.shape[0]  # N == T
        
        x = x.transpose(1, 2).contiguous()  # [B, T, hidden_dim]
        
        x = F.conv1d(x.contiguous().view(B, T, -1), weight_cosn.contiguous().view(N, T, 1))  # [B, N, hidden_dim]
        
        weight_exp = torch.pow(weight_exp[:, None], self.k)
        x = torch.einsum("bnc,nc->bnc", x, weight_exp)  # exp decay
        
        x = F.conv1d(x.contiguous().view(B, N, -1), weight_cosn.t().contiguous().view(T, N, 1))  # [B, T, hidden_dim]
 
        x = self.out_norm(x)  # [B, T, hidden_dim]
        
        z = z.transpose(1, 2).contiguous()  # [B, T, hidden_dim]
        x = x * nn.functional.silu(z)  # [B, T, hidden_dim]
        
        x = x.transpose(1, 2).contiguous()  # [B, hidden_dim, T]
        x = self.out_linear(x)  # [B, hidden_dim, T]
 
        x = x.transpose(1, 2).contiguous()  # [B, T, hidden_dim]
 
        return x
 
class CLA(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, dropout_rate, Layer_scale_init=1.0e-5):
        super().__init__()
        # self.layer_norm = torch.nn.LayerNorm(in_channels)
        self.heat1d = Heat1D(in_channels, in_channels)
        self.GN1 = torch.nn.GroupNorm(1, in_channels)
        self.dw_conv_1d = torch.nn.Conv1d(in_channels, in_channels, kernel_size, padding='same', groups=in_channels)  
        self.GN2 = torch.nn.GroupNorm(1, in_channels)
        self.linear3 = torch.nn.Sequential(
            torch.nn.GELU(),
            torch.nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            torch.nn.Dropout(dropout_rate))
        self.Layer_scale = LayerScale(dims=3, input_size=in_channels, Layer_scale_init=Layer_scale_init) 
    
    def forward(self, x):
        # y = self.layer_norm(x)
        y = self.heat1d(x)
        y = y.permute([0, 2, 1]) # B, F, T
        y = self.GN1(y)
        y = self.dw_conv_1d(y)
        y = self.linear3(y)
        y = self.GN2(y)  # [B, in_channels, T]
        y = y.permute(0, 2, 1)  # B, T, in_channels     
        return x + self.Layer_scale(y)
    
class GlobalBlock(torch.nn.Module):
    def __init__(self, in_channels: int, num_mha_heads: int, dropout_rate: float):
        super().__init__()
        self.block = torch.nn.ModuleDict({
            'DU_MHSA': DU_MHSA(
                num_mha_heads=num_mha_heads, in_channels=in_channels, dropout_rate=dropout_rate),
            'FFN': FFN(in_channels=in_channels, dropout_rate=dropout_rate)
        })
    
    def forward(self, x: torch.Tensor, pos_k: torch.Tensor):
        """
        Compute encoded features.
            :param torch.Tensor x: encoded source features (batch, max_time_in, size)
            :param torch.Tensor mask: mask for x (batch, max_time_in)
            :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        x = self.block['DU_MHSA'](x, pos_k)
        x = self.block['FFN'](x)
        x = x.permute([0, 2, 1])

        return x


class LocalBlock(torch.nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, dropout_rate: float):
        super().__init__()
        self.block = torch.nn.ModuleDict({
            'CLA': CLA(in_channels, kernel_size, dropout_rate),
            'FFN': FFN(in_channels, dropout_rate)
        })
    
    def forward(self, x: torch.Tensor):
        x = self.block['CLA'](x)
        x = self.block['FFN'](x)

        return x
    
class AudioEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, groups: int, bias: bool):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, groups=groups, bias=bias)
        self.gelu = torch.nn.GELU()
    
    def forward(self, x: torch.Tensor):
        x = torch.unsqueeze(x, dim=0) if len(x.shape) == 1 else torch.unsqueeze(x, dim=1) # [T] - >[1, T] OR [B, T] -> [B, 1, T]
        x = self.conv1d(x)
        x = self.gelu(x)
        return x
    
class FeatureProjector(torch.nn.Module):
    def __init__(self, num_channels: int, in_channels: int, out_channels: int, kernel_size: int, bias: bool):
        super().__init__()
        self.norm = torch.nn.GroupNorm(num_groups=1, num_channels=num_channels, eps=1e-8)
        self.conv1d = torch.nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias)
    
    def forward(self, x: torch.Tensor):
        x = self.norm(x)
        x = self.conv1d(x)
        return x

class HeatConvNorm(nn.Module):
    """
    This class defines the convolution layer with normalization and PReLU activation
    """

    def __init__(
        self, nIn, nOut, kSize, stride=1, groups=1, bias=True, norm_type="gLN"
    ):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = Heat1D(
            nIn, nOut, groups=groups
        )
        if norm_type == "gLN":
            self.norm = nn.GroupNorm(1, nOut, eps=1e-8)
        if norm_type == "BN":
            self.norm = nn.BatchNorm1d(nOut)

    def forward(self, input):
        input = input.permute(0, 2, 1)
        output = self.conv(input).permute(0, 2, 1)
        return self.norm(output)

class ConvNorm(nn.Module):
    """
    This class defines the convolution layer with normalization and PReLU activation
    """

    def __init__(
        self, nIn, nOut, kSize, stride=1, groups=1, bias=True, norm_type="gLN"
    ):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(
            nIn, nOut, kSize, stride=stride, padding=padding, bias=bias, groups=groups
        )
        if norm_type == "gLN":
            self.norm = nn.GroupNorm(1, nOut, eps=1e-8)
        if norm_type == "BN":
            self.norm = nn.BatchNorm1d(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)
    
class AVFModule(nn.Module):
    """
    1D Attention Fusion Cell，将 tensor_b 导引 tensor_a 的 key & value：
      Input:
        tensor_a: [B, Ca, T] 
        tensor_b: [B, Cb, Tb]
      Output:
        [B, Ca, T]
    """
    def __init__(self,
                 in_chan_a: int,
                 in_chan_b: int,
                 kernel_size: int = 1):
        super().__init__()
        self.in_chan_a = in_chan_a
        self.in_chan_b = in_chan_b
        self.kernel_size = kernel_size
        # audio key embedding (depthwise 1×1)
        self.key_embed = ConvNormAct(
            nIn=in_chan_a, nOut=in_chan_a, kSize=1,
            groups=in_chan_a, norm_type="gLN"
        )
        # audio value embedding (depthwise 1×1)
        self.value_embed = ConvNormAct(
            nIn=in_chan_a, nOut=in_chan_a, kSize=1,
            groups=in_chan_a, norm_type="gLN"
        )
        
        self.resize = ConvNormAct(
            nIn=in_chan_b, nOut=in_chan_a, kSize=1,
            norm_type="gLN"
        )
        
        self.attention_embed = ConvNormAct(
            nIn=in_chan_b,
            nOut=in_chan_a * kernel_size,
            kSize=1,
            groups=in_chan_b,
            norm_type="gLN"
        )

    def forward(self, tensor_a: torch.Tensor, tensor_b: torch.Tensor):
        """
        tensor_a: [B, Ca, T]
        tensor_b: [B, Cb, Tb]
        """
        B, Ca, T = tensor_a.shape
        # 1) Use video to guide key_embed
        b2a = self.resize(tensor_b)               # [B, Ca, Tb]
        b2a = F.interpolate(b2a, size=T, mode="nearest")  # [B, Ca, T]
        k1 = self.key_embed(tensor_a) * b2a       # [B, Ca, T]
        # 2) audio value
        v = self.value_embed(tensor_a)            # [B, Ca, T]
        # 3) Calculate attention scores
        att = self.attention_embed(tensor_b)      # [B, Ca*kernel, Tb]
        # reshape → [B, Ca, kernel, Tb]
        att = att.view(B, Ca, self.kernel_size, -1)
        att = att.mean(dim=2)
        att = torch.softmax(att, dim=-1)          # [B, Ca, Tb]
        att = F.interpolate(att, size=T, mode="nearest")  # [B, Ca, T]
        # 4) k2 = attention * value
        k2 = att * v            
        
        fused = k1 + k2                           # [B, Ca, T]
        return fused

class RelativePositionalEncoding(torch.nn.Module):
    def __init__(self, in_channels: int, num_heads: int, maxlen: int, embed_v=False):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.embedding_dim = self.in_channels // self.num_heads
        self.maxlen = maxlen
        self.pe_k = torch.nn.Embedding(num_embeddings=2*maxlen, embedding_dim=self.embedding_dim)
        self.pe_v = torch.nn.Embedding(num_embeddings=2*maxlen, embedding_dim=self.embedding_dim) if embed_v else None
    
    def forward(self, pos_seq: torch.Tensor):
        pos_seq.clamp_(-self.maxlen, self.maxlen - 1)
        pos_seq += self.maxlen
        pe_k_output = self.pe_k(pos_seq)
        pe_v_output = self.pe_v(pos_seq) if self.pe_v is not None else None
        return pe_k_output, pe_v_output

class DownConvLayer(torch.nn.Module):
    def __init__(self, in_channels: int, samp_kernel_size: int):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.down_conv = torch.nn.Conv1d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=samp_kernel_size, stride=2, padding=(samp_kernel_size-1)//2, groups=in_channels)
        self.GN = nn.GroupNorm(1, num_channels=in_channels)
        self.gelu = torch.nn.GELU()
    
    def forward(self, x: torch.Tensor):
        x = x.permute([0, 2, 1])
        x = self.down_conv(x)
        x = self.GN(x)
        x = self.gelu(x)
        x = x.permute([0, 2, 1])
        return x

class ConvNormAct(nn.Module):
    """
    This class defines the convolution layer with normalization and a PReLU
    activation
    """

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1, norm_type="gLN"):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(
            nIn, nOut, kSize, stride=stride, padding=padding, bias=True, groups=groups
        )
        if norm_type == "gLN":
            self.norm = nn.GroupNorm(1, nOut, eps=1e-8)
        if norm_type == "BN":
            self.norm = nn.BatchNorm1d(nOut)
        self.act = nn.PReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.norm(output)
        return self.act(output)

class DilatedConvNorm(nn.Module):
    """
    This class defines the dilated convolution with normalized output.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1, norm_type="gLN"):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        self.conv = nn.Conv1d(
            nIn,
            nOut,
            kSize,
            stride=stride,
            dilation=d,
            padding=((kSize - 1) // 2) * d,
            groups=groups,
        )
        # self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
        if norm_type == "gLN":
            self.norm = nn.GroupNorm(1, nOut, eps=1e-8)
        if norm_type == "BN":
            self.norm = nn.BatchNorm1d(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_size, drop=0.1, norm_type="gLN"):
        super().__init__()
        self.fc1 = ConvNorm(
            in_features, hidden_size, 1, bias=False, norm_type=norm_type
        )
        self.dwconv = nn.Conv1d(
            hidden_size, hidden_size, 5, 1, 2, bias=True, groups=hidden_size
        )
        self.act = nn.ReLU()
        self.fc2 = ConvNorm(
            hidden_size, in_features, 1, bias=False, norm_type=norm_type
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class InjectionMultiSum(nn.Module):
    def __init__(self, inp: int, oup: int, kernel: int = 1, norm_type="gLN") -> None:
        super().__init__()
        groups = 1
        if inp == oup:
            groups = inp
        self.local_embedding = HeatConvNorm(    
            inp, oup, kernel, groups=groups, bias=False, norm_type=norm_type
        )
        self.global_embedding = HeatConvNorm(
            inp, oup, kernel, groups=groups, bias=False, norm_type=norm_type
        )
        self.global_act = HeatConvNorm(
            inp, oup, kernel, groups=groups, bias=False, norm_type=norm_type
        )
        self.act = nn.Sigmoid()

    def forward(self, x_l, x_g):
        """
        x_g: global features
        x_l: local features
        """
        B, N, T = x_l.shape
        local_feat = self.local_embedding(x_l)

        global_act = self.global_act(x_g)
        sig_act = torch.nn.functional.interpolate(self.act(global_act), size=T, mode="nearest")
        # sig_act = self.act(global_act)

        global_feat = self.global_embedding(x_g)
        global_feat = torch.nn.functional.interpolate(global_feat, size=T, mode="nearest")

        out = local_feat * sig_act + global_feat
        return out
    
class UConvBlock(nn.Module):
    """
    This class defines the block which performs successive downsampling and
    upsampling in order to be able to analyze the input features in multiple
    resolutions.
    """

    def __init__(
        self, out_channels=128, in_channels=512, upsampling_depth=4, norm_type="gLN"
    ):
        super().__init__()
        self.proj_1x1 = ConvNormAct(out_channels, in_channels, 1, stride=1, groups=1, norm_type=norm_type)
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList()
        self.spp_dw.append(
            DilatedConvNorm(
                in_channels, in_channels, kSize=5, stride=1, groups=in_channels, d=1, norm_type=norm_type
            )
        )
        for i in range(1, upsampling_depth):
            self.spp_dw.append(
                DilatedConvNorm(
                    in_channels,
                    in_channels,
                    kSize=5,
                    stride=2,
                    groups=in_channels,
                    d=1,
                    norm_type=norm_type
                )
            )

        self.loc_glo_fus = nn.ModuleList([])
        for i in range(upsampling_depth):
            self.loc_glo_fus.append(InjectionMultiSum(in_channels, in_channels, norm_type=norm_type))

        self.res_conv = nn.Conv1d(in_channels, out_channels, 1)

        self.globalatt = Mlp(in_channels, in_channels, drop=0.1)

        self.last_layer = nn.ModuleList([])
        for i in range(self.depth - 1):
            self.last_layer.append(InjectionMultiSum(in_channels, in_channels, 5, norm_type=norm_type))

    def forward(self, x):
        """
        :param x: input feature map
        :return: transformed feature map
        """
        residual = x.clone()
        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]

        # Do the downsampling process from the previous level
        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        # global features
        global_f = torch.zeros(
            output[-1].shape, requires_grad=True, device=output1.device
        )
        for fea in output:
            global_f = global_f + torch.nn.functional.adaptive_avg_pool1d(
                fea, output_size=output[-1].shape[-1]
            )
            # global_f = global_f + fea
        global_f = self.globalatt(global_f)  # [B, N, T]

        x_fused = []
        # Gather them now in reverse order
        for idx in range(self.depth):
            local = output[idx]
            x_fused.append(self.loc_glo_fus[idx](local, global_f))

        expanded = None
        for i in range(self.depth - 2, -1, -1):
            if i == self.depth - 2:
                expanded = self.last_layer[i](x_fused[i], x_fused[i - 1])
            else:
                expanded = self.last_layer[i](x_fused[i], expanded)
        # import pdb; pdb.set_trace()
        return self.res_conv(expanded) + residual

class EncoderLayer(torch.nn.Module):
    def __init__(self, global_blocks: dict, local_blocks: dict, down_conv_layer: dict, down_conv=True):
        super().__init__()
        
        self.g_block_1 = GlobalBlock(**global_blocks)
        self.l_block_1 = LocalBlock(**local_blocks)
        
        self.g_block_2 = GlobalBlock(**global_blocks)
        self.l_block_2 = LocalBlock(**local_blocks)
        
        self.downconv = DownConvLayer(**down_conv_layer) if down_conv == True else None
        
    def forward(self, x: torch.Tensor, pos_k: torch.Tensor):
        '''
        x: [B, N, T]
        '''
        x = self.g_block_1(x, pos_k)
        x = x.permute(0, 2, 1).contiguous()
        x = self.l_block_1(x)
        x = x.permute(0, 2, 1).contiguous()
        
        x = self.g_block_2(x, pos_k)
        x = x.permute(0, 2, 1).contiguous()
        x = self.l_block_2(x)
        x = x.permute(0, 2, 1).contiguous()
        
        skip = x
        if self.downconv:
            x = x.permute(0, 2, 1).contiguous()
            x = self.downconv(x)
            x = x.permute(0, 2, 1).contiguous()
        # [BK, S, N]
        return x, skip

class DecoderLayer(torch.nn.Module):
    def __init__(self, global_blocks: dict, local_blocks: dict, spk_attention: dict):
        super().__init__()
        
        self.g_block_1 = GlobalBlock(**global_blocks)
        self.l_block_1 = LocalBlock(**local_blocks)
        
        self.g_block_2 = GlobalBlock(**global_blocks)
        self.l_block_2 = LocalBlock(**local_blocks)
        
        self.g_block_3 = GlobalBlock(**global_blocks)
        self.l_block_3 = LocalBlock(**local_blocks)
    
    def forward(self, x: torch.Tensor, pos_k: torch.Tensor):
        '''
        x: [B, N, T]
        '''
        # [BS, K, H]
        x = self.g_block_1(x, pos_k)
        x = x.permute(0, 2, 1).contiguous()
        x = self.l_block_1(x)
        x = x.permute(0, 2, 1).contiguous()
        
        x = self.g_block_2(x, pos_k)
        x = x.permute(0, 2, 1).contiguous()
        x = self.l_block_2(x)
        x = x.permute(0, 2, 1).contiguous()
        
        x = self.g_block_3(x, pos_k)
        x = x.permute(0, 2, 1).contiguous()
        x = self.l_block_3(x)
        x = x.permute(0, 2, 1).contiguous()
        
        skip = x
        
        return x, skip

class Separator(torch.nn.Module):
    def __init__(self, num_stages: int, relative_positional_encoding: dict, enc_stage: dict, simple_fusion:dict, dec_stage: dict):
        super().__init__()
        
        self.num_stages = num_stages
        self.pos_emb = RelativePositionalEncoding(**relative_positional_encoding)
        
        # Temporal Contracting Part
        self.enc_stages = torch.nn.ModuleList([])
        for _ in range(self.num_stages):
            self.enc_stages.append(EncoderLayer(**enc_stage, down_conv=True))
        
        self.bottleneck_G = nn.ModuleList([
            MultiHeadAttention(
                n_head=enc_stage['global_blocks']['num_mha_heads'],
                in_channels=enc_stage['global_blocks']['in_channels'],
                dropout_rate=enc_stage['global_blocks']['dropout_rate']
            ),
            FFN(
                in_channels=enc_stage['global_blocks']['in_channels'],
                dropout_rate=enc_stage['global_blocks']['dropout_rate']
            )
        ])

        # top-down fusion
        self.loc_glo_fus = nn.ModuleList([])
        for i in range(self.num_stages):
            self.loc_glo_fus.append(InjectionMultiSum(simple_fusion['out_channels'], simple_fusion['out_channels']))
        
        # Temporal Expanding Part
        self.simple_fusion = torch.nn.ModuleList([])
        self.dec_stages = torch.nn.ModuleList([])
        for _ in range(self.num_stages):
            self.simple_fusion.append(InjectionMultiSum(simple_fusion['out_channels'], simple_fusion['out_channels'], kernel=5))
            self.dec_stages.append(DecoderLayer(**dec_stage))
    
    def forward(self, input: torch.Tensor):
        '''input: [B, N, L]'''
        # feature projection
        x, _ = self.pad_signal(input)
        len_x = x.shape[-1]
        # Temporal Contracting Part
        min_len = len_x//2**(self.num_stages-1)
        pos_seq = torch.arange(0, len_x//2**self.num_stages).long().to(x.device)
        pos_seq = pos_seq[:, None] - pos_seq[None, :]
        pos_k, _ = self.pos_emb(pos_seq)

        skip = []
        fusion_x = torch.zeros([x.shape[0], x.shape[1], min_len], requires_grad=True, device=x.device)
        for idx in range(self.num_stages):
            x, skip_ = self.enc_stages[idx](x, pos_k)
            skip.append(skip_)
            fusion_x = fusion_x + F.adaptive_avg_pool1d(x, min_len)

        global_x = self.bottleneck_G[0](fusion_x.permute(0, 2, 1).contiguous(), None, None)
        global_x = self.bottleneck_G[1](global_x).permute(0, 2, 1).contiguous()
        
        # Global topdown attention
        fusion_skip = []
        for idx in range(self.num_stages):
            fusion_skip.append(self.loc_glo_fus[idx](skip[idx], global_x))
        
        each_stage_outputs = []
        # Temporal Expanding Part
        for idx in range(self.num_stages):
            each_stage_outputs.append(x)
            idx_en = self.num_stages - (idx + 1)
            x = self.simple_fusion[idx](fusion_skip[idx_en], x)
            x, _ = self.dec_stages[idx](x, pos_k)
        
        last_stage_output = x 
        return last_stage_output, each_stage_outputs
    
    def pad_signal(self, input: torch.Tensor):
        #  (B, T) or (B, 1, T)
        if input.dim() == 1: input = input.unsqueeze(0)
        elif input.dim() not in [2, 3]: raise RuntimeError("Input can only be 2 or 3 dimensional.")
        elif input.dim() == 2: input = input.unsqueeze(1)
        L = 2**self.num_stages
        batch_size = input.size(0)  
        ndim = input.size(1)
        nframe = input.size(2)
        padded_len = (nframe//L + 1)*L
        rest = 0 if nframe%L == 0 else padded_len - nframe
        if rest > 0:
            pad = torch.autograd.Variable(torch.zeros(batch_size, ndim, rest)).type(input.type()).to(input.device)
            input = torch.cat([input, pad], dim=-1)
        return input, rest


class OutputLayer(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, masking: bool = False):
        super().__init__()
        # feature expansion back
        self.masking = masking
        self.spe_block = Masking(in_channels)
        self.end_conv1x1 = torch.nn.Sequential(
            torch.nn.Linear(out_channels, 4*out_channels),
            torch.nn.GLU(),
            torch.nn.Linear(2*out_channels, in_channels))
            
    def forward(self, x: torch.Tensor, input: torch.Tensor):
        x = x[...,:input.shape[-1]]
        x = x.permute([0, 2, 1])
        x = self.end_conv1x1(x)
        x = x.permute([0, 2, 1])
        
        if self.masking:
            x = self.spe_block(x, input)
        
        return x

class AudioDecoder(torch.nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        # x: [B, N, L]
        if x.dim() not in [2, 3]: 
            raise RuntimeError("{} accept 2/3D tensor as input".format(self.__class__.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        x = torch.squeeze(x, dim=1) if torch.squeeze(x).dim() == 1 else torch.squeeze(x)
        return x

class ReconstructionPath(nn.Module):
    def __init__(
        self,
        layers = [
            'residual',
            'residual',
            'residual'
        ],
        image_size=88,
        in_channel=1,
        init_channel=16,
        max_dim=128,
        # conv相关
        input_conv_kernel_size = [7, 7, 7],
        output_conv_kernel_size = [3, 3, 3],
        residual_conv_kernel_size=3,
        pad_mode="constant",
        # attn相关
        attn_dim_head = 32,
        attn_heads = 8,
        attn_dropout = 0.,
        flash_attn = True,
        linear_attn_dim_head = 8,
        linear_attn_heads = 16,
        fuse_dim=32,
        # quantizer相关
        num_quantizers = 1,
        codebook_size = 256,
        codebook_dim=64,
        commitment_cost=0.25,
    ):
        super().__init__()
        input_conv_kernel_size=tuple(input_conv_kernel_size)

        self.conv_in = nn.Conv3d(in_channel, init_channel, input_conv_kernel_size,padding='same')

        layer_fmap_size=image_size
        self.encoder_layers = nn.ModuleList([]) 
        dim=init_channel
        dim_out=dim
        time_downsample_factor=1

        for layer_type in layers:
            if layer_type == 'residual':
                encoder_layer = ResidualUnit(dim, residual_conv_kernel_size)

            elif layer_type == 'consecutive_residual':
                num_consecutive = 2
                encoder_layer = Sequential(*[ResidualUnit(dim, residual_conv_kernel_size) for _ in range(num_consecutive)])

            elif layer_type == 'compress_space':
                dim_out = dim * 2
                dim_out = min(dim_out, max_dim)

                encoder_layer = SpatialDownsample2x(dim, dim_out)

                assert layer_fmap_size > 1
                layer_fmap_size //= 2

            elif layer_type == 'compress_time':
                dim_out = dim * 2
                dim_out = min(dim_out, max_dim)

                encoder_layer = TimeDownsample2x(dim, dim_out)

                time_downsample_factor *= 2

            elif layer_type == 'attend_space':
                attn_kwargs = dict(
                    dim = dim,
                    dim_head = attn_dim_head,
                    heads = attn_heads,
                    dropout = attn_dropout,
                    flash = flash_attn
                )

                encoder_layer = Sequential(
                    Residual(SpaceAttention(**attn_kwargs)),
                    Residual(FeedForward(dim))
                )

            elif layer_type == 'linear_attend_space':
                linear_attn_kwargs = dict(
                    dim = dim,
                    dim_head = linear_attn_dim_head,
                    heads = linear_attn_heads
                )

                encoder_layer = Sequential(
                    Residual(LinearSpaceAttention(**linear_attn_kwargs)),
                    Residual(FeedForward(dim))
                )

            else:
                raise ValueError(f'unknown layer type {layer_type}')

            self.encoder_layers.append(encoder_layer)

            dim = dim_out
        
        self.encoder_layers.append(Sequential(
            Rearrange('b c ... -> b ... c'),
            nn.LayerNorm(dim),
            Rearrange('b ... c -> b c ...'),
        ))


    def forward(self, x, semantic_fea=None):
        x = self.conv_in(x)
        for layer in self.encoder_layers:
            x = layer(x)
        z_e = x

        z_q=z_e

        if semantic_fea!=None:
            B,C,T,H,W=z_q.shape
            z_q=z_q.contiguous().permute(0,2,1,3,4)
            z_q=z_q.contiguous().view(B,T,-1)
            z_q=z_q + semantic_fea
        
        return z_q

class SemanticPath(nn.Module):
    def __init__(
        self,
        layers = [
            'residual',
            'residual',
            'residual'
        ],
        image_size=88,
        in_channel=1,
        init_channel=4,
        max_dim=32,
        input_conv_kernel_size = [7, 7, 7],
        output_conv_kernel_size = [3, 3, 3],
        residual_conv_kernel_size=3,
        pad_mode="constant",
        attn_dim_head = 32,
        attn_heads = 8,
        attn_dropout = 0.,
        flash_attn = True,
        linear_attn_dim_head = 8,
        linear_attn_heads = 16,
        num_quantizers = 1,
        codebook_size = 256,
        codebook_dim= 64,
        commitment_cost=0.25,
        distill_dim=1024,
        config=None,
        pretrain=None
    ):
        super().__init__()
        input_conv_kernel_size=tuple(input_conv_kernel_size)

        self.conv_in = nn.Conv3d(in_channel, init_channel, input_conv_kernel_size,padding='same')

        layer_fmap_size=image_size
        self.encoder_layers = nn.ModuleList([]) 
        dim=init_channel
        dim_out=dim
        time_downsample_factor=1

        for layer_type in layers:
            if layer_type == 'residual':
                encoder_layer = ResidualUnit(dim, residual_conv_kernel_size)

            elif layer_type == 'consecutive_residual':
                num_consecutive = 2
                encoder_layer = Sequential(*[ResidualUnit(dim, residual_conv_kernel_size) for _ in range(num_consecutive)])

            elif layer_type == 'compress_space':
                dim_out = dim * 2
                dim_out = min(dim_out, max_dim)

                encoder_layer = SpatialDownsample2x(dim, dim_out)

                assert layer_fmap_size > 1
                layer_fmap_size //= 2

            elif layer_type == 'compress_time':
                dim_out = dim * 2
                dim_out = min(dim_out, max_dim)

                encoder_layer = TimeDownsample2x(dim, dim_out)

                time_downsample_factor *= 2

            elif layer_type == 'attend_space':
                attn_kwargs = dict(
                    dim = dim,
                    dim_head = attn_dim_head,
                    heads = attn_heads,
                    dropout = attn_dropout,
                    flash = flash_attn
                )

                encoder_layer = Sequential(
                    Residual(SpaceAttention(**attn_kwargs)),
                    Residual(FeedForward(dim))
                )

            elif layer_type == 'linear_attend_space':
                linear_attn_kwargs = dict(
                    dim = dim,
                    dim_head = linear_attn_dim_head,
                    heads = linear_attn_heads
                )

                encoder_layer = Sequential(
                    Residual(LinearSpaceAttention(**linear_attn_kwargs)),
                    Residual(FeedForward(dim))
                )

            else:
                raise ValueError(f'unknown layer type {layer_type}')

            self.encoder_layers.append(encoder_layer)

            dim = dim_out
        
        self.encoder_layers.append(Sequential(
            Rearrange('b c ... -> b ... c'),
            nn.LayerNorm(dim),
            Rearrange('b ... c -> b c ...'),
        ))

        # layer_fmap_size = 3
        self.quantizer = ResidualVQ(
            dim = dim*layer_fmap_size*layer_fmap_size,  
            num_quantizers = num_quantizers,
            codebook_size = codebook_size,
            codebook_dim = codebook_dim,
            quantize_dropout=False,
            stochastic_sample_codes = True,
            sample_codebook_temp = 0.1,
            kmeans_init = True, 
            kmeans_iters = 10 
        )

    def forward(self, x):
        x = self.conv_in(x)
        for layer in self.encoder_layers:
            x = layer(x)
        b,c,t,h,w=x.shape
        x = x.contiguous().permute(0,2,1,3,4)
        z_e = x.contiguous().view(b,t,-1)

        z_q,_,_=self.quantizer(z_e)

        return z_q

class VideoEncoder(nn.Module):
    def __init__(
        self,
        layers,
        image_size=88,
        in_channel=1,
        init_channel=16,
        max_dim=128,
        input_conv_kernel_size = [7, 7, 7],
        output_conv_kernel_size = [3, 3, 3],
        residual_conv_kernel_size=3,
        pad_mode="constant",
        # attn相关
        attn_dim_head = 32,
        attn_heads = 8,
        attn_dropout = 0.,
        flash_attn = True,
        linear_attn_dim_head = 8,
        linear_attn_heads = 16,
        num_quantizers = 1,
        codebook_size = 256,
        codebook_dim=64,
        commitment_cost=0.25,
        distill_cost=1.0,
    ):
        super().__init__()
        self.semantic_model=SemanticPath(
            layers=layers,
            image_size=image_size,
            in_channel=in_channel,
            init_channel=init_channel,
            max_dim=max_dim,
            input_conv_kernel_size=input_conv_kernel_size,
            output_conv_kernel_size=output_conv_kernel_size,
            residual_conv_kernel_size=residual_conv_kernel_size,
            pad_mode=pad_mode,
            attn_dim_head = attn_dim_head,
            attn_heads = attn_heads,
            attn_dropout = attn_dropout,
            flash_attn = flash_attn,
            linear_attn_dim_head = linear_attn_dim_head,
            linear_attn_heads = linear_attn_heads,
            num_quantizers = num_quantizers,
            codebook_size = codebook_size,
            codebook_dim = codebook_dim,
            commitment_cost = commitment_cost,
        )
        self.recon_model=ReconstructionPath(
            layers=layers,
            image_size=image_size,
            in_channel=in_channel,
            init_channel=init_channel,
            max_dim=max_dim,
            input_conv_kernel_size=input_conv_kernel_size,
            output_conv_kernel_size=output_conv_kernel_size,
            residual_conv_kernel_size=residual_conv_kernel_size,
            pad_mode=pad_mode,
            attn_dim_head = attn_dim_head,
            attn_heads = attn_heads,
            attn_dropout = attn_dropout,
            flash_attn = flash_attn,
            linear_attn_dim_head = linear_attn_dim_head,
            linear_attn_heads = linear_attn_heads,
            num_quantizers = num_quantizers,
            codebook_size = codebook_size,
            codebook_dim = codebook_dim,
            commitment_cost = commitment_cost,
        )

    def forward(self, x):
        semantic_fea = self.semantic_model(x)
        return self.recon_model(x,semantic_fea)
    
class Dolphin(nn.Module, PyTorchModelHubMixin):
    def __init__(self, 
                 num_stages: int, 
                 sample_rate: int,
                 module_audio_enc: dict, 
                 module_feature_projector: dict, 
                 module_separator: dict, 
                 module_output_layer: dict, 
                 module_audio_dec: dict,
                 video_encoder_params: dict,
                 vpre_channels=512,
                 vmid_channels=512,
                 vin_channels=64,
                 vout_channels=64,):
        super(Dolphin, self).__init__()

        self.pre_v1 = ConvNormAct(vpre_channels, vin_channels, kSize=3, norm_type="BN")

        self.num_stages = num_stages
        self.audio_encoder = AudioEncoder(**module_audio_enc) 
        self.feature_projector = FeatureProjector(**module_feature_projector)
        self.separator = Separator(**module_separator)
        self.out_layer = OutputLayer(**module_output_layer)
        self.audio_decoder = AudioDecoder(**module_audio_dec)
        
        self.video_blocks = UConvBlock(vin_channels, vout_channels, 3, norm_type="BN")
        self.modalfuse = AVFModule(module_feature_projector["out_channels"], vout_channels)
    
        self.video_encoder = VideoEncoder(**video_encoder_params)
    
    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str,
        cache_dir: str,
        force_download: bool,
        proxies: dict,
        resume_download: bool,
        local_files_only: bool,
        token: str,
        map_location: str = "cpu",
        strict: bool = False,
        **model_kwargs,
    ):
        """Load model from HuggingFace Hub with proper configuration handling."""
        import json
        from huggingface_hub import hf_hub_download
        
        # Download config file
        config_file = hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
        )
        
        # Load configuration
        with open(config_file, "r") as f:
            config = json.load(f)
        
        # Extract only the model parameters, excluding HF metadata
        hf_metadata_keys = {
            "model_type", "task", "framework", "license", "tags", 
            "architectures", "auto_map"
        }
        model_config = {k: v for k, v in config.items() if k not in hf_metadata_keys}
        
        # Create model instance with config
        model = cls(**model_config)
        
        # Try to download different possible model file formats
        import torch
        model_files_to_try = [
            "model.safetensors", 
        ]
        
        state_dict = None
        for filename in model_files_to_try:
            try:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=filename,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    token=token,
                )
                
                # Try to load the state dict
                if filename.endswith('.safetensors'):
                    # Handle safetensors format
                    try:
                        from safetensors.torch import load_file
                        state_dict = load_file(model_file, device=map_location)
                    except ImportError:
                        print("safetensors not available, skipping .safetensors files")
                        continue
                else:
                    # Handle PyTorch format
                    checkpoint = torch.load(model_file, map_location=map_location, weights_only=False)
                    
                    # Handle different checkpoint formats
                    if isinstance(checkpoint, dict):
                        if 'state_dict' in checkpoint:
                            state_dict = checkpoint['state_dict']
                        elif 'model_state_dict' in checkpoint:
                            state_dict = checkpoint['model_state_dict']
                        else:
                            state_dict = checkpoint
                    else:
                        state_dict = checkpoint
                
                # If we successfully loaded a state dict, break
                if state_dict is not None:
                    break
                    
            except Exception as e:
                print(f"Failed to load {filename}: {e}")
                continue
        
        if state_dict is None:
            raise RuntimeError(f"Could not load model weights from any of the tried files: {model_files_to_try}")
        
        model.load_state_dict(state_dict, strict=strict)
        
        return model
        
    def forward(self, input, mouth):
        mouth = self.video_encoder(mouth).permute(0, 2, 1).contiguous()
        v=self.pre_v1(mouth)
        v=self.video_blocks(v)

        encoder_output = self.audio_encoder(input)
        projected_feature = self.feature_projector(encoder_output)
        
        projected_feature = self.modalfuse(projected_feature,v)

        last_stage_output, each_stage_outputs = self.separator(projected_feature)
    
        out_layer_output = self.out_layer(last_stage_output, encoder_output)
        audio=self.audio_decoder(out_layer_output)
        
        return audio.unsqueeze(dim=1)