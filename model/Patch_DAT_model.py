from datetime import time
from typing import Callable, Optional
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from utils import *

def compute_derivatives(waveform,sequence_len,interval):
    waveform = np.squeeze(waveform, axis=0)
    time = np.linspace(0, interval * (sequence_len - 1), sequence_len)
    first_derivative = np.gradient(waveform,time)
    return first_derivative

class Patch_DAT_backbone(nn.Module):
    def __init__(self, c_in: int, sequence_len: int, target: int, patch_len: int, stride: int, n_layers:int,d_model:int,n_heads:int,
                 max_seq_len: Optional[int] = 1024,d_k: Optional[int] = None, d_v: Optional[int] = None,d_ff: int = 256, norm: str = 'BatchNorm',
                 attn_dropout: float = 0., dropout: float = 0.,act: str = "gelu", key_padding_mask: bool = 'auto',padding_var: Optional[int] = None,
                 attn_mask: Optional[Tensor] = None, res_attention: bool = True,pre_norm: bool = False, store_attn: bool = False,pe: str = 'zeros',
                 learn_pe: bool = True, fc_dropout: float = 0., head_dropout=0, padding_patch=None,pretrain_head: bool = False, head_type='flatten',
                 individual=False, revin=False, affine=True,subtract_last=False,verbose: bool = False, **kwargs):
        super().__init__()

        '''
        main parameters:
        c_in: The number of input feature channels, typically corresponding to the dimensionality of the input data. Default is 1
        target: The size of output
        sequence_len: The size of the origin sequence.
        patch_len: The length of each data patch, influencing how the model segments the input data into smaller blocks for processing.
        stride: The step size between data patches, determining the degree of overlap between patches.
        n_layers,d_model,n_heads: These parameters define the number of layers, model dimensionality, and the number of attention heads.
        d_ff: The dimensionality of the feed-forward network.
        ...
        '''
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((sequence_len - patch_len) / stride + 1)
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        # Backbone
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                    n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                    attn_dropout=attn_dropout, dropout=dropout, act=act,
                                    key_padding_mask=key_padding_mask, padding_var=padding_var,
                                    attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                    store_attn=store_attn,
                                    pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)
        self.globalmaxpool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(patch_num, target)

    def forward(self, z):  # z: [bs, c_in, sequence_len]
        # Compute the first derivative for each waveform
        waveforms = z.cpu().numpy()
        first_derivatives = np.array([compute_derivatives(waveform,z.size(2),1/30) for waveform in waveforms])

        # Convert to tensor and prepare for patching
        first_derivatives_tensor = torch.tensor(first_derivatives).unsqueeze(1)
        first_derivatives_tensor = first_derivatives_tensor.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        # Apply padding if necessary
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)

        # Unfold the input tensor for patching
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # z: [bs, nvars, patch_num, patch_len]
        z = z.permute(0, 1, 3, 2)  # z: [bs, nvars, patch_len, patch_num]

        # Prepare the first derivative tensor for the model
        z_first_derivative = first_derivatives_tensor.permute(0, 1, 3, 2).to(z.device)

        # Pass through the backbone model
        z = self.backbone(z, z_first_derivative)  # z: [bs, nvars, d_model, patch_num]

        length2 = z.size(-1) // 2
        z = z[:, :, :, :length2]
        z = torch.squeeze(z)

        # Ensure the output has the correct dimensions
        if z.dim() == 2:
            z = z.unsqueeze(0)

        # Permute and apply global max pooling
        z = z.permute(0, 2, 1)
        z = self.globalmaxpool(z)
        z = z.view(-1, z.shape[1] * z.shape[2])  # Flatten the tensor for the fully connected layer
        z = self.fc(z)

        return z



class TSTiEncoder(nn.Module):
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=6, d_model=64, n_heads=8, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):
        super().__init__()
        self.patch_num = patch_num
        self.patch_len = patch_len
        self.res_attention = res_attention
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)  # projection of origin feature vectors onto a d-dim vector space
        self.W_P1 = nn.Linear(patch_len, d_model)  # projection of derivative feature vectors onto a d-dim vector space
        self.seq_len = q_len
        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)
        # Residual dropout
        self.dropout = nn.Dropout(dropout)
        # Encoder
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                  attn_dropout=attn_dropout, dropout=dropout,
                                  pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers,
                                  store_attn=store_attn)


    def forward(self, x,x_first_derivative) -> Tensor:  # x: [bs x nvars x patch_len x patch_num]
        n_vars = x.shape[1]
        x = x.permute(0, 1, 3, 2)  # x: [bs x nvars x patch_num x patch_len]
        x_first_derivative=x_first_derivative.permute(0, 1, 3, 2)

        x = self.W_P(x)  # x: [bs x nvars x patch_num x d_model]
        x_first_derivative=self.W_P1(x_first_derivative)

        u = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        u1 = torch.reshape(x_first_derivative, (x_first_derivative.shape[0] * x_first_derivative.shape[1], x_first_derivative.shape[2], x_first_derivative.shape[3]))

        u = self.dropout(u + self.W_pos)  # u: [bs * nvars x patch_num x d_model]
        u1 = self.dropout(u1 + self.W_pos)  # u: [bs * nvars x patch_num x d_model]
        u = torch.cat((u, u1), dim=1)
        z = self.encoder(u)  # z: [bs * nvars x patch_num x d_model]

        if self.res_attention:
            score = z[1]
            z = torch.reshape(z[0], (-1, n_vars, z[0].shape[-2], z[0].shape[-1]))  # z: [bs x nvars x patch_num x d_model]
            z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x d_model x patch_num]
        else:
            score=0
            z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))  # z: [bs x nvars x patch_num x d_model]
            z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x d_model x patch_num]
        return z


class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None,
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList(
            [TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                             attn_dropout=attn_dropout, dropout=dropout,
                             activation=activation, res_attention=res_attention,
                             pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        # print(f"n_heads: {n_heads}")
        # print(f"dim: {d_model}")
        self.res_attention = res_attention

    def forward(self, src: Tensor, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask,attn_mask=attn_mask)
            return output, scores
        else:
            for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output


class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False,
                 pre_norm=False):
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout,proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                # nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src: Tensor, prev: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2,src2_1, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask,attn_mask=attn_mask) # y+y'
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        if self.store_attn:
            self.attn = attn

        ## Add & Norm
        src2 = src2.repeat(1, 2, 1)
        src2_1 = src2_1.repeat(1, 2, 1)
        src = src + self.dropout_attn(src2+src2_1)
        if not self.pre_norm:
            src = self.norm_attn(src)
        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)
        if self.res_attention:
            return src, scores
        else:
            return src


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0.,
                 qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_Q1 = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)
        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout,res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))
        self.to_out1 = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q: Tensor, K: Optional[Tensor] = None, V: Optional[Tensor] = None, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q
        len1 = int(Q.size(1)/2)
        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q[:, :len1, :]).view(bs, -1, self.n_heads, self.d_k).transpose(1,2) # q
        q_s_1 = self.W_Q1(Q[:, len1:Q.size(1), :]).view(bs, -1, self.n_heads, self.d_k).transpose(1,2) # q'
        k_s = self.W_K(K[:, :len1, :]).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3,1) # k
        v_s = self.W_V(V[:, :len1, :]).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2) # v

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            if prev == None:
                output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev,key_padding_mask=key_padding_mask,attn_mask=attn_mask)
                output1, attn_weights1, attn_scores1 = self.sdp_attn(q_s_1, k_s, v_s, prev=prev,key_padding_mask=key_padding_mask,attn_mask=attn_mask)
            else:
                output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev[:, :prev.size(1)//2, :],key_padding_mask=key_padding_mask, attn_mask=attn_mask)
                output1, attn_weights1, attn_scores1 = self.sdp_attn(q_s_1, k_s, v_s, prev=prev[:, prev.size(1)//2:prev.size(1), :],key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            if prev == None:
                output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev,key_padding_mask=key_padding_mask,attn_mask=attn_mask)
                output1, attn_weights1, attn_scores1 = self.sdp_attn(q_s_1, k_s, v_s, prev=prev,key_padding_mask=key_padding_mask,attn_mask=attn_mask)
            else:
                output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev[:, :prev.size(1)//2, :],key_padding_mask=key_padding_mask, attn_mask=attn_mask)
                output1, attn_weights1, attn_scores1 = self.sdp_attn(q_s_1, k_s, v_s, prev=prev[:, prev.size(1)//2:prev.size(1), :],key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1,self.n_heads * self.d_v)  # output: [bs x q_len x n_heads * d_v]
        output1 = output1.transpose(1, 2).contiguous().view(bs, -1,self.n_heads * self.d_v)  # output: [bs x q_len x n_heads * d_v]

        output = self.to_out(output)
        output1 = self.to_out1(output1)
        attn_scores1 = torch.cat((attn_scores, attn_scores1), dim=1)
        attn_weights1 = torch.cat((attn_weights, attn_weights1), dim=1)

        if self.res_attention:
            # return output1, attn_weights1, attn_scores1
            return output,output1, attn_weights1, attn_scores1
        else:
            return output1, attn_weights1


class _ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q: Tensor, k: Tensor, v: Tensor, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None ):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        attn_scores = torch.matmul(q, k) * self.scale  # attn_scores : [bs x n_heads x max_q_len x q_len]
        if prev is not None: attn_scores = attn_scores + prev
        # Attention mask (optional)
        if attn_mask is not None:  # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask
        # Key padding mask (optional)
        if key_padding_mask is not None:  # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)  # output: [bs x n_heads x max_q_len x d_v]
        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights
