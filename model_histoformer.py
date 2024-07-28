import os
import cv2
# import time
import numpy as np
# import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
# from PIL import Image

from timm.models.layers import DropPath, trunc_normal_
# from einops import rearrange, repeat
# from einops.layers.torch import Rearrange
from torch import einsum
from torch.nn import init
from torchvision import models 
##############Intra_SA#######################
class Intra_Attention(nn.Module):
    def __init__(self, head_num):
        super(Intra_Attention, self).__init__()
        self.num_attention_heads = head_num
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        B, N, C = x.size()
        # print('x.size()',x.size(),'self.num_attention_heads',self.num_attention_heads)
        attention_head_size = int(C / self.num_attention_heads)
        # print('attention_head_size',attention_head_size)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, attention_head_size)
        x = x.view(*new_x_shape)
        # print('x',x.shape)
        x = x.permute(0, 2, 3, 1).contiguous()
        x_5 = torch.unsqueeze(x, -1)###########intra key point 37 1 32 256 1
        return x_5

    def forward(self, query_layer, key_layer, value_layer):
        # print('query_layer',query_layer.size())
        B, N, C = query_layer.size() #x.size() torch.Size([37, 256, 32]) self.num_attention_heads 1
        query_layer = self.transpose_for_scores(query_layer)
        # print('query_layer',query_layer.shape)#query_layer torch.Size([37, 1, 32, 256,1])
        key_layer = self.transpose_for_scores(key_layer)#torch.Size([37, 1, 32, 256, 1])
        # print('key_layer',key_layer.shape)
        value_layer = self.transpose_for_scores(value_layer)#torch.Size([37, 1, 32, 256, 1])
        # print('value_layer',value_layer.shape)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  #[37, 1, 32, 256,256]
        # print('attention_scores',attention_scores.shape)
        _, _, d, _ ,_ = query_layer.size()
        attention_scores = attention_scores / math.sqrt(d)
        attention_probs = self.softmax(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer) #[37, 1, 32, 256,1]
        # print('context_layer',context_layer.shape)
        #####key point
        context_layer = torch.squeeze(context_layer, -1) #[37, 1, 32, 256]
        # print('context_layersq',context_layer.shape)
        context_layer = context_layer.permute(0, 3, 1, 2).contiguous()
        # print('context_layer2',context_layer.shape)
        new_context_layer_shape = context_layer.size()[:-2] + (C,)
        # print('new_context_layer_shape',new_context_layer_shape)
        attention_out = context_layer.view(*new_context_layer_shape)
        # print('attention_out',attention_out.shape)
        return attention_out
class MlpINTRA(nn.Module):
    def __init__(self, hidden_size):
        super(MlpINTRA, self).__init__()
        self.fc1 = nn.Linear(hidden_size, 4*hidden_size)
        self.fc2 = nn.Linear(4*hidden_size, hidden_size)
        self.act_fn = torch.nn.functional.gelu
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        return x



class Intra_SA(nn.Module):
    def __init__(self, dim, head_num):
        super(Intra_SA, self).__init__()
        self.hidden_size = dim ######dim/2
        self.head_num = head_num
        self.attention_norm = nn.LayerNorm(dim)
        self.conv_input = nn.Conv1d(dim, dim, kernel_size=1, padding=0)
        self.qkv_local_h = nn.Linear(self.hidden_size, self.hidden_size * 3)  # qkv_h
        self.fuse_out = nn.Conv1d(dim, dim, kernel_size=1, padding=0)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = MlpINTRA(dim)
        self.attn = Intra_Attention(head_num=self.head_num)
    def forward(self, x):
        x = x.permute(0,2,1)
        sh = x
        B, C, H = x.size()
        x_input = self.conv_input(x)###need like restormer stripformer
        feature_h = (x_input).permute(0, 2, 1).contiguous() #BHC
        qkv_h = torch.chunk(self.qkv_local_h(feature_h), 3, dim=2)
        q_h, k_h, v_h = qkv_h[0], qkv_h[1], qkv_h[2]
        attention_output_h = self.attn(q_h, k_h, v_h)
        attention_output_h = attention_output_h.view(B, H, C).permute(0, 2, 1).contiguous()
        attn_out = attention_output_h
        x = attn_out + sh
        x = x.view(B, C, H).permute(0, 2, 1).contiguous()
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)##MLP
        x = x + h
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(B, C, H)
        return x

########### Downsample/Upsample #############
class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        x = x.transpose(1, 2)# (B, C, H, W)
        out = self.conv(x).transpose(1,2)# B H*W C
        return out

class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel
        
    def forward(self, x):
        x = x.transpose(1, 2)# (B, C, H, W)
        out = self.deconv(x).transpose(1,2)# B H*W C
        return out


########### Input/Output Projection #############
class InputProjection(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, norm_layer=None,act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
            act_layer(inplace=True)
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        # print('InputProjection',x.shape)
        x = self.proj(x).transpose(1, 2)
        # print(x.shape)
        if self.norm is not None:
            x = self.norm(x)
        return x
    
class OutputProjection(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None,act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        x = x.transpose(1, 2)#(B, C, H, W)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


########### Multi-head Self-Attention #############
class LinearProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., bias=True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias = bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = bias)
        self.dim = dim
        self.inner_dim = inner_dim
        self.conv_input = nn.Conv1d(dim, dim, kernel_size=1, padding=0)

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        x = x.permute(0,2,1).contiguous()##bcn
        
        x_input = self.conv_input(x)###need like restormer stripformer
        x_input = x_input.permute(0,2,1).contiguous()##b n c
        attn_kv = x_input if attn_kv is None else attn_kv
        q_inter = self.to_q(x_input).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 4, 1)

        kv_inter = self.to_kv(attn_kv).reshape(B_, N, 2, self.heads, C // self.heads).permute(2, 0, 3, 4, 1)
        
        
        q_inter = q_inter[0]
        k_inter, v_inter = kv_inter[0], kv_inter[1] 
        
        return q_inter, k_inter, v_inter

class Attention(nn.Module):
    def __init__(self, dim, num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,se_layer=False):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if token_projection =='linear':
            self.qkv = LinearProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        
        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.se_layer = SELayer(dim) if se_layer else nn.Identity()

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q_inter, k_inter, v_inter = self.qkv(x,attn_kv)

        scale_inter = N** -0.5
        q_inter = q_inter * scale_inter
        attn_inter = (q_inter @ k_inter.transpose(-2, -1))
        attn_inter = self.softmax(attn_inter)
        attn_inter = self.attn_drop(attn_inter)

        x_inter = (attn_inter @ v_inter).transpose(1, 2).reshape(B_, N, C)
        x_inter = self.proj(x_inter)
        x_inter = self.se_layer(x_inter)
        x_inter = self.proj_drop(x_inter)

        return x_inter


########### Feed-Forward Network #############
class TwoDCFF(nn.Module): #2D
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU,drop = 0.):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                act_layer())
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_dim,hidden_dim,groups=hidden_dim,kernel_size=3,stride=1,padding=1),
                        act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim
    
    def forward(self, x):        
        
        x = self.linear1(x) 
        bs, hw, c = x.size()
        x = x.reshape(bs,16,-1,c)
        x = x.permute(0,3,1,2)
        
        x = self.dwconv(x)
        x = x.permute(0,2,3,1)
        x = x.reshape(bs,-1,c)

        x = self.linear2(x)

        return x


########### Transformer #############
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,token_projection='linear',token_mlp='TwoDCFF',se_layer=False):        
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp
        
        # self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            token_projection=token_projection)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,act_layer=act_layer, drop=drop) if token_mlp=='ffn' else TwoDCFF(dim,mlp_hidden_dim,act_layer=act_layer, drop=drop)
        # self.intra_block = Intra_SA(dim, num_heads)
    def forward(self, x, mask=None):
        # shortcut = x #  B N C 10 256 32
        # B_, N, C = x.shape
        # x = self.norm1(x)#BNC
        # hx = self.intra_block(x)
        # hx = hx.permute(0, 2, 1)
        # x = shortcut + self.drop_path(hx)
        
        shortcut_inter = x
        x = self.norm2(x)
        inter = self.attn(x, mask=None)
        x = shortcut_inter + self.drop_path(inter)

        x = x + self.drop_path(self.mlp(self.norm3(x)))
        
        return x


########### Basic layer of Histoformer ################
class BasicUformerLayer(nn.Module):
    def __init__(self, dim, output_dim, depth, num_heads,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 token_projection='linear',token_mlp='TwoDCFF',se_layer=False):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # build blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=dim, num_heads=num_heads,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
            for i in range(depth)])
         

    def forward(self, x, mask=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x,mask)
        return x


########### Histoformer ################
class Histoformer(nn.Module): #[2, 2, 2, 2, 2, 2, 2, 2, 2] [1, 2, 8, 8, 8, 8, 8, 2, 1]
    def __init__(self, in_chans=3, embed_dim=32,
                 depths=[2, 2, 2, 2, 2, 2, 2, 2, 2], num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, token_projection='linear', token_mlp='TwoDCFF', se_layer=False,
                 dowsample=Downsample, upsample=Upsample, **kwargs):
        super().__init__()

        self.num_enc_layers = len(depths)//2
        self.num_dec_layers = len(depths)//2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))] 
        conv_dpr = [drop_path_rate]*depths[4]
        dec_dpr = enc_dpr[::-1]

        # build layers

        # Input/Output
        self.input_projection = InputProjection(in_channel=in_chans, out_channel=embed_dim, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.output_projection = OutputProjection(in_channel=2*embed_dim, out_channel=in_chans, kernel_size=3, stride=1)
        
        # Encoder
        self.encoderlayer_0 = BasicUformerLayer(dim=embed_dim,
                            output_dim=embed_dim,
                            depth=depths[0],
                            num_heads=num_heads[0],
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.dowsample_0 = dowsample(embed_dim, embed_dim*2)
        self.encoderlayer_1 = BasicUformerLayer(dim=embed_dim*2,
                            output_dim=embed_dim*2,
                            depth=depths[1],
                            num_heads=num_heads[1],
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.dowsample_1 = dowsample(embed_dim*2, embed_dim*4)
        self.encoderlayer_2 = BasicUformerLayer(dim=embed_dim*4,
                            output_dim=embed_dim*4,
                            depth=depths[2],
                            num_heads=num_heads[2],
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.dowsample_2 = dowsample(embed_dim*4, embed_dim*8)
        self.encoderlayer_3 = BasicUformerLayer(dim=embed_dim*8,
                            output_dim=embed_dim*8,
                            depth=depths[3],
                            num_heads=num_heads[3],
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:3]):sum(depths[:4])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.dowsample_3 = dowsample(embed_dim*8, embed_dim*16)
        
        # Bottleneck
        self.conv = BasicUformerLayer(dim=embed_dim*16,
                            output_dim=embed_dim*16,
                            depth=depths[4],
                            num_heads=num_heads[4],
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=conv_dpr,
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)

        # Decoder
        self.upsample_0 = upsample(embed_dim*16, embed_dim*8)
        self.decoderlayer_0 = BasicUformerLayer(dim=embed_dim*16,
                            output_dim=embed_dim*16,
                            depth=depths[5],
                            num_heads=num_heads[5],
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[:depths[5]],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.upsample_1 = upsample(embed_dim*16, embed_dim*4)
        self.decoderlayer_1 = BasicUformerLayer(dim=embed_dim*8,
                            output_dim=embed_dim*8,
                            depth=depths[6],
                            num_heads=num_heads[6],
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[sum(depths[5:6]):sum(depths[5:7])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.upsample_2 = upsample(embed_dim*8, embed_dim*2)
        self.decoderlayer_2 = BasicUformerLayer(dim=embed_dim*4,
                            output_dim=embed_dim*4,
                            depth=depths[7],
                            num_heads=num_heads[7],
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[sum(depths[5:7]):sum(depths[5:8])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.upsample_3 = upsample(embed_dim*4, embed_dim)
        self.decoderlayer_3 = BasicUformerLayer(dim=embed_dim*2,
                            output_dim=embed_dim*2,
                            depth=depths[8],
                            num_heads=num_heads[8],
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[sum(depths[5:8]):sum(depths[5:9])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)

        self.apply(self._init_weights)
        self.softmax = nn.Softmax(2)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask=None):
        # Input Projection
        y = self.input_projection(x)
        y = self.pos_drop(y)
        # print('y',y.size())
        #Encoder
        conv0 = self.encoderlayer_0(y,mask=mask)
        # print('conv0',conv0.size())
        pool0 = self.dowsample_0(conv0)
        # print('pool0',pool0.size())
        conv1 = self.encoderlayer_1(pool0,mask=mask)
        # print('conv1',conv1.size())
        pool1 = self.dowsample_1(conv1)
        # print('pool1',pool1.size())
        conv2 = self.encoderlayer_2(pool1,mask=mask)
        # print('conv2',conv2.size())
        pool2 = self.dowsample_2(conv2)
        # print('pool2',pool2.size())
        conv3 = self.encoderlayer_3(pool2,mask=mask)
        pool3 = self.dowsample_3(conv3)
        
        # Bottleneck
        conv4 = self.conv(pool3, mask=mask)
        # print('conv4',conv4.size())

        #Decoder
        up0 = self.upsample_0(conv4)
        deconv0 = torch.cat([up0,conv3],-1)
        deconv0 = self.decoderlayer_0(deconv0,mask=mask)
        # print('deconv0',deconv0.size())
        
        up1 = self.upsample_1(deconv0)
        deconv1 = torch.cat([up1,conv2],-1)
        deconv1 = self.decoderlayer_1(deconv1,mask=mask)
        # print('deconv1',deconv1.size())

        up2 = self.upsample_2(deconv1)
        deconv2 = torch.cat([up2,conv1],-1)
        deconv2 = self.decoderlayer_2(deconv2,mask=mask)
        # print('deconv2',deconv2.size())

        up3 = self.upsample_3(deconv2)
        deconv3 = torch.cat([up3,conv0],-1)
        deconv3 = self.decoderlayer_3(deconv3,mask=mask)
        
        # Output Projection
        y = self.output_projection(deconv3)
        x_y = self.softmax(x+y)   
        # print('x_y',x_y.size())     
        return  x_y
        



# class SELayer(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(SELayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):  # x: [B, N, C]
#         x = torch.transpose(x, 1, 2)  # [B, C, N]
#         b, c, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1)
#         x = x * y.expand_as(x)
#         x = torch.transpose(x, 1, 2)  # [B, N, C]
#         return x