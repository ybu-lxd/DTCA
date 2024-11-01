        # to_cat_flow_x = []
        # to_cat_flow_y.append(self.cls_token_flow_y.expand(flow_y.shape[0], -1, -1))
        # to_cat_flow_x.append(self.cls_token_flow_x.expand(flow_x.shape[0], -1, -1))
        # flow_x = torch.cat(to_cat_flow_x + [flow_x], dim=1)        # to_cat_flow_x = []
        # to_cat_flow_y.append(self.cls_token_flow_y.expand(flow_y.shape[0], -1, -1))
        # to_cat_flow_x.append(self.cls_token_flow_x.expand(flow_x.shape[0], -1, -1))
        # flow_x = torch.cat(to_cat_flow_x + [flow_x], dim=1)# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import math
import torch
import torch.nn as nn
import einops
import os
import numpy as np
from timm.models.layers import DropPath
from timm.models.vision_transformer import PatchEmbed, Mlp
from torch.cuda.amp import autocast
from diffusion.model.builder import MODELS
from diffusion.model.utils import auto_grad_checkpoint, to_2tuple
from diffusion.model.nets.PixArt_blocks import Attention, t2i_modulate, CaptionEmbedder, WindowAttention, MultiHeadCrossAttention, T2IFinalLayer, TimestepEmbedder, LabelEmbedder, FinalLayer
from diffusion.utils.logger import get_root_logger



class MyConditiontrsBlock(nn.Module):
    """
    A PixArt block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop_path=0., window_size=0, input_size=None, use_rel_pos=False, **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = WindowAttention(hidden_size, num_heads=num_heads, qkv_bias=True,
                                    input_size=input_size if window_size == 0 else (window_size, window_size),
                                    use_rel_pos=use_rel_pos, **block_kwargs)
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, **block_kwargs)
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # to be compatible with lower version pytorch
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.window_size = window_size
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)
        # self.scale_shift_flow_x = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)
        # self.scale_shift_flow_y = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)
        # self.gamma = nn.Parameter(torch.zeros(1, 1, hidden_size))
        # self.bate = nn.Parameter(torch.zeros(1, 1, hidden_size))
    
    def forward(self, x,  t,clearly_conditon,flow_y):
        B, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)).reshape(B, N, C))
        
        
        x = einops.rearrange(x,"b (c f) l -> (b f) c l",f = 20)
        clearly_conditon = einops.rearrange(clearly_conditon,"b (c f) l -> (b f) c l",f = 4)
        x = x + self.cross_attn(x, clearly_conditon)
        x = einops.rearrange(x,"(b f) c l -> b (c f) l",f = 20)
        
        
        # x = x +  self.cross_attn(x,flow_y)*self.bate
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))
        return x

class MyTransformerBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            # init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 =  nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 =nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x



#############################################################################
#                                 Core PixArt Model                                #
#################################################################################
@MODELS.register_module()
class DTCA(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(self, input_size=32, patch_size=2, in_channels=4, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, class_dropout_prob=0.1, pred_sigma=True, drop_path: float = 0., window_size=0, window_block_indexes=None, use_rel_pos=False, caption_channels=4096, lewei_scale=1.0, config=None,need_flow = True, model_max_length=120, **kwargs):
        if window_block_indexes is None:
            window_block_indexes = []
        super().__init__()
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.lewei_scale = lewei_scale,
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        
        self.condition_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        
        
        
        self.t_embedder = TimestepEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        
        self.num_patches = num_patches
        self.base_size = input_size // self.patch_size
        # Will use fixed sin-cos embedding:
        self.register_buffer("pos_embed", torch.zeros(1, num_patches, hidden_size))
        # self.pos_embed_flow = nn.Parameter(torch.randn(1, 257, hidden_size) * .02)
        self.register_buffer("pos_embed_temporal", self.get_temporal_pos_embed())
        self.register_buffer("pos_embed_condition_temporal", self.get_condition_temporal_pos_embed())
        
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        
        
        
        
        self.condition_mlp = Mlp(in_features=1152, hidden_features=hidden_size, out_features=hidden_size, act_layer=approx_gelu, drop=0)
        #self.y_embedder = CaptionEmbedder(in_channels=caption_channels, hidden_size=hidden_size, uncond_prob=class_dropout_prob, act_layer=approx_gelu, token_num=model_max_length)
        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            MyConditiontrsBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path[i],
                          input_size=(input_size // patch_size, input_size // patch_size),
                          window_size=window_size if i in window_block_indexes else 0,
                          use_rel_pos=use_rel_pos if i in window_block_indexes else False)
            for i in range(depth)
        ])
        self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()
        # self.Pix3dEncoder = PixEncoder()
        if config:
            logger = get_root_logger(os.path.join(config.work_dir, 'train_log.log'))
            logger.warning(f"lewei scale: {self.lewei_scale}, base size: {self.base_size}")
        else:
            print(f'Warning: lewei scale: {self.lewei_scale}, base size: {self.base_size}')
    
    def get_temporal_pos_embed(self):
        pos_embed = get_1d_sincos_pos_embed(
            1152,
            20
        )
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        return pos_embed
    
    def get_condition_temporal_pos_embed(self):
        pos_embed = get_1d_sincos_pos_embed(
            1152,
            4
        )
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        return pos_embed
    
    
    # @autocast()
    # def forward(self, x, timestep,  mask=None, data_info=None, **kwargs):
    #     """
    #     Forward pass of PixArt.
    #     x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
    #     t: (N,) tensor of diffusion timesteps
    #     y: (N, 1, 120, C) tensor of class labels
    #     """
    #     x = x.to(self.dtype)
    #     timestep = timestep.to(self.dtype)
    #     pos_embed = self.pos_embed.to(self.dtype)
    #     self.h, self.w = x.shape[-2]//self.patch_size, x.shape[-1]//self.patch_size
    #     x = self.x_embedder(x)  # (N, T, D), where T = H * W / patch_size ** 2
    #     x = einops.rearrange(x,"(b f) c l -> b (c f) l",f = 16)
    #     # print(pos_embed.shape,x.shape)
    #     pos_embed = einops.repeat(pos_embed,"b c l -> b (c f) l",f = 16)
    #     frame = self.pos_embed_temporal
    #     tem_emb = einops.repeat(frame,"b t l -> b (t a) l", a = self.num_patches)
    #     x = x + pos_embed + tem_emb
    #     timestep = einops.rearrange(timestep,"(t a)  -> t a " ,a =16)[:,0]
    #     t = self.t_embedder(timestep.to(x.dtype))  # (N, D)
    #     t_end = einops.repeat(t,"b l -> (b a) l",a = 16)   
    #     t0 = self.t_block(t)
    #     for block in self.blocks:
    #         x = auto_grad_checkpoint(block, x,  t0)  # (N, T, D) #support grad checkpoint
    #     x = einops.rearrange(x,"b (c f) l -> (b f) c l",f = 16)
    #     x = self.final_layer(x, t_end)  # (N, T, patch_size ** 2 * out_channels)
    #     x = self.unpatchify(x)  # (N, out_channels, H, W)
    #     return x
    
    
    def get_flow_token(self,x):
        for block in self.flow_transblocks:
            x = block(x)
        return x[:,0]
    
    def Dividespacetime(self, x, timestep, condition, flow_x,flow_y,mask=None, data_info=None, **kwargs):
        
        # 时空分离 
        x = x.to(self.dtype)
        
        condition = einops.rearrange(condition,"(b f) c w h -> b c f w h",f=4)
        # condition = einops.rearrange(condition,"b c f w h -> (b f) c w h")
        x = torch.cat([condition,einops.rearrange(x,"(b f) c w h -> b c f w h",f=16)],dim=2)
        
        x = einops.rearrange(x,"b c f w h -> (b f) c w h")
        timestep = timestep.to(self.dtype)
        pos_embed = self.pos_embed.to(self.dtype)
        
        
        
        self.h, self.w = x.shape[-2]//self.patch_size, x.shape[-1]//self.patch_size
        x = self.x_embedder(x)  # (N, T, D), where T = H * W / patch_size ** 2
        x = x + pos_embed 
        
        
        
        # #建立时间平移token
        # flow_x = self.x_embedder(flow_x)
        # flow_y = self.x_embedder(flow_y)
        # to_cat_flow_y = []
        # to_cat_flow_x = []
        # to_cat_flow_y.append(self.cls_token_flow_y.expand(flow_y.shape[0], -1, -1))
        # to_cat_flow_x.append(self.cls_token_flow_x.expand(flow_x.shape[0], -1, -1))
        # flow_x = torch.cat(to_cat_flow_x + [flow_x], dim=1)
        # flow_y = torch.cat(to_cat_flow_y + [flow_y], dim=1)
        # flow_y = flow_y+self.pos_embed_flow
        # flow_x = flow_x+self.pos_embed_flow
        # flow_x_token = self.get_flow_token(flow_x )
        # flow_y_token= self.get_flow_token(flow_y )
        # flow_x_token = self.token_block(flow_x_token)
        # flow_y_token = self.token_block(flow_y_token)
        # flow_x_tokenfor_times = einops.repeat(flow_x_token,"b l -> (b f) l",f= self.num_patches)
        # flow_y_tokenfor_times = einops.repeat(flow_y_token,"b l -> (b f) l",f= self.num_patches)
        # flow_x_tokenfor_space = einops.repeat(flow_x_token,"b l -> (b f) l",f= 20)
        # flow_y_tokenfor_space = einops.repeat(flow_y_token,"b l -> (b f) l",f= 20)
        
        
        
        
        
        ############
        frame = self.pos_embed_temporal
        timestep = einops.rearrange(timestep,"(t a)  -> t a " ,a =16)[:,0]
        t = self.t_embedder(timestep.to(x.dtype))  # (N, D)
        t_end = einops.repeat(t,"b l -> (b a) l",a = 20)
        t0 = self.t_block(t)
        t1 = einops.repeat(t0,"b l -> (b f) l",f= self.num_patches)
        t_ = einops.repeat(t0,"b l -> (b f) l",f= 20)
        
        
        
        # print(flow_y.shape,flow_x.shape,t1.shape,t_.shape)
        # for block in self.blocks:
        #     x = auto_grad_checkpoint(block, x,  t0)  # (N, T, D) #support grad checkpoint
        # x = einops.rearrange(x,"b (c f) l -> (b f) c l",f  =16)
        flow_x_tokenfor_space = None
        flow_y_tokenfor_space = None
        flow_x_tokenfor_times = None
        flow_y_tokenfor_times = None
        for i in range(0, len(self.blocks), 2):
            ss = self.blocks[i:i+2]
            sss = ss[0]
            tt = ss[1]
            
            x = auto_grad_checkpoint(sss, x,  t_,flow_x_tokenfor_space,flow_y_tokenfor_space) 
            x = einops.rearrange(x,"(b f) c l -> (b c) f l",f = 20)
            
            if i== 0:
                # print(x.shape,tem_emb.shape)
                x = frame+x
            
            x = auto_grad_checkpoint(tt, x,  t1,flow_x_tokenfor_times,flow_y_tokenfor_times)
            x = einops.rearrange(x,"(b c) f l -> (b f) c  l" ,c = self.num_patches)
            
        # x = einops.rearrange(x,"b (c f) l -> (b f) c l",f = 16)
       
        x = self.final_layer(x, t_end)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        x = einops.rearrange(x,"(b f) c w h -> b c f w h",f = 20)[:,:,4:,...]
        x = einops.rearrange(x,"b c f w h -> (b f) c w h")
        return x

    def jion_sapce_times(self, x, timestep, condition, condition_high_pix,flow_x,flow_y,mask=None, data_info=None, **kwargs):
        #时空联合建模
        condition_shape = condition.shape
        x = x.to(self.dtype)
        pos_embed = self.pos_embed.to(self.dtype)
        # 准备像素空间以及时空位置信息,为交叉注意力做准备
        # condition_high_pix = condition_high_pix[:,0,...]
        # condition_high_pix = einops.repeat(condition_high_pix,"b w h -> b c w h",c=4)
        # condition_high_pix = self.Pix3dEncoder(condition_high_pix)
        # condition_high_pix = self.condition_embedder(condition_high_pix)
        # condition_high_pix = condition_high_pix+pos_embed
        # condition_high_pix = einops.rearrange(condition_high_pix,"(b f) c l -> (b c) f l",f = 4)+self.pos_embed_condition_temporal.to(self.dtype)
        # condition_high_pix = einops.rearrange(condition_high_pix,"(b c) f l -> b (c f) l",c = self.num_patches)
        # #
        # print(condition_high_pix.shape)
        
        # # 准备隐空间以及时空位置信息,为交叉注意力做准备
        
        #shift=4
        # print(condition.shape)
        condition_clearly_latent = condition.clone()
        condition_clearly_latent = self.condition_embedder(condition_clearly_latent)
        condition_clearly_latent = condition_clearly_latent+pos_embed
        condition_clearly_latent = einops.rearrange(condition_clearly_latent,"(b f) c l -> (b c) f l",f = 4)+self.pos_embed_condition_temporal.to(self.dtype)
        # print(condition_clearly_latent.shape)
        
        
        # 打乱位移
        # condition_clearly_latent = einops.rearrange(condition_clearly_latent,"(b c) f l -> b c f l",c = self.num_patches)
        
        # selected_data_1 = condition_clearly_latent[:, ::4, ...]
        # selected_data_2 = condition_clearly_latent[:, 1::4, ...]
        # selected_data_3 = condition_clearly_latent[:, 2::4, ...]
        # selected_data_4 = condition_clearly_latent[:, 3::4, ...]

        # selected_data = torch.cat([selected_data_1, selected_data_2, selected_data_3, selected_data_4], dim=1)
        
        # condition_clearly_latent = einops.rearrange(selected_data,"b (a c) f l -> (b a) (c f) l",a=4)
        
        # print(condition_clearly_latent.shape)
        ###################
        
        # 
        # 顺序位移
        condition_clearly_latent = einops.rearrange(condition_clearly_latent,"(b c) f l -> b (c f) l",b = condition_shape[0]) # shift 4
        
        # print(condition_clearly_latent.shape)
        
        # condition_clearly_latent = einops.rearrange(condition_clearly_latent,"(b c) f l -> b (c f) l",c = self.num_patches)# shift 0
        
        #condition_clearly_latent = einops.rearrange(condition_clearly_latent,"(b c) f l -> b (c f) l",b = condition_shape[0]//4 * 8)# shift 8
        
        #condition_clearly_latent = einops.rearrange(condition_clearly_latent,"(b c) f l -> b (c f) l",b = condition_shape[0]//4 * 16)# shift 16
        # print(condition_clearly_latent.shape)
        # print(condition_clearly_latent.shape)
        # print(condition_clearly_latent.shape)
        # #

        
        
        
        
        
        
        
        
        
        
        
        condition_clearly_latent = self.condition_mlp(condition_clearly_latent)
        # condition_clearly_latent =self.condition_mlp(condition_clearly_latent)
        
        
        condition = einops.rearrange(condition,"(b f) c w h -> b c f w h",f=4)
        # condition = self.oma_x(torch.concat((flow_x,flow_y),dim=1),condition)
        x = torch.cat([condition,einops.rearrange(x,"(b f) c w h -> b c f w h",f=16)],dim=2)
        t = self.t_embedder(timestep.to(x.dtype))  # (N, D)
        x_batch = x.shape[0]
        
        x = einops.rearrange(x,"b c f w h -> (b f) c w h")
        timestep = timestep.to(self.dtype)
        
        
        self.h, self.w = x.shape[-2]//self.patch_size, x.shape[-1]//self.patch_size
        x = self.x_embedder(x)  # (N, T, D), where T = H * W / patch_size ** 2
        x = x + pos_embed 
        frame = self.pos_embed_temporal
        # x = einops.rearrange(x,"(b f) c l -> (b c) f l",f = 20)
        
        # flow_x_token,flow_y_token = self.oma_x(torch.concat((flow_x,flow_y),dim=0))
        # flow_y_token  = self.oma_y(flow_y)
        
        
        # t_end = einops.repeat(t,"b l -> (b a) l",a = 20)
        
        ############
        
        timestep = einops.rearrange(timestep,"(t a)  -> t a " ,a =16)[:,0]
        t = self.t_embedder(timestep.to(x.dtype))  # (N, D)
        tt = t
        # t_end = einops.repeat(t,"b l -> (b a) l",a = 20)
        t0 = self.t_block(t)
        x = einops.rearrange(x,"(b f ) c l -> (b c) f l",f = 20)+frame
        
        
        
        
        # clearly_condition = x[:,:4,...].clone()
        # clearly_condition = einops.rearrange(clearly_condition," (b c) f l -> b (c f) l",b = condition_shape[0])
        
        
        
        x = einops.rearrange(x,"(b c) f l -> b (c f) l",b = x_batch)
        for block in self.blocks:
            x = block(x,t0,condition_clearly_latent,None)
        x = self.final_layer(x, tt)  # (N, T, patch_size ** 2 * out_channels)
        x = einops.rearrange(x,"b (c f) l -> (b f) c l",f = 20)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        x = einops.rearrange(x,"(b f) c w h -> b c f w h",f = 20)[:,:,4:,...]
        x = einops.rearrange(x,"b c f w h -> (b f) c w h")
        return x 
        
    def jion_sapce_times_space(self, x, timestep, condition, flow_x,flow_y,mask=None, data_info=None, **kwargs):
        #时空联合建模
        x = x.to(self.dtype)
        condition = einops.rearrange(condition,"(b f) c w h -> b c f w h",f=4)
        x = torch.cat([condition,einops.rearrange(x,"(b f) c w h -> b c f w h",f=16)],dim=2)
        t = self.t_embedder(timestep.to(x.dtype))  # (N, D)
        x_batch = x.shape[0]
        
        x = einops.rearrange(x,"b c f w h -> (b f) c w h")
        timestep = timestep.to(self.dtype)
        pos_embed = self.pos_embed.to(self.dtype)
        
        self.h, self.w = x.shape[-2]//self.patch_size, x.shape[-1]//self.patch_size
        x = self.x_embedder(x)  # (N, T, D), where T = H * W / patch_size ** 2
        x = x + pos_embed 
        frame = self.pos_embed_temporal
        # x = einops.rearrange(x,"(b f) c l -> (b c) f l",f = 20)
        
        flow_x_tokenfor_space = None
        flow_y_tokenfor_space = None
        flow_x_tokenfor_times = None
        flow_y_tokenfor_times = None
        
        
        # t_end = einops.repeat(t,"b l -> (b a) l",a = 20)
        
        ############
        
        timestep = einops.rearrange(timestep,"(t a)  -> t a " ,a =16)[:,0]
        t = self.t_embedder(timestep.to(x.dtype))  # (N, D)
        tt = t
        # t_end = einops.repeat(t,"b l -> (b a) l",a = 20)
        t0 = self.t_block(t)
        t_ = einops.repeat(t0,"b l -> (b f) l",f= 20)
        x = einops.rearrange(x,"(b f ) c l -> (b c) f l",f = 20)+frame
        x = einops.rearrange(x,"(b c) f l -> b (c f) l",b = x_batch)
        
        
        
        for i in range(0, len(self.blocks), 2):
            ss = self.blocks[i:i+2]
            sapce_times_block = ss[0]
            sapce = ss[1]
            x = auto_grad_checkpoint(sapce_times_block, x,  t0,flow_x_tokenfor_space,flow_y_tokenfor_space) 
            x = einops.rearrange(x,"b (c f) l -> (b f) c l",f = 20)
            x = auto_grad_checkpoint(sapce, x,  t_,flow_x_tokenfor_times,flow_y_tokenfor_times)
            x = einops.rearrange(x,"(b f) c l -> b (c f)  l" ,f =20)
            
        x = self.final_layer(x, tt)  # (N, T, patch_size ** 2 * out_channels)
        x = einops.rearrange(x,"b (c f) l -> (b f) c l",f = 20)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        x = einops.rearrange(x,"(b f) c w h -> b c f w h",f = 20)[:,:,4:,...]
        x = einops.rearrange(x,"b c f w h -> (b f) c w h")
        
        return x 
        
        
        
        
        
        
        
        
        
        
        
        
    
    @autocast()
    def forward(self, x, timestep, condition, condition_high_pix,flow_x,flow_y,mask=None, data_info=None, **kwargs):
        # x = self.Dividespacetime(x=x,timestep=timestep,condition=condition,flow_x=flow_x,flow_y=flow_y)
        x = self.jion_sapce_times(x=x,timestep=timestep,condition=condition,condition_high_pix= condition_high_pix,flow_x=flow_x,flow_y=flow_y)
        # x = self.Dividespacetime(x=x,timestep=timestep,condition=condition,flow_x=flow_x,flow_y=flow_y)
        return x








    def forward_with_sample(self, x, timestep, mask=None, **kwargs):
        return self.forward(x,timestep, mask=None, **kwargs)


    def forward_with_dpmsolver(self, x, timestep, y, mask=None, **kwargs):
        """
        dpm solver donnot need variance prediction
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        model_out = self.forward(x, timestep, y, mask)
        return model_out.chunk(2, dim=1)[0]

    def forward_with_cfg(self, x, timestep, y, cfg_scale, mask=None, **kwargs):
        """
        Forward pass of PixArt, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, timestep, y, mask, kwargs)
        model_out = model_out['x'] if isinstance(model_out, dict) else model_out
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    def initialize_weights(self):
        # Initialize transformer layers:
        # 已经拥有预训练权重 不再需要初始化
        
        # def _basic_init(module):
        #     if isinstance(module, nn.Linear):
        #         torch.nn.init.xavier_uniform_(module.weight)
        #         if module.bias is not None:
        #             nn.init.constant_(module.bias, 0)

        # self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int((self.x_embedder.num_patches)** 0.5), lewei_scale=self.lewei_scale, base_size=self.base_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # 加载与训练条件权重打patch
        # self.condition_embedder.load_state_dict(torch.load("x_embedder_weights.pth"))
        
        # print(self.pos_embed_flow.shape)
        # #初始化光流位置编码
        # pos_embed_flow = get_2d_sincos_pos_embed(self.pos_embed_flow.shape[-1], int((258)** 0.5), lewei_scale=self.lewei_scale, base_size=self.base_size)
        # print(pos_embed_flow.shape)
        # self.pos_embed_flow.data.copy_(torch.from_numpy(pos_embed_flow).float().unsqueeze(0))
        
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)

        # Initialize caption embedding MLP:
        # nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        # nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)
        
        # nn.init.normal_(self.cls_token_flow_y, std=1e-6)
        # nn.init.normal_(self.cls_token_flow_x, std=1e-6)
        
        # for flow_blocks in self.flow_transblocks:
        #     nn.init.constant_(flow_blocks.attn.proj.weight,0)
            # nn.init.constant_(flow_blocks.attn.bias.weight,0)
            
            # nn.init.normal_(flow_blocks.mlp[0].weight, std=0.02)
            # nn.init.normal_(flow_blocks.mlp[2].weight, std=0.02)
            
        
        
        # Zero-out adaLN modulation layers in PixArt blocks:
        # for block in self.blocks:
        #     nn.init.constant_(block.cross_attn.proj.weight, 0)
        #     nn.init.constant_(block.cross_attn.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    @property
    def dtype(self):
        return next(self.parameters()).dtype


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, lewei_scale=1.0, base_size=16):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_size = to_2tuple(grid_size)
    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0]/base_size) / lewei_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1]/base_size) / lewei_scale
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    return np.concatenate([emb_sin, emb_cos], axis=1)

def get_1d_sincos_pos_embed(embed_dim, length, scale=1.0):
    pos = np.arange(0, length)[..., None] / scale
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)

#################################################################################
#                                   PixArt Configs                                  #
#################################################################################
@MODELS.register_module()
def DTCA_XL_2(**kwargs):
    return DTCA(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)




class OMA(nn.Module):
    """Some Information about OEA"""
    def __init__(self,hidden_size=1152,num_heads=16,mlp_ratio=4.0,depth = 4,drop_path = 0.0):
        super(OMA, self).__init__()
        
        
        
        self.x_embedder = PatchEmbed(32, 2, 4, hidden_size, bias=True)
        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.flow_transblocks = nn.ModuleList([MyTransformerBlock(hidden_size,num_heads,mlp_ratio=mlp_ratio,drop_path=drop_path[i]) 
                                               for i in range(depth)
                                               ])
        
        self.pos_embed_flow = nn.Parameter(torch.randn(1, 257, hidden_size) * .02)
        self.cls_token_flow_x = nn.Parameter(torch.zeros(1, 1, hidden_size))
        # self.token_block = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(hidden_size,  hidden_size, bias=True)
        # )
        
        
        self.init_weight()
    def init_weight(self):
        x_embedder_weights = torch.load("x_embedder_weights.pth",map_location="cpu")
        self.x_embedder.proj.weight.data = x_embedder_weights['proj.weight']
        self.x_embedder.proj.bias.data = x_embedder_weights['proj.bias']
        
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.x_embedder(x)
        to_cat_flow_x = []
        to_cat_flow_x.append(self.cls_token_flow_x.expand(x.shape[0], -1, -1))
        x = torch.cat(to_cat_flow_x + [x], dim=1)
        # 添加位置信息
        x = x+self.pos_embed_flow
        
        for block in self.flow_transblocks:
            x = block(x)
        # x = self.token_block(x)
        x = x[:,0,...]
        x = torch.unsqueeze(x,dim=1)
        flow_x,flow_y = x.chunk(2,dim=0)
        return flow_x,flow_y
    

 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
