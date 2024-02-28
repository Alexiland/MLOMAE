from functools import partial
import timm
import torch
import torch.nn as nn

import timm.models.vision_transformer
import cifar_mae_model as mae_model
from timm.models.vision_transformer import PatchEmbed, Block


class FinetuneVisionTransformer(nn.Module):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, num_classes = 10, drop_rate = 0.1, embed_dim = 768, global_pool=False, **kwargs):
        super(FinetuneVisionTransformer, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.norm_layer=partial(nn.LayerNorm, eps=1e-6)
#        self.norm_layer=nn.LayerNorm
        self.embed_dim = embed_dim
        self.fc_norm = self.norm_layer(self.embed_dim)
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.global_pool = global_pool
        # if self.global_pool:
            # self.norm_layer = kwargs['norm_layer']
            # self.embed_dim = kwargs['embed_dim']
            # self.fc_norm = self.norm_layer(self.embed_dim)
            # self.head_drop = nn.Dropout(drop_rate)
            # self.head = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
            # del self.norm  # remove the original norm

    def forward(self, x, mae_model=None, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)
        
        
class FinetuneVisionTransformer_c100(nn.Module):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, num_classes = 100, drop_rate = 0.1, embed_dim = 768, global_pool=False, **kwargs):
        super(FinetuneVisionTransformer_c100, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.norm_layer=partial(nn.LayerNorm, eps=1e-6)
#        self.norm_layer=nn.LayerNorm
        self.embed_dim = embed_dim
        self.fc_norm = self.norm_layer(self.embed_dim)
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.global_pool = global_pool
        # if self.global_pool:
            # self.norm_layer = kwargs['norm_layer']
            # self.embed_dim = kwargs['embed_dim']
            # self.fc_norm = self.norm_layer(self.embed_dim)
            # self.head_drop = nn.Dropout(drop_rate)
            # self.head = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
            # del self.norm  # remove the original norm

    def forward(self, x, mae_model=None, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)



# class ViTClassifier(nn.Module):
#     def __init__(self, encoder, num_classes=10):
#         super(ViTClassifier, self).__init__()
#         self.encoder = encoder
#         self.num_classes = num_classes
#         drop_rate = 0.1
#         self.norm_layer=partial(nn.LayerNorm, eps=1e-6)
# #        self.norm_layer=nn.LayerNorm
# #        self.embed_dim = 768
#         self.fc_norm = self.norm_layer(encoder.embedding.out_features)
#         self.classifier = nn.Linear(encoder.embedding.out_features, num_classes)
        
#     def forward(self, x):
#         x = self.encoder(x)
#         # Assuming the shape is [batch_size, seq_length, features]
#         # Take the first token (often called the [CLS] token in some transformer models) for classification
#         x = x[:, 0, :]
#         x = self.fc_norm(x)
#         x = self.head_drop(x)
#         return self.classifier(x)




class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        img_size=32
        patch_size=2
        in_chans=3
        embed_dim=768
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim),
                                      requires_grad=True)  # fixed sin-cos embedding
        pos_embed = mae_model.get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            # del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
    
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
    
        for blk in self.blocks:
            x = blk(x)

        
        if self.global_pool:
            print(x.size())
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            print(x.size())
            outcome = self.fc_norm(x)
            print(outcome.size())
        else:
            x = self.norm(x)
            outcome = x[:, 0]
    
        return outcome

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        num_classes=1000, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def cifar10_vit_base_patch2(**kwargs):
    model = VisionTransformer(
        num_classes=10, patch_size=2, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0.1, **kwargs)
    return model

def cifar100_vit_base_patch2(**kwargs):
    model = VisionTransformer(
        num_classes=100, patch_size=2, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0.1, **kwargs)
    return model
    
def cifar10_vit_base_patch16(**kwargs):
    model = VisionTransformer(
        num_classes=10, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0.1, **kwargs)
    return model