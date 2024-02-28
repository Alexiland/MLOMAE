from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer


class FinetuneVisionTransformer(nn.Module):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, num_classes = 1000, drop_rate = 0.1, embed_dim = 768, global_pool=False, **kwargs):
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



class ViTClassifier(nn.Module):
    def __init__(self, encoder, num_classes=10):
        super(ViTClassifier, self).__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        drop_rate = 0.1
        self.norm_layer=partial(nn.LayerNorm, eps=1e-6)
#        self.norm_layer=nn.LayerNorm
#        self.embed_dim = 768
        self.fc_norm = self.norm_layer(encoder.embedding.out_features)
        self.classifier = nn.Linear(encoder.embedding.out_features, num_classes)
        
    def forward(self, x):
        x = self.encoder(x)
        # Assuming the shape is [batch_size, seq_length, features]
        # Take the first token (often called the [CLS] token in some transformer models) for classification
        x = x[:, 0, :]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return self.classifier(x)





class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=True, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        # self.num_classes = 10
        # drop_rate = 0
        self.global_pool = global_pool
        if self.global_pool:
            self.norm_layer = kwargs['norm_layer']
            self.embed_dim = kwargs['embed_dim']
            self.fc_norm = self.norm_layer(self.embed_dim)
            # self.head_drop = nn.Dropout(drop_rate)
            # self.head = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

            del self.norm  # remove the original norm
            
        self.initialize_norm_params()

    def initialize_norm_params(self):
        for block in self.blocks:
            norm_pre = block.norm1
            nn.init.ones_(norm_pre.weight)
            nn.init.zeros_(norm_pre.bias)

    # def forward_features(self, x, mae_model):
    #     B = x.shape[0]
    #     x = self.patch_embed(x)
    
    #     cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    #     x = torch.cat((cls_tokens, x), dim=1)
    #     x = x + self.pos_embed
    #     x = self.pos_drop(x)
    
    #     for blk in self.blocks:
    #         x = blk(x)
    
    #     if self.global_pool:
    #         x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
    #         outcome = self.fc_norm(x)
    #     else:
    #         x = self.norm(x)
    #         outcome = x[:, 0]
    
    #     return outcome

    def forward_features(self, x, mae_model):
        B = x.shape[0]
        x = mae_model.patch_embed(x)

        cls_tokens = mae_model.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + mae_model.pos_embed
        x = self.pos_drop(x)

        for blk in mae_model.blocks:
            x = blk(x)

        if self.global_pool:
            outcome = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
            # x = self.norm(x)
            # outcome = x[:, 0]
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        # print(outcome.shape)
        return outcome


    # def forward_head(self, x, pre_logits: bool = False):
    #     if self.global_pool:
    #         x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
    #     x = self.fc_norm(x)
    #     x = self.head_drop(x)
    #     return x if pre_logits else self.head(x)
        
    def forward(self, x, mae_model=None):
        # x = self.forward_features(x, mae_model)
        x = self.forward_head(x)
        return x


#def vit_base_patch16(**kwargs):
#    model = VisionTransformer(
#        num_classes=1000, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#    return model
def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
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
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
