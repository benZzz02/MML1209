# model.py
from collections import OrderedDict
import os
import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from config import cfg
from log import logger

from peft import LoraConfig, get_peft_model, TaskType

_tokenizer = _Tokenizer()


# ------------------------------
# Helpers
# ------------------------------
def _cfg(name: str, default):
    return getattr(cfg, name, default)


# ------------------------------
# LoRA injection
# ------------------------------
def add_lora_to_clip(clip_model: nn.Module):
    """
    根据 cfg 配置动态创建 LoRA 适配器，并注入到 CLIP 编码器。
    返回：LoRA 包装后的 clip_model 模块。
    """
    if not hasattr(cfg, "LORA") or not cfg.LORA.ENABLED:
        logger.info(
            "LoRA is disabled or LORA config not found. Freezing CLIP encoders for default Prompt Tuning."
        )
        for param in clip_model.parameters():
            param.requires_grad_(False)
        return clip_model

    logger.info(f"LoRA enabled. Injecting LoRA adapters with r={cfg.LORA.R}, alpha={cfg.LORA.ALPHA}.")

    # 1) freeze base
    for param in clip_model.parameters():
        param.requires_grad_(False)

    # 2) lora config
    lora_config = LoraConfig(
        r=cfg.LORA.R,
        lora_alpha=cfg.LORA.ALPHA,
        target_modules=cfg.LORA.TARGET_MODULES,
        lora_dropout=cfg.LORA.DROPOUT,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    # 3) inject
    clip_model.visual = get_peft_model(clip_model.visual, lora_config)
    clip_model.transformer = get_peft_model(clip_model.transformer, lora_config)

    return clip_model


def load_clip_to_cpu():
    backbone_name = cfg.backbone
    if cfg.backbone == "SurgVLP":
        backbone_name = "RN50"
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()  # type: ignore
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())  # type: ignore
    return model


def load_clip_model():
    clip_model = load_clip_to_cpu()
    clip_model.float()
    return clip_model, clip._transform(clip_model.visual.input_resolution)


# ------------------------------
# Text Encoder
# ------------------------------
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, retrun_adapater_func=None):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        if retrun_adapater_func is None:
            x = self.transformer(x)
        else:
            x = self.transformer([x, retrun_adapater_func])
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class LoRATextEncoder(nn.Module):
    """
    TextEncoder adapted for PeftModel (LoRA) injection.
    它显式访问基座 Transformer 模块，以避免 Peft 注入 'input_ids' 关键字参数导致的 TypeError。
    """

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, retrun_adapater_func=None):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)

        transformer_module = self.transformer.base_model if hasattr(self.transformer, "base_model") else self.transformer
        if retrun_adapater_func is None:
            x = transformer_module(x)
        else:
            x = self.transformer([x, retrun_adapater_func])

        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


# ------------------------------
# Prompt Learners
# ------------------------------
class ParentPromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        assert classnames is None
        with open(cfg.super_labels, "r") as f:
            text = f.readlines()
        classnames = [t.strip() for t in text]
        logger.info(f"Super classnames: {classnames}")

        n_cls = len(classnames)
        n_ctx = cfg.parent_n_ctx
        dtype = clip_model.dtype

        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize

        logger.info(f"Use {cfg.parent_ctx_init} initialize parent prompt")
        ctx_init = cfg.parent_ctx_init.replace("_", " ")
        assert n_ctx == len(ctx_init.split(" "))

        prompt = clip.tokenize(ctx_init)
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt).type(dtype)
        ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
        prompt_prefix = ctx_init

        logger.info(f'Initial context: "{prompt_prefix}"')
        logger.info(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # type: ignore

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS
        self.register_buffer("token_middle", embedding[:, 1 : (1 + n_ctx), :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

        assert embedding.requires_grad is False
        assert cfg.parent_ctx_init != "random"
        self.register_buffer("embedding", embedding)

    def forward(self):
        return self.embedding


class ChildPromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        classnames = classnames[0 : cfg.child_num]
        n_cls = len(classnames)
        n_ctx = cfg.child_n_ctx
        dtype = clip_model.dtype

        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize

        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim

        prompt_prefix = " ".join(["X"] * n_ctx)
        self.meta_net = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(1024, ctx_dim * n_ctx)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("linear2", nn.Linear(ctx_dim * n_ctx, ctx_dim * n_ctx)),
                ]
            )
        )

        logger.info(f"Number of child context words (tokens): {n_ctx}")

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

        self.parent_index = np.load(cfg.super_labels_index)
        self.parent_index = torch.Tensor(self.parent_index).type(torch.long)
        logger.info(f"{self.parent_index}")
        assert self.parent_index.requires_grad is False

    def forward(self, parent):
        prefix = self.token_prefix
        suffix = self.token_suffix

        parent = self.meta_net(parent)
        parent = parent[self.parent_index]
        parent = parent.reshape(-1, self.n_ctx, self.ctx_dim)

        prompts = torch.cat([prefix, parent, suffix], dim=1)
        return prompts


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        classnames = classnames[0 : cfg.child_num]
        n_cls = len(classnames)
        n_ctx = cfg.child_n_ctx
        dtype = clip_model.dtype

        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize

        prompt_prefix = "a photo "

        logger.info(f"Number of child context words (tokens): {n_ctx}")

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

        assert embedding.requires_grad is False
        self.register_buffer("embedding", embedding)

    def forward(self):
        return self.embedding


# ------------------------------
# GCN basics
# ------------------------------
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        return output + self.bias if self.bias is not None else output


class GCN(nn.Module):
    def __init__(self):
        super().__init__()
        if cfg.backbone == "RN50":
            self.gc1 = GraphConvolution(1024, 2048)
            self.gc2 = GraphConvolution(2048, 2048)
            self.gc3 = GraphConvolution(2048, 1024)
        elif cfg.backbone == "ViT-B/16":
            self.gc1 = GraphConvolution(512, 1024)
            self.gc2 = GraphConvolution(1024, 1024)
            self.gc3 = GraphConvolution(1024, 512)
        elif cfg.backbone == "ViT-L/14":
            self.gc1 = GraphConvolution(768, 1536)
            self.gc2 = GraphConvolution(1536, 1536)
            self.gc3 = GraphConvolution(1536, 768)
        elif cfg.backbone == "SurgVLP":
            self.gc1 = GraphConvolution(768, 1536)
            self.gc2 = GraphConvolution(1536, 1536)
            self.gc3 = GraphConvolution(1536, 768)
        else:
            raise NameError

        self.relu = nn.LeakyReLU(0.2)
        self.relu2 = nn.LeakyReLU(0.2)
        self.gamma = torch.nn.Parameter(torch.ones(1) * 0.9, requires_grad=True)

    def forward(self, features, relation):
        identity = features
        assert relation.requires_grad is False
        x = self.gc1(features, relation.cuda())
        x = self.relu(x)
        x = self.gc2(x, relation.cuda())
        x = self.relu2(x)
        x = self.gc3(x, relation.cuda())
        x = self.gamma * x + (1 - self.gamma) * identity
        return x


# ------------------------------
# Cross-model attention blocks
# ------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, heads, dropout_rate=0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.heads = heads
        self.head_dim = emb_dim // heads
        assert self.head_dim * heads == emb_dim

        self.values = nn.Linear(emb_dim, emb_dim, bias=False)
        self.keys = nn.Linear(emb_dim, emb_dim, bias=False)
        self.queries = nn.Linear(emb_dim, emb_dim, bias=False)
        self.fc_out = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, values, keys, queries):
        bs, _, _ = queries.shape

        values = self.values(values).reshape(bs, -1, self.heads, self.head_dim)
        keys = self.keys(keys).reshape(bs, -1, self.heads, self.head_dim)
        queries = self.queries(queries).reshape(bs, -1, self.heads, self.head_dim)

        energy = torch.einsum("bqhd,bkhd->bhqk", [queries, keys])
        attention = F.softmax(energy / (self.emb_dim ** (1 / 2)), dim=3)
        attention = self.dropout(attention)

        out = torch.einsum("bhql,blhd->bqhd", [attention, values]).reshape(bs, -1, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out


class TextToImageAttentionLayer(nn.Module):
    def __init__(self, emb_dim, dropout_rate=0.1):
        super().__init__()
        self.cross_attention = MultiHeadAttention(emb_dim, heads=4, dropout_rate=dropout_rate)
        self.self_attention = MultiHeadAttention(emb_dim, heads=4, dropout_rate=dropout_rate)
        self.layer_norm1 = nn.LayerNorm(emb_dim)
        self.layer_norm2 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, text, image):
        attended_image = self.cross_attention(text, text, image)
        attended_image = self.dropout(attended_image)
        attended_image = self.layer_norm1(attended_image + image)

        self_attended_image = self.self_attention(attended_image, attended_image, attended_image)
        self_attended_image = self.dropout(self_attended_image)
        self_attended_image = self.layer_norm2(self_attended_image + attended_image)
        return self_attended_image


class ImageToTextAttentionLayer(nn.Module):
    def __init__(self, emb_dim, dropout_rate=0.1):
        super().__init__()
        self.cross_attention = MultiHeadAttention(emb_dim, heads=4, dropout_rate=dropout_rate)
        self.self_attention = MultiHeadAttention(emb_dim, heads=4, dropout_rate=0.1)
        self.layer_norm1 = nn.LayerNorm(emb_dim)
        self.layer_norm2 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, text, image):
        attended_text = self.cross_attention(image, image, text)
        attended_text = self.dropout(attended_text)
        attended_text = self.layer_norm1(attended_text + text)

        self_attended_text = self.self_attention(attended_text, attended_text, attended_text)
        self_attended_text = self.dropout(self_attended_text)
        self_attended_text = self.layer_norm2(self_attended_text + attended_text)
        return self_attended_text


# ------------------------------
# CrossModel
# ------------------------------
class CrossModel(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.gcn = GCN()
        self.attend_txt = True
        self.attend_img = False
        self.attend_both = False
        assert (self.attend_img + self.attend_txt + self.attend_both) == 1

        if self.attend_img or self.attend_both:
            self.image_attention = nn.ModuleList([TextToImageAttentionLayer(emb_dim=512) for _ in range(2)])
        if self.attend_txt or self.attend_both:
            self.text_attention = nn.ModuleList([ImageToTextAttentionLayer(emb_dim=512) for _ in range(2)])

        self.relation = torch.Tensor(np.load(cfg.relation_file))
        self.parent_index = torch.Tensor(np.load(cfg.super_labels_index)).type(torch.long)

        child = self.relation[: cfg.child_num, : cfg.child_num].clone()
        parent = self.relation[cfg.child_num :, cfg.child_num :].clone()
        child = self.split(child)
        parent = self.split(parent)

        self.parent_self = parent.clone()
        self.relation = child

    def split(self, relation):
        _, max_idx = torch.topk(relation, int(3 / 4 * len(relation)))
        mask = torch.ones_like(relation).type(torch.bool)
        for i, idx in enumerate(max_idx):
            mask[i][idx] = 0
        relation[mask] = 0
        diag = torch.eye(len(relation)).type(torch.bool)
        relation[diag] = 0
        relation = relation / torch.sum(relation, dim=1).reshape(-1, 1) * cfg.reweight_p
        relation[diag] = (1 - cfg.reweight_p)
        return relation

    def attend_to_img(self, text_labels, image, label):
        if self.training:
            mask = label.unsqueeze(-1)
        image = image.unsqueeze(1)
        text_labels = text_labels.unsqueeze(0).repeat(image.shape[0], 1, 1)
        if self.training:
            text_labels = text_labels * mask

        img = []
        for i in range(text_labels.shape[1]):
            text = text_labels[:, i, :]
            im = image
            for layer in self.image_attention:
                im = layer(text, im)
            im = im.squeeze(1)
            img.append(im)

        image_features = torch.stack(img).permute(1, 0, 2)
        if self.training:
            image_features = image_features * mask
            num_ones = mask.sum(dim=1)
            image_features_sum = image_features.sum(dim=1)
            mean_image_features = image_features_sum / num_ones.clamp(min=1)
            mean_image_features = torch.where(num_ones == 0, image.squeeze(1), mean_image_features)
            return mean_image_features
        else:
            return text_labels, image_features

    def attend_to_text(self, text_labels, image1, label):
        text_labels = text_labels.unsqueeze(0).repeat(image1.shape[0], 1, 1)
        txt = []
        for i in range(text_labels.shape[1]):
            text = text_labels[:, i, :]
            if self.training:
                labels = label[:, i].unsqueeze(1)
            tx = text.unsqueeze(1)
            for layer in self.text_attention:
                tx = layer(tx, image1)
            tx = tx.squeeze(1)
            if self.training:
                tx = torch.where(labels == 0, text, tx)
            txt.append(tx)
        text_features = torch.stack(txt).permute(1, 0, 2)
        return text_features

    def forward(self, image, target=None):
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.tokenized_prompts)
        text_features = self.gcn(text_features, self.relation)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        if self.attend_txt:
            text_features = self.attend_to_text(text_features, image_features, target)
            image_features = image_features.unsqueeze(1)
            logits = logit_scale * torch.matmul(image_features, text_features.transpose(-1, -2)).squeeze(1)
        elif self.attend_img:
            if self.training:
                image_features = self.attend_to_img(text_features, image_features, target)
                logits = logit_scale * (image_features @ text_features.t())
            else:
                text_features, image_features = self.attend_to_img(text_features, image_features, target)
                logits = logit_scale * (image_features * text_features).sum(dim=-1)
        else:
            tf = self.attend_to_text(text_features, image_features, target)
            imf = self.attend_to_img(text_features, image_features, target).unsqueeze(1)
            logits = logit_scale * torch.matmul(imf, tf.transpose(-1, -2)).squeeze(1)

        return logits


# ------------------------------
# Baselines
# ------------------------------
class MMLSurgAdapt(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.gcn = GCN()

        self.relation = torch.Tensor(np.load(cfg.relation_file))
        self.parent_index = torch.Tensor(np.load(cfg.super_labels_index)).type(torch.long)

        child = self.relation[: cfg.child_num, : cfg.child_num].clone()
        parent = self.relation[cfg.child_num :, cfg.child_num :].clone()
        child = self.split(child)
        parent = self.split(parent)

        self.parent_self = parent.clone()
        self.relation = child

    def split(self, relation):
        _, max_idx = torch.topk(relation, int(3 / 4 * len(relation)))
        mask = torch.ones_like(relation).type(torch.bool)
        for i, idx in enumerate(max_idx):
            mask[i][idx] = 0
        relation[mask] = 0
        diag = torch.eye(len(relation)).type(torch.bool)
        relation[diag] = 0
        relation = relation / torch.sum(relation, dim=1).reshape(-1, 1) * cfg.reweight_p
        relation[diag] = (1 - cfg.reweight_p)
        return relation

    def encode_image(self, image, visual_adapter_func=None):
        if visual_adapter_func is not None:
            return self.image_encoder([image.type(self.dtype), visual_adapter_func])
        return self.image_encoder(image.type(self.dtype))

    def forward(self, image):
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.tokenized_prompts)
        text_features = self.gcn(text_features, self.relation)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features = self.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = 10 * image_features @ text_features.t()
        return logits


class VLPL(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.child_prompt_learner = PromptLearner(classnames, clip_model)
        self.child_tokeninzed_prompts = self.child_prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.temp = 0.03

    def encode_image(self, image, visual_adapter_func=None):
        if visual_adapter_func is not None:
            return self.image_encoder([image.type(self.dtype), visual_adapter_func])
        return self.image_encoder(image.type(self.dtype))

    def forward(self, image):
        prompts = self.child_prompt_learner()
        text_features = self.text_encoder(prompts, self.child_tokeninzed_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features = self.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = (1 / self.temp) * image_features @ text_features.t()
        return logits


class Resnet(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.image_encoder = models.resnet50(pretrained=True)
        self.image_encoder.fc = nn.Linear(self.image_encoder.fc.in_features, len(classnames))

    def forward(self, image):
        return self.image_encoder(image)


class ViT(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.image_encoder = models.vit_b_16(weights="ViT_B_16_Weights.DEFAULT")
        self.image_encoder.heads = nn.Sequential(nn.Linear(self.image_encoder.heads[0].in_features, len(classnames)))

    def forward(self, image):
        return self.image_encoder(image)


class CLIP_for_train(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        image_features = self.image_encoder(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = logit_scale * image_features @ text_features.t()
        return logits


class HSPNet(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.parent_prompt_learner = ParentPromptLearner(None, clip_model)
        self.child_prompt_learner = ChildPromptLearner(classnames, clip_model)
        self.parent_tokenized_prompts = self.parent_prompt_learner.tokenized_prompts
        self.child_tokeninzed_prompts = self.child_prompt_learner.tokenized_prompts

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.gcn = GCN()
        self.relation = torch.Tensor(np.load(cfg.relation_file))

        child = self.relation[: cfg.child_num, : cfg.child_num].clone()
        parent = self.relation[cfg.child_num :, cfg.child_num :].clone()
        child = self.split(child)
        parent = self.split(parent)

        self.parent_self = parent.clone()
        self.relation = child

    def split(self, relation):
        _, max_idx = torch.topk(relation, int(3 / 4 * len(relation)))
        mask = torch.ones_like(relation).type(torch.bool)
        for i, idx in enumerate(max_idx):
            mask[i][idx] = 0
        relation[mask] = 0
        diag = torch.eye(len(relation)).type(torch.bool)
        relation[diag] = 0
        relation = relation / torch.sum(relation, dim=1).reshape(-1, 1) * cfg.reweight_p
        relation[diag] = (1 - cfg.reweight_p)
        return relation

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        parent_prompts = self.parent_prompt_learner()
        parent_text_features = self.text_encoder(parent_prompts, self.parent_tokenized_prompts)
        parent_text_features = self.gcn(parent_text_features, self.parent_self)

        child_prompts = self.child_prompt_learner(parent_text_features)
        child_text_features = self.text_encoder(child_prompts, self.child_tokeninzed_prompts)

        text_features = self.gcn(child_text_features, self.relation)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = 10 * image_features @ text_features.t()
        return logits


# ======================================================================
# SCPNet (cos-only + external fusion by lambda; no use_gate; no 01 branch; no block row-max)
# ======================================================================
class StructuredPriorPrompter(nn.Module):
    """
    Cos-only SPP:
      - A_clip_cos = z @ z.t()  (≈[-1,1])
      - If external matrix provided: A_fused = (1-lam)*A_clip_cos + lam*A_ext_cos
      - Postprocess: (optional) intra-mutex mask + (optional) inter-cooccur mask
                    + sim_threshold + block row-max norm + self-loop(s_reweight)
      - Save raw CLIP text relation matrix (A_clip_cos) to npy/json if paths provided.
    """

    def __init__(self, classnames, clip_model):
        super().__init__()

        self.s_reweight = _cfg("SCP_S_REWEIGHT", _cfg("reweight_p", 0.2))
        self.sim_threshold = _cfg("SCP_SIM_THRESHOLD", _cfg("sim_threshold", 0.05))

        # structure switches (directly relevant)
        self.enable_intra_mutex = _cfg("SCP_ENABLE_INTRA_MUTEX_MASK", True)
        self.enable_inter_cooccur = _cfg("SCP_ENABLE_INTER_COOCCUR_MASK", True)

        if hasattr(cfg, "child_num") and cfg.child_num > 0:
            classnames = classnames[0 : cfg.child_num]
        self.n_cls = len(classnames)
        dtype = clip_model.dtype

        # fixed protocol ranges
        self.phase_range = slice(0, 7)
        self.view_range = slice(7, 10)
        self.action_range = slice(10, self.n_cls)
        self.ranges = [(0, 7), (7, 10), (10, self.n_cls)]

        # ---- CLIP text cosine similarity ----
        template = "a photo of a {}."
        classnames_proc = [n.replace("_", " ") for n in classnames]
        prompts = [template.format(n) for n in classnames_proc]
        tokenized = torch.cat([clip.tokenize(p) for p in prompts])

        # device-safe
        try:
            dev = next(clip_model.parameters()).device
            tokenized = tokenized.to(dev)
        except Exception:
            pass

        with torch.no_grad():
            z = clip_model.encode_text(tokenized).type(dtype)
        z = F.normalize(z, p=2, dim=-1)

        A_cos = z @ z.t()  # ≈[-1,1]
        self.register_buffer("A_clip_cos", A_cos)

        # ---- SAVE raw CLIP text relation matrix (A_clip_cos) ----
        save_path = _cfg("SCP_SAVE_RELATION_PATH", None)
        save_json_path = _cfg("SCP_SAVE_RELATION_JSON_PATH", None)
        if save_path or save_json_path:
            try:
                mat_np = self.A_clip_cos.detach().cpu().numpy()
                if save_path:
                    np.save(save_path, mat_np)
                    logger.info(f"SCP relation matrix saved to {save_path} (cos).")
                if save_json_path:
                    with open(save_json_path, "w", encoding="utf-8") as f:
                        json.dump(mat_np.tolist(), f)
                    logger.info(f"SCP relation matrix saved to {save_json_path} (cos).")
            except Exception as e:
                logger.warning(f"SCP relation matrix save failed: {e}")

        # ---- external fusion (weighted) ----
        self.use_gate = False  # keep name for compatibility with your current MMLSurgAdaptSCPNet
        lam_default = float(_cfg("SCP_EXTERNAL_FUSE_LAMBDA", 0.5))
        self.fuse_lam_phase  = float(_cfg("SCP_EXTERNAL_FUSE_LAM_PHASE",  lam_default))
        self.fuse_lam_view   = float(_cfg("SCP_EXTERNAL_FUSE_LAM_VIEW",   lam_default))
        self.fuse_lam_action = float(_cfg("SCP_EXTERNAL_FUSE_LAM_ACTION", lam_default))
        self.fuse_lam = self.fuse_lam_phase
        external_path = getattr(cfg, "SCP_EXTERNAL_MATRIX_PATH", None)
        if external_path and os.path.isfile(external_path):
            try:
                ext_np = np.load(external_path, allow_pickle=False)
                if isinstance(ext_np, np.lib.npyio.NpzFile):
                    raise ValueError("expected .npy but got .npz")
                ext = torch.as_tensor(ext_np, dtype=A_cos.dtype, device=A_cos.device)
                if ext.shape != A_cos.shape:
                    raise ValueError(f"shape mismatch: expected {A_cos.shape}, got {ext.shape}")
            except Exception as e:
                logger.warning(f"SCP external matrix load failed: {e}. External fusion disabled.")
            else:
                # symmetrize + clamp to cosine range (assume ext is cosine-like)
                ext = 0.5 * (ext + ext.t())
                ext = ext.clamp(-1.0, 1.0)
                self.register_buffer("A_ext_cos", ext)
                self.use_gate = True
                logger.info(
                    "SCP external fusion source loaded: lam_phase=%.3f, lam_view=%.3f, lam_action=%.3f, path=%s",
                    float(self.fuse_lam_phase),
                    float(self.fuse_lam_view),
                    float(self.fuse_lam_action),
                    str(external_path),
                )
        elif external_path:
            logger.warning(f"SCP external matrix path not found: {external_path}. External fusion disabled.")

        # legacy behavior when no external
        if not self.use_gate:
            self.register_buffer("A_star", self._postprocess_cos(self.A_clip_cos))

    def _postprocess_cos(self, A_cos: torch.Tensor):
        pr, vr, ar = self.phase_range, self.view_range, self.action_range
        ranges = self.ranges

        structure_mask = torch.ones_like(A_cos, dtype=torch.bool)

        # intra-group mutex
        if self.enable_intra_mutex:
            structure_mask[pr, pr] = False
            structure_mask[vr, vr] = False
            structure_mask[ar, ar] = False

        # inter-group cooccur
        if not self.enable_inter_cooccur:
            structure_mask[pr, vr] = False
            structure_mask[vr, pr] = False
            structure_mask[pr, ar] = False
            structure_mask[ar, pr] = False
            structure_mask[vr, ar] = False
            structure_mask[ar, vr] = False

        structure_mask.fill_diagonal_(True)
        A_masked = A_cos * structure_mask.float()

        # block-wise row-max normalization with threshold
        A_norm = torch.zeros_like(A_masked)
        for ss, se in ranges:
            for ts, te in ranges:
                block = A_masked[ss:se, ts:te]
                if block.sum() == 0:
                    continue
                block = torch.where(block > self.sim_threshold, block, torch.zeros_like(block))
                block_max, _ = block.max(dim=1, keepdim=True)
                r = block / (block_max + 1e-12)
                r = r.pow(_cfg("SCP_BLOCK_GAMMA", 1.5))   # 1.0=原版；1.5~2.0更尖锐
                A_norm[ss:se, ts:te] = r

        # self-loop rewrite
        diag = torch.eye(self.n_cls, dtype=torch.bool, device=A_cos.device)
        A_norm[diag] = 0.0
        A_norm = A_norm * (1.0 - self.s_reweight)
        A_norm[diag] = self.s_reweight
        return A_norm

    def forward(self):
        # no external: fixed A_star
        if not self.use_gate:
            return self.A_star

        # weighted fusion in cosine domain
        lam_p = max(0.0, min(1.0, float(self.fuse_lam_phase)))
        lam_v = max(0.0, min(1.0, float(self.fuse_lam_view)))
        lam_a = max(0.0, min(1.0, float(self.fuse_lam_action)))

        A_clip = self.A_clip_cos
        A_ext  = self.A_ext_cos

        groups = [
            (self.phase_range,  lam_p),
            (self.view_range,   lam_v),
            (self.action_range, lam_a),
        ]

        A_fused = A_clip.clone()

        # block-wise fusion: lam_block = (lam_i + lam_j)/2
        for si, li in groups:
            for sj, lj in groups:
                lam_block = 0.5 * (li + lj)
                A_fused[si, sj] = (1.0 - lam_block) * A_clip[si, sj] + lam_block * A_ext[si, sj]

        A_fused = 0.5 * (A_fused + A_fused.t())
        return self._postprocess_cos(A_fused)


class SemanticAssociationModule(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        num_layers = getattr(cfg, "gcn_layers", 3)
        mid_features = in_features * 2

        layers = nn.ModuleList()
        layers.append(GraphConvolution(in_features, mid_features))
        for _ in range(num_layers - 2):
            layers.append(GraphConvolution(mid_features, mid_features))
        layers.append(GraphConvolution(mid_features, out_features))

        self.gcn_layers = layers
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, H0, A_star):
        H_l = H0.float()
        for i, layer in enumerate(self.gcn_layers):
            A_star = A_star.to(H_l.device)
            H_l = layer(H_l, A_star)
            if i < len(self.gcn_layers) - 1:
                H_l = self.relu(H_l)
        return H0 + H_l


def get_topk_related_labels(pred_idx: int, logits_row: torch.Tensor, A_star: torch.Tensor):
    """
    score = sigmoid(logits_row) * A_star[pred_idx]
    In the other two big groups, each take top-k.
    """
    k = _cfg("SCP_TOPK_K", 1)
    use_sigmoid = _cfg("SCP_TOPK_USE_SIGMOID", True)

    n_cls = A_star.shape[0]
    phase_range = slice(0, 7)
    view_range = slice(7, 10)
    action_range = slice(10, n_cls)

    def in_range(i, r):
        return r.start <= i < r.stop

    if in_range(pred_idx, phase_range):
        targets = [view_range, action_range]
    elif in_range(pred_idx, view_range):
        targets = [phase_range, action_range]
    elif in_range(pred_idx, action_range):
        targets = [phase_range, view_range]
    else:
        raise ValueError("pred_idx not in any group range")

    conf = torch.sigmoid(logits_row) if use_sigmoid else F.softmax(logits_row, dim=-1)
    prior = A_star[pred_idx].to(logits_row.device)
    joint = conf * prior

    results = []
    for r in targets:
        scores = joint[r]
        kk = min(k, scores.numel())
        _, rel_idx = torch.topk(scores, kk)
        results.append((rel_idx + r.start).tolist())
    return results


def compensate_logits_by_pred_prob(logits: torch.Tensor, pred_idx: torch.Tensor, topk_related_labels: list):
    alpha = _cfg("SCP_COMP_ALPHA", 2.0)
    temp = _cfg("SCP_COMP_TEMP", 1.0)

    logits_comp = logits.clone()
    probs = F.softmax(logits_comp / temp, dim=-1)

    B = logits_comp.shape[0]
    for b in range(B):
        p = probs[b, int(pred_idx[b].item())]
        delta = alpha * p
        for group_topk in topk_related_labels[b]:
            for idx in group_topk:
                logits_comp[b, idx] += delta
    return logits_comp


def build_ignore_neg_mask_from_topk(topk_related_labels: list, n_cls: int, device):
    B = len(topk_related_labels)
    mask = torch.zeros((B, n_cls), dtype=torch.bool, device=device)
    for b in range(B):
        for group_topk in topk_related_labels[b]:
            for idx in group_topk:
                if 0 <= idx < n_cls:
                    mask[b, idx] = True
    return mask


class MMLSurgAdaptSCPNet(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.prompt_learner = PromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        # always returns final A_star (with/without external fusion)
        self.spp = StructuredPriorPrompter(classnames, clip_model)

        try:
            feat_dim = clip_model.text_projection.shape[1]
        except Exception:
            feat_dim = 1024

        self.sam = SemanticAssociationModule(feat_dim, feat_dim)

    def encode_image(self, image, visual_adapter_func=None):
        if visual_adapter_func is not None:
            return self.image_encoder([image.type(self.dtype), visual_adapter_func])
        return self.image_encoder(image.type(self.dtype))

    def forward(self, image, text_features=None):
        image_features = self.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        child_prompts = self.prompt_learner()
        Z = self.text_encoder(child_prompts, self.tokenized_prompts)

        # final structured prior
        A_star = self.spp()

        # GCN needs row-sum normalized adjacency
        row_sum = A_star.sum(dim=1, keepdim=True)
        A_gcn = A_star / (row_sum + 1e-12)

        text_features_refined = self.sam(Z, A_gcn)
        text_features_refined = text_features_refined / text_features_refined.norm(dim=-1, keepdim=True)

        logits = 10.0 * image_features @ text_features_refined.t()

        # ===== switches =====
        use_comp = _cfg("SCP_ENABLE_LOGIT_COMP", True)
        use_ignore = _cfg("SCP_ENABLE_IGNORE_MASK", True)

        ignore_neg_mask = None

        if use_comp or use_ignore:
            pred_idx = logits.argmax(dim=-1)
            topk_related_labels = [
                get_topk_related_labels(int(pred_idx[b].item()), logits[b], A_star)
                for b in range(pred_idx.shape[0])
            ]

            if use_ignore:
                ignore_neg_mask = build_ignore_neg_mask_from_topk(
                    topk_related_labels, n_cls=logits.shape[1], device=logits.device
                )

            if use_comp:
                logits = compensate_logits_by_pred_prob(logits, pred_idx, topk_related_labels)

        return logits, ignore_neg_mask
