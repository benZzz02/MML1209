from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
import torchvision.transforms as T
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from config import cfg
from log import logger
from peft import LoraConfig, get_peft_model, TaskType 
import numpy as np
_tokenizer = _Tokenizer()

def add_lora_to_clip(clip_model: nn.Module):
    """
    根据 cfg 配置动态创建 LoRA 适配器，并注入到 CLIP 编码器。
    
    返回：LoRA 包装后的 clip_model 模块。
    """
    
    # 检查是否启用 LoRA，如果未启用则跳过注入，但仍然冻结基座参数
    if not hasattr(cfg, 'LORA') or not cfg.LORA.ENABLED:
        logger.info("LoRA is disabled or LORA config not found. Freezing CLIP encoders for default Prompt Tuning.")
        # 如果 LoRA 关闭，则进行原始的 Prompt Tuning 冻结行为
        for param in clip_model.parameters():
            param.requires_grad_(False)
        return clip_model

    logger.info(f"LoRA enabled. Injecting LoRA adapters with r={cfg.LORA.R}, alpha={cfg.LORA.ALPHA}.")

    # 1. 冻结所有 CLIP 基座参数
    for param in clip_model.parameters():
        param.requires_grad_(False)
        
    # 2. 创建 LoRA 配置 (从 cfg 中读取参数)
    lora_config = LoraConfig(
        r=cfg.LORA.R,
        lora_alpha=cfg.LORA.ALPHA,
        target_modules=cfg.LORA.TARGET_MODULES, 
        lora_dropout=cfg.LORA.DROPOUT,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
        
    # 3. 对 Visual Encoder (clip_model.visual) 应用 LoRA
    clip_model.visual = get_peft_model(clip_model.visual, lora_config)
    
    # 4. 对 Text Transformer (clip_model.transformer) 应用 LoRA
    clip_model.transformer = get_peft_model(clip_model.transformer, lora_config)
    
    # LoRA 适配器参数会自动设置为 requires_grad=True
    return clip_model

# --- model.py (Modifying MMLSurgAdaptCoOp) ---
# 您无需修改 TextEncoder 的 forward，因为 LoRA 是在 self.transformer 内部注入的。
def load_clip_to_cpu():
    backbone_name = cfg.backbone
    if cfg.backbone == 'SurgVLP':
        backbone_name = 'RN50'
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(  # type: ignore
            model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())  # type: ignore

    return model


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
        if retrun_adapater_func == None:
            x = self.transformer(x)
        else:
            x = self.transformer([x, retrun_adapater_func])
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]),
              tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
class LoRATextEncoder(nn.Module):
    """
    TextEncoder adapted for PeftModel (LoRA) injection.
    它显式访问基座 Transformer 模块，以避免 Peft 注入 'input_ids' 关键字参数导致的 TypeError。
    """
    def __init__(self, clip_model):
        super().__init__()
        # 属性继承与原 TextEncoder 相同
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, retrun_adapater_func=None):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        # FIX: 显式获取 PeftModel 内部的基座 Transformer 模块
        # 确保调用时只传入位置参数 x
        transformer_module = self.transformer.base_model if hasattr(self.transformer, 'base_model') else self.transformer

        if retrun_adapater_func == None:
            x = transformer_module(x) 
        else:
            # 兼容原有的 adapter_func 逻辑
            x = self.transformer([x, retrun_adapater_func])
            
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        x = x[torch.arange(x.shape[0]),
              tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
class ParentPromptLearner(nn.Module):

    def __init__(self, classnames, clip_model):
        super().__init__()

        assert(classnames is None)
        with open(cfg.super_labels, 'r') as f:
            text = f.readlines()
        classnames = [ t.strip() for t in text]

        logger.info(f"Super classnames: {classnames}")

        n_cls = len(classnames)
        n_ctx = cfg.parent_n_ctx
        dtype = clip_model.dtype

        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # use given words to initialize context vectors
        logger.info(f"Use {cfg.parent_ctx_init} initialize parent prompt")
        ctx_init = cfg.parent_ctx_init.replace("_", " ")
        assert (n_ctx == len(ctx_init.split(" ")))
        prompt = clip.tokenize(ctx_init)
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt).type(dtype)
        ctx_vectors = embedding[0, 1:1 + n_ctx, :]
        prompt_prefix = ctx_init

        logger.info(f'Initial context: "{prompt_prefix}"')
        logger.info(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # type: ignore

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        #print(f"Prompt shape : {tokenized_prompts.shape}")
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(
                dtype)
        #print(f"Embedding shape : {embedding.shape}")

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix",
                             embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        self.register_buffer("token_middle", embedding[:, 1:(1 + n_ctx), :])
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

        assert (embedding.requires_grad == False)
        assert (cfg.parent_ctx_init != "random")
        self.register_buffer("embedding", embedding)

    def forward(self):
        return self.embedding

class ChildPromptLearner(nn.Module):

    def __init__(self, classnames, clip_model):
        super().__init__()
        classnames = classnames[0:cfg.child_num]
        n_cls = len(classnames)
        n_ctx = cfg.child_n_ctx
        dtype = clip_model.dtype
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        ctx_dim = clip_model.ln_final.weight.shape[0]
        #print(f"Ctx_dim : {ctx_dim}")
        self.ctx_dim = ctx_dim

        prompt_prefix = " ".join(["X"] * n_ctx)
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(1024, ctx_dim * n_ctx)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(ctx_dim * n_ctx, ctx_dim * n_ctx))
        ]))
    
        # use given words to initialize context vectors
        logger.info(f"Number of child context words (tokens): {n_ctx}")

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(
                dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix",
                             embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

        self.parent_index = np.load(cfg.super_labels_index)
        self.parent_index = torch.Tensor(self.parent_index).type(torch.long)
        logger.info(f"{self.parent_index}")
        assert(self.parent_index.requires_grad == False)

    def forward(self, parent):
        prefix = self.token_prefix
        suffix = self.token_suffix
        #print(f"Parent shape before meta net: {parent.shape}")
        parent = self.meta_net(parent)
        #print(f"Parent shape after meta net: {parent.shape}")
        parent = parent[self.parent_index]
        parent = parent.reshape(-1, self.n_ctx, self.ctx_dim)
        #print(f"Parent shape after reshaping: {parent.shape}")
        #print(f"Prefix shape: {prefix.shape}")
        #print(f"Suffix shape: {suffix.shape}")
        prompts = torch.cat(
            [
                prefix,
                parent,
                suffix
            ],
            dim = 1
        )
        return prompts
    
class PromptLearner(nn.Module):

    def __init__(self, classnames, clip_model):
        super().__init__()
        classnames = classnames[0:cfg.child_num]
        n_cls = len(classnames)
        n_ctx = cfg.child_n_ctx
        dtype = clip_model.dtype
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        ctx_dim = clip_model.ln_final.weight.shape[0]
        #print(f"Ctx_dim : {ctx_dim}")
        self.ctx_dim = ctx_dim

        prompt_prefix = "a photo "
    
        # use given words to initialize context vectors
        logger.info(f"Number of child context words (tokens): {n_ctx}")

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(
                dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix",
                             embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        assert (embedding.requires_grad == False)
        self.register_buffer("embedding", embedding)

    def forward(self):
        return self.embedding

def load_clip_model():
    clip_model = load_clip_to_cpu()

    # CLIP's default precision is fp16
    clip_model.float()
    return clip_model, clip._transform(clip_model.visual.input_resolution)

import math
import numpy as np
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        if cfg.backbone == 'RN50':
            self.gc1 = GraphConvolution(1024, 2048)
            self.gc2 = GraphConvolution(2048, 2048)
            self.gc3 = GraphConvolution(2048, 1024)
        elif cfg.backbone == 'ViT-B/16':
            self.gc1 = GraphConvolution(512, 1024)
            self.gc2 = GraphConvolution(1024, 1024)
            self.gc3 = GraphConvolution(1024, 512)
        elif cfg.backbone == 'ViT-L/14':
            self.gc1 = GraphConvolution(768, 1536)
            self.gc2 = GraphConvolution(1536, 1536)
            self.gc3 = GraphConvolution(1536, 768)
        elif cfg.backbone == 'SurgVLP':
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
        assert(relation.requires_grad == False)
        text_features = features
        text_features = self.gc1(text_features, relation.cuda())
        text_features = self.relu(text_features)
        text_features = self.gc2(text_features, relation.cuda())
        text_features = self.relu2(text_features)
        text_features = self.gc3(text_features, relation.cuda())
        text_features = self.gamma * text_features + (1-self.gamma) * identity
        return text_features
    
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, heads, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()

        self.emb_dim = emb_dim
        self.heads = heads
        self.head_dim = emb_dim // heads

        assert (
            self.head_dim * heads == emb_dim
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(emb_dim, emb_dim, bias=False)
        self.keys = nn.Linear(emb_dim, emb_dim, bias=False)
        self.queries = nn.Linear(emb_dim, emb_dim, bias=False)
        self.fc_out = nn.Linear(emb_dim, emb_dim)

        self.dropout = nn.Dropout(dropout_rate)

        # nn.init.xavier_uniform_(self.queries.weight)
        # nn.init.xavier_uniform_(self.keys.weight)
        # nn.init.xavier_uniform_(self.values.weight)
        # nn.init.xavier_uniform_(self.fc_out.weight)
        # nn.init.zeros_(self.fc_out.bias)

    def forward(self, values, keys, queries):
        bs, _, _ = queries.shape

        values = self.values(values).reshape(bs, -1, self.heads, self.head_dim)
        keys = self.keys(keys).reshape(bs, -1, self.heads, self.head_dim)
        queries = self.queries(queries).reshape(bs, -1, self.heads, self.head_dim)

        energy = torch.einsum("bqhd,bkhd->bhqk", [queries, keys])

        attention = F.softmax(energy / (self.emb_dim ** (1 / 2)), dim=3)
        attention = self.dropout(attention)

        out = torch.einsum("bhql,blhd->bqhd", [attention, values]).reshape(
            bs, -1, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out
    
class TextToImageAttentionLayer(nn.Module):
    def __init__(self, emb_dim, dropout_rate=0.1):
        super(TextToImageAttentionLayer, self).__init__()

        self.cross_attention = MultiHeadAttention(
            emb_dim, heads=4, dropout_rate=dropout_rate
        )
        self.self_attention = MultiHeadAttention(
            emb_dim, heads=4, dropout_rate=dropout_rate
        )

        self.layer_norm1 = nn.LayerNorm(emb_dim)
        self.layer_norm2 = nn.LayerNorm(emb_dim)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, text, image):

        attended_image = self.cross_attention(text, text, image)
        attended_image = self.dropout(attended_image)
        attended_image = self.layer_norm1(attended_image + image)  # Residual connection

        self_attended_image = self.self_attention(
            attended_image, attended_image, attended_image
        )
        self_attended_image = self.dropout(self_attended_image)
        self_attended_image = self.layer_norm2(
            self_attended_image + attended_image
        )  # Residual connection

        return self_attended_image
    

class ImageToTextAttentionLayer(nn.Module):
    def __init__(self, emb_dim, dropout_rate=0.1):
        super(ImageToTextAttentionLayer, self).__init__()

        self.cross_attention = MultiHeadAttention(
            emb_dim, heads=4, dropout_rate=dropout_rate
        )
        self.self_attention = MultiHeadAttention(
            emb_dim, heads=4, dropout_rate=0.1
        )

        self.layer_norm1 = nn.LayerNorm(emb_dim)
        self.layer_norm2 = nn.LayerNorm(emb_dim)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, text, image):

        attended_text = self.cross_attention(image, image, text)
        attended_text = self.dropout(attended_text)
        attended_text = self.layer_norm1(attended_text + text)  # Residual connection

        self_attended_text = self.self_attention(
            attended_text, attended_text, attended_text
        )
        self_attended_text = self.dropout(self_attended_text)
        self_attended_text = self.layer_norm2(
            self_attended_text + attended_text
        )  # Residual connection

        return self_attended_text
    
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
            self.image_attention = nn.ModuleList(
                [
                    TextToImageAttentionLayer(emb_dim=512, dropout_rate=0.1)
                    for _ in range(2)
                ]
            )
        if self.attend_txt or self.attend_both:
            self.text_attention = nn.ModuleList(
                [
                    ImageToTextAttentionLayer(emb_dim=512, dropout_rate=0.1)
                    for _ in range(2)
                ]
            )

        self.relation = torch.Tensor(np.load(cfg.relation_file))
        self.parent_index = np.load(cfg.super_labels_index)
        self.parent_index = torch.Tensor(self.parent_index).type(torch.long)

        child = self.relation[:cfg.child_num, :cfg.child_num].clone()
        parent = self.relation[cfg.child_num:, cfg.child_num:].clone()
        child = self.split(child)
        parent = self.split(parent)
        
        self.parent_self = parent.clone()
        self.relation = child

    def split(self, relation):
        _ ,max_idx = torch.topk(relation, int(3/4 * len(relation)))
        mask = torch.ones_like(relation).type(torch.bool)
        for i, idx in enumerate(max_idx):
            mask[i][idx] = 0
        relation[mask] = 0
        dialog = torch.eye(len(relation)).type(torch.bool)
        relation[dialog] = 0
        relation = relation / torch.sum(relation, dim=1).reshape(-1, 1) * cfg.reweight_p 
        relation[dialog] = (1-cfg.reweight_p)
        return relation
    
    def attend_to_img(self,text_labels,image,label): # 110,512   32,512  32,110
        if self.training:
            mask = label.unsqueeze(-1) # 32,110,1
        image = image.unsqueeze(1) # 32,1,512
        text_labels = text_labels.unsqueeze(0).repeat(image.shape[0],1,1) # 32,110,512
        if self.training:
            text_labels = text_labels * mask # 32,110,512
        img = []
        for i in range(text_labels.shape[1]):
            text = text_labels[:,i,:] # 32,512
            im = image # 32,1,512
            for layer in self.image_attention:
                im = layer(text,im) # 32,1,512
            im = im.squeeze(1) # 32,512
            img.append(im)

        image_features = torch.stack(img) # 110,32,512
        image_features = image_features.permute(1,0,2) # 32,110,512
        if self.training:
            image_features = image_features * mask # 32,110,512
            num_ones = mask.sum(dim=1) # 32,1
            image_features_sum = image_features.sum(dim=1)
            mean_image_features = image_features_sum/num_ones.clamp(min=1)
            mean_image_features = torch.where(num_ones==0,image,mean_image_features) #32,512
            return mean_image_features
        else:
            return text_labels, image_features
    
    def attend_to_text(self,text_labels,image1,label): # 110,512   32,512   32,110
        image = image1.unsqueeze(1) # 32,1,512
        text_labels = text_labels.unsqueeze(0).repeat(image.shape[0],1,1) # 32,110,512
        txt = []
        for i in range(text_labels.shape[1]):
            text = text_labels[:,i,:] # 32,512
            if self.training:
                labels = label[:,i].unsqueeze(1) # 32
            tx = text.unsqueeze(1) # 32,1,512
            for layer in self.text_attention:
                tx = layer(tx,image1) # 32,1,512
            tx = tx.squeeze(1) # 32,512
            if self.training:
                tx = torch.where(labels==0,text,tx)
            txt.append(tx)

        text_features = torch.stack(txt) # 110,32,512
        text_features = text_features.permute(1,0,2) # 32,110,512
        return text_features
    
    def forward(self, image, target = None):
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts,self.tokenized_prompts)

        text_features = self.gcn(text_features, self.relation)
        
        text_features = text_features / text_features.norm(dim=-1,
                                                            keepdim=True)
        logit_scale = self.logit_scale.exp()
        print(f"Logit scale: {logit_scale}")
    
        image_features = self.image_encoder(image)
        image_features = image_features / image_features.norm(dim=-1,
                                                              keepdim=True)
        
        if self.attend_txt:
            text_features = self.attend_to_text(text_features,image_features,target)
            image_features = image_features.unsqueeze(1)
            #print(f"text feat: {text_features}")
            logits = logit_scale * torch.matmul(image_features,text_features.transpose(-1,-2)).squeeze(1)
        if self.attend_img:
            if self.training:
                image_features = self.attend_to_img(text_features,image_features,target)
                logits = logit_scale * (image_features @ text_features.t())
            else:
                text_features, image_features = self.attend_to_img(text_features,image_features,target)
                # print(f"Img feat: {image_features.shape}")
                # print(f"txt feat: {text_features.shape}")
                logits = logit_scale * (image_features*text_features).sum(dim=-1)
        if self.attend_both:
            tf = self.attend_to_text(text_features,image_features,target)
            imf = self.attend_to_img(text_features,image_features,target).unsqueeze(1)
            logits = logit_scale * torch.matmul(imf,tf.transpose(-1,-2)).squeeze(1)

        # if not self.training:
        #     print(logits.shape)

        return logits

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
        self.parent_index = np.load(cfg.super_labels_index)
        self.parent_index = torch.Tensor(self.parent_index).type(torch.long)

        child = self.relation[:cfg.child_num, :cfg.child_num].clone()
        parent = self.relation[cfg.child_num:, cfg.child_num:].clone()
        child = self.split(child)
        parent = self.split(parent)
        
        self.parent_self = parent.clone()
        self.relation = child

    def split(self, relation):
        _ ,max_idx = torch.topk(relation, int(3/4 * len(relation)))
        mask = torch.ones_like(relation).type(torch.bool)
        for i, idx in enumerate(max_idx):
            mask[i][idx] = 0
        relation[mask] = 0
        dialog = torch.eye(len(relation)).type(torch.bool)
        relation[dialog] = 0
        relation = relation / torch.sum(relation, dim=1).reshape(-1, 1) * cfg.reweight_p 
        relation[dialog] = (1-cfg.reweight_p)
        return relation
    
    def encode_text(self, prompts, tokenized_prompts, text_adapter_func=None):
        if text_adapter_func is not None:
            text_features = self.text_encoder(
                prompts, tokenized_prompts, text_adapter_func
            )
        else:
            text_features = self.text_encoder(
                prompts, tokenized_prompts
            )
        return text_features
    
    def encode_image(self, image, visual_adapter_func=None):
        if visual_adapter_func is not None:
            image_features = self.image_encoder(
                [image.type(self.dtype), visual_adapter_func]
            )
        else:
            image_features = self.image_encoder(
                image.type(self.dtype)
            )
        return image_features
    
    def forward(self, image):

        child_prompts = self.prompt_learner()

        child_text_features = self.text_encoder(child_prompts,self.tokenized_prompts)

        text_features = child_text_features

        text_features = self.gcn(text_features, self.relation)
        
        text_features = text_features / text_features.norm(dim=-1,
                                                            keepdim=True)
        image_features = self.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1,
                                                              keepdim=True)
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

        self.gcn = GCN()

        self.relation = torch.Tensor(np.load(cfg.relation_file))
        self.parent_index = np.load(cfg.super_labels_index)
        self.parent_index = torch.Tensor(self.parent_index).type(torch.long)

        child = self.relation[:cfg.child_num, :cfg.child_num].clone()
        parent = self.relation[cfg.child_num:, cfg.child_num:].clone()
        child = self.split(child)
        parent = self.split(parent)
        
        self.parent_self = parent.clone()
        self.relation = child

    def split(self, relation):
        _ ,max_idx = torch.topk(relation, int(3/4 * len(relation)))
        mask = torch.ones_like(relation).type(torch.bool)
        for i, idx in enumerate(max_idx):
            mask[i][idx] = 0
        relation[mask] = 0
        dialog = torch.eye(len(relation)).type(torch.bool)
        relation[dialog] = 0
        relation = relation / torch.sum(relation, dim=1).reshape(-1, 1) * cfg.reweight_p 
        relation[dialog] = (1-cfg.reweight_p)
        return relation
    
    def encode_text(self, prompts, tokenized_prompts, text_adapter_func=None):
        if text_adapter_func is not None:
            text_features = self.text_encoder(
                prompts, tokenized_prompts, text_adapter_func
            )
        else:
            text_features = self.text_encoder(
                prompts, tokenized_prompts
            )
        return text_features
    
    def encode_image(self, image, visual_adapter_func=None):
        if visual_adapter_func is not None:
            image_features = self.image_encoder(
                [image.type(self.dtype), visual_adapter_func]
            )
        else:
            image_features = self.image_encoder(
                image.type(self.dtype)
            )
        return image_features
    
    def forward(self, image):
        child_prompts = self.child_prompt_learner()
        child_text_features = self.text_encoder(child_prompts,self.child_tokeninzed_prompts)

        text_features = child_text_features
        
        text_features = text_features / text_features.norm(dim=-1,
                                                            keepdim=True)
        image_features = self.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1,
                                                              keepdim=True)
        logits = (1/self.temp) * image_features @ text_features.t()
        return logits
    
class Resnet(nn.Module):

    def __init__(self, classnames, clip_model):
        super().__init__()
        self.image_encoder = models.resnet50(pretrained=True)
        self.image_encoder.fc = nn.Linear(self.image_encoder.fc.in_features,len(classnames))
    
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

        child_prompts = self.prompt_learner()
        text_features = self.text_encoder(child_prompts,self.tokenized_prompts)
        
        text_features = text_features / text_features.norm(dim=-1,
                                                            keepdim=True)
        logit_scale = self.logit_scale.exp()
        image_features = self.image_encoder(image)
        image_features = image_features / image_features.norm(dim=-1,
                                                                keepdim=True)
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
        self.parent_index = np.load(cfg.super_labels_index)
        self.parent_index = torch.Tensor(self.parent_index).type(torch.long)

        child = self.relation[:cfg.child_num, :cfg.child_num].clone()
        parent = self.relation[cfg.child_num:, cfg.child_num:].clone()
        child = self.split(child)
        parent = self.split(parent)
        
        self.parent_self = parent.clone()
        self.relation = child

    def split(self, relation):
        _ ,max_idx = torch.topk(relation, int(3/4 * len(relation)))
        mask = torch.ones_like(relation).type(torch.bool)
        for i, idx in enumerate(max_idx):
            mask[i][idx] = 0
        relation[mask] = 0
        dialog = torch.eye(len(relation)).type(torch.bool)
        relation[dialog] = 0
        relation = relation / torch.sum(relation, dim=1).reshape(-1, 1) * cfg.reweight_p 
        relation[dialog] = (1-cfg.reweight_p)
        return relation
    
    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1,
                                                              keepdim=True)
        parent_prompts = self.parent_prompt_learner()
        parent_text_features = self.text_encoder(parent_prompts, self.parent_tokenized_prompts)
        
        parent_text_features = self.gcn(parent_text_features, self.parent_self)
        child_prompts = self.child_prompt_learner(parent_text_features)
        child_text_features = self.text_encoder(child_prompts, self.child_tokeninzed_prompts)

        text_features = child_text_features

        text_features = self.gcn(text_features, self.relation)
        
        text_features = text_features / text_features.norm(dim=-1,
                                                            keepdim=True)
        logits = 10 * image_features @ text_features.t()
        return logits

class CoOpPromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        if hasattr(cfg, 'child_num') and cfg.child_num > 0: classnames = classnames[0:cfg.child_num]
        n_cls, n_ctx = len(classnames), cfg.child_n_ctx if hasattr(cfg, 'child_n_ctx') else 16
        dtype, ctx_dim = clip_model.dtype, clip_model.ln_final.weight.shape[0]
        logger.info("Initializing CoOp prompts..."); ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype); nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        prompt_prefix = " ".join(["X"] * n_ctx)
        prompts = [prompt_prefix + " " + name.replace("_", " ") + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad(): embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        self.register_buffer("token_prefix", embedding[:, :1, :]); self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])
        self.n_cls = n_cls; self.tokenized_prompts = tokenized_prompts
    def forward(self):
        return torch.cat([self.token_prefix, self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1), self.token_suffix], dim=1)

class MMLSurgAdaptCoOp(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__(); self.prompt_learner = CoOpPromptLearner(classnames, clip_model); self.text_encoder = TextEncoder(clip_model)
        self.image_encoder = clip_model.visual; self.logit_scale = clip_model.logit_scale; self.dtype = clip_model.dtype
    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype)); prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
        image_features = F.normalize(image_features, p=2, dim=-1); text_features = F.normalize(text_features, p=2, dim=-1)
        return self.logit_scale.exp() * image_features @ text_features.t()

# ==================== DualCoOp (双静态提示) 实现 ====================
class DualCoOpPromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        if hasattr(cfg, 'child_num') and cfg.child_num > 0: classnames = classnames[0:cfg.child_num]
        n_cls, n_ctx = len(classnames), cfg.child_n_ctx if hasattr(cfg, 'child_n_ctx') else 16
        dtype, ctx_dim = clip_model.dtype, clip_model.ln_final.weight.shape[0]
        logger.info("Initializing DualCoOp prompts...")
        ctx_vectors_pos = torch.empty(n_ctx, ctx_dim, dtype=dtype); nn.init.normal_(ctx_vectors_pos, std=0.02)
        ctx_vectors_neg = torch.empty(n_ctx, ctx_dim, dtype=dtype); nn.init.normal_(ctx_vectors_neg, std=0.02)
        self.ctx_pos = nn.Parameter(ctx_vectors_pos); self.ctx_neg = nn.Parameter(ctx_vectors_neg)
        prompt_prefix = " ".join(["X"] * n_ctx)
        prompts = [prompt_prefix + " " + name.replace("_", " ") + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad(): embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        self.register_buffer("token_prefix", embedding[:, :1, :]); self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])
        self.n_cls = n_cls; self.tokenized_prompts = tokenized_prompts
    def forward(self):
        ctx_pos = self.ctx_pos.unsqueeze(0).expand(self.n_cls, -1, -1); ctx_neg = self.ctx_neg.unsqueeze(0).expand(self.n_cls, -1, -1)
        prompts_pos = torch.cat([self.token_prefix, ctx_pos, self.token_suffix], dim=1)
        prompts_neg = torch.cat([self.token_prefix, ctx_neg, self.token_suffix], dim=1)
        return prompts_pos, prompts_neg

class MMLSurgAdaptDualCoOp(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__(); self.prompt_learner = DualCoOpPromptLearner(classnames, clip_model); self.text_encoder = TextEncoder(clip_model)
        self.image_encoder = clip_model.visual; self.logit_scale = clip_model.logit_scale; self.dtype = clip_model.dtype
    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype)); prompts_pos, prompts_neg = self.prompt_learner()
        text_features_pos = self.text_encoder(prompts_pos, self.prompt_learner.tokenized_prompts)
        text_features_neg = self.text_encoder(prompts_neg, self.prompt_learner.tokenized_prompts)
        image_features, text_features_pos, text_features_neg = map(lambda t: F.normalize(t, p=2, dim=-1), (image_features, text_features_pos, text_features_neg))
        logits_pos = self.logit_scale.exp() * image_features @ text_features_pos.t()
        logits_neg = self.logit_scale.exp() * image_features @ text_features_neg.t()
        return logits_pos - logits_neg

# ==================== CoCoOp (动态提示) 实现 ====================
class CoCoOpPromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        if hasattr(cfg, 'child_num') and cfg.child_num > 0: classnames = classnames[0:cfg.child_num]
        n_cls, n_ctx = len(classnames), cfg.child_n_ctx if hasattr(cfg, 'child_n_ctx') else 16
        dtype, ctx_dim, vis_dim = clip_model.dtype, clip_model.ln_final.weight.shape[0], clip_model.visual.output_dim
        logger.info("Initializing CoCoOp prompts with MetaNet..."); ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype); nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        self.meta_net = nn.Sequential(OrderedDict([("linear1", nn.Linear(vis_dim, vis_dim // 16)), ("relu", nn.ReLU(inplace=True)), ("linear2", nn.Linear(vis_dim // 16, ctx_dim))]))
        if dtype == torch.float16: self.meta_net.half()
        prompt_prefix = " ".join(["X"] * n_ctx); prompts = [prompt_prefix + " " + name.replace("_", " ") + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad(): embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        self.register_buffer("token_prefix", embedding[:, :1, :]); self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])
        self.n_cls = n_cls; self.tokenized_prompts = tokenized_prompts
    def forward(self, im_features):
        batch_size = im_features.shape[0]; bias = self.meta_net(im_features).unsqueeze(1)
        ctx_shifted = (self.ctx.unsqueeze(0) + bias).unsqueeze(1).expand(-1, self.n_cls, -1, -1)
        prefix = self.token_prefix.unsqueeze(0).expand(batch_size, -1, -1, -1); suffix = self.token_suffix.unsqueeze(0).expand(batch_size, -1, -1, -1)
        return torch.cat([prefix, ctx_shifted, suffix], dim=2)

class MMLSurgAdaptCoCoOp(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__(); self.prompt_learner = CoCoOpPromptLearner(classnames, clip_model); self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model); self.logit_scale = clip_model.logit_scale; self.dtype = clip_model.dtype
    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype)); prompts = self.prompt_learner(image_features)
        batch_size, n_cls, n_tkn, dim = prompts.shape; tokenized_prompts = self.prompt_learner.tokenized_prompts.repeat(batch_size, 1)
        prompts = prompts.reshape(batch_size * n_cls, n_tkn, dim)
        text_features = self.text_encoder(prompts, tokenized_prompts).reshape(batch_size, n_cls, -1)
        image_features_norm = F.normalize(image_features, p=2, dim=-1); text_features_norm = F.normalize(text_features, p=2, dim=-1)
        return self.logit_scale.exp() * torch.einsum('bd,bcd->bc', image_features_norm, text_features_norm)

# ==================== DualCoCoOp (双动态提示) 实现 [新增+修正] ====================
class DualCoCoOpPromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        if hasattr(cfg, 'child_num') and cfg.child_num > 0: classnames = classnames[0:cfg.child_num]
        n_cls, n_ctx = len(classnames), cfg.child_n_ctx if hasattr(cfg, 'child_n_ctx') else 16
        dtype, ctx_dim, vis_dim = clip_model.dtype, clip_model.ln_final.weight.shape[0], clip_model.visual.output_dim
        logger.info("Initializing DualCoCoOp prompts with MetaNet...")
        ctx_vectors_pos = torch.empty(n_ctx, ctx_dim, dtype=dtype); nn.init.normal_(ctx_vectors_pos, std=0.02)
        ctx_vectors_neg = torch.empty(n_ctx, ctx_dim, dtype=dtype); nn.init.normal_(ctx_vectors_neg, std=0.02)
        self.ctx_pos = nn.Parameter(ctx_vectors_pos); self.ctx_neg = nn.Parameter(ctx_vectors_neg)
        self.meta_net = nn.Sequential(OrderedDict([("linear1", nn.Linear(vis_dim, vis_dim // 16)), ("relu", nn.ReLU(inplace=True)), ("linear2", nn.Linear(vis_dim // 16, ctx_dim))]))
        if dtype == torch.float16: self.meta_net.half()
        prompt_prefix = " ".join(["X"] * n_ctx); prompts = [prompt_prefix + " " + name.replace("_", " ") + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad(): embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        self.register_buffer("token_prefix", embedding[:, :1, :]); self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])
        self.n_cls = n_cls; self.tokenized_prompts = tokenized_prompts
    def forward(self, im_features):
        batch_size = im_features.shape[0]; bias = self.meta_net(im_features).unsqueeze(1)
        ctx_pos_shifted = (self.ctx_pos.unsqueeze(0) + bias).unsqueeze(1).expand(-1, self.n_cls, -1, -1)
        ctx_neg_shifted = (self.ctx_neg.unsqueeze(0) + bias).unsqueeze(1).expand(-1, self.n_cls, -1, -1)
        prefix = self.token_prefix.unsqueeze(0).expand(batch_size, -1, -1, -1); suffix = self.token_suffix.unsqueeze(0).expand(batch_size, -1, -1, -1)
        prompts_pos = torch.cat([prefix, ctx_pos_shifted, suffix], dim=2)
        prompts_neg = torch.cat([prefix, ctx_neg_shifted, suffix], dim=2)
        return prompts_pos, prompts_neg

class MMLSurgAdaptDualCoCoOp(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__(); self.prompt_learner = DualCoCoOpPromptLearner(classnames, clip_model); self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model); self.logit_scale = clip_model.logit_scale; self.dtype = clip_model.dtype
    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype)); prompts_pos, prompts_neg = self.prompt_learner(image_features)
        batch_size, n_cls, n_tkn, dim = prompts_pos.shape; tokenized_prompts = self.prompt_learner.tokenized_prompts.repeat(batch_size, 1)
        prompts_pos = prompts_pos.reshape(batch_size * n_cls, n_tkn, dim); prompts_neg = prompts_neg.reshape(batch_size * n_cls, n_tkn, dim)
        text_features_pos = self.text_encoder(prompts_pos, tokenized_prompts).reshape(batch_size, n_cls, -1)
        text_features_neg = self.text_encoder(prompts_neg, tokenized_prompts).reshape(batch_size, n_cls, -1)
        image_features, text_features_pos, text_features_neg = map(lambda t: F.normalize(t, p=2, dim=-1), (image_features, text_features_pos, text_features_neg))
        logits_pos = self.logit_scale.exp() * torch.einsum('bd,bcd->bc', image_features, text_features_pos)
        logits_neg = self.logit_scale.exp() * torch.einsum('bd,bcd->bc', image_features, text_features_neg)
        return logits_pos - logits_neg

# ==================== 冻结编码器的版本 (Prompt Tuning) ====================
def freeze_encoders(model, model_name):
    logger.info(f"Initializing {model_name}: Freezing CLIP encoders for prompt tuning.")
    for name, param in model.named_parameters():
        if "prompt_learner" not in name: param.requires_grad = False
    model.image_encoder.eval(); model.text_encoder.eval()

class MMLSurgAdaptCoOpFrozen(MMLSurgAdaptCoOp):
    def __init__(self, classnames, clip_model): 
        super().__init__(classnames, clip_model)
        freeze_encoders(self, "MMLSurgAdaptCoOpFrozen")
        
    # [新增] 重写 train 方法，确保 encoder 永远是 eval 模式
    def train(self, mode=True):
        super().train(mode)
        self.image_encoder.eval()
        self.text_encoder.eval()
        return self

class MMLSurgAdaptDualCoOpFrozen(MMLSurgAdaptDualCoOp):
    def __init__(self, classnames, clip_model): 
        super().__init__(classnames, clip_model)
        freeze_encoders(self, "MMLSurgAdaptDualCoOpFrozen")

    def train(self, mode=True):
        super().train(mode)
        self.image_encoder.eval()
        self.text_encoder.eval()
        return self
class MMLSurgAdaptCoCoOpFrozen(MMLSurgAdaptCoCoOp):
    def __init__(self, classnames, clip_model): 
        super().__init__(classnames, clip_model)
        freeze_encoders(self, "MMLSurgAdaptCoCoOpFrozen")
        
    # [新增] 重写 train 方法，确保 encoder 永远是 eval 模式
    def train(self, mode=True):
        super().train(mode)
        self.image_encoder.eval()
        self.text_encoder.eval()
        return self
    
# 1. 文本特征自注意力模块 (TextSelfAttention) - 替代 GCN
class TextSelfAttention(nn.Module):
    """
    Transformer Encoder blocks for Self-Attention on Text/Class Embeddings.
    输入和输出形状：[N_cls, D_embed]。
    """
    def __init__(self, embed_dim, num_layers=2, heads=8, dropout_rate=0.1):
        super(TextSelfAttention, self).__init__()
        
        # 模块参数与 GCN 类中的维度保持一致
        self_attn_layer = lambda: MultiHeadAttention(embed_dim, heads=heads, dropout_rate=dropout_rate)
        
        encoder_layers = []
        for _ in range(num_layers): # 默认使用 2 层
            layer = nn.ModuleDict({
                'norm1': nn.LayerNorm(embed_dim),
                'attn': self_attn_layer(),
                'norm2': nn.LayerNorm(embed_dim),
                'mlp': nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Linear(embed_dim * 4, embed_dim),
                    nn.Dropout(dropout_rate)
                )
            })
            encoder_layers.append(layer)
        
        self.layers = nn.ModuleList(encoder_layers)

    def forward(self, x):
        # x shape: [N_cls, D] -> 转换为 [1, N_cls, D] 以适应 MultiHeadAttention
        x = x.unsqueeze(0) 
        
        for layer in self.layers:
            residual = x
            x_norm = layer['norm1'](x)
            attn_output = layer['attn'](x_norm, x_norm, x_norm) 
            x = residual + attn_output 

            residual = x
            x_norm = layer['norm2'](x)
            x = residual + layer['mlp'](x_norm) 

        return x.squeeze(0) # 移除 Batch 维度，返回 [N_cls, D]


# 2. 主模型类：CLIP_TextAttention (CLIP + Text Attention，无 Prompt Learner)
class CLIP_TextAttention(nn.Module):
    """
    模仿 CLIP_for_train，使用固定文本模板，并替换 GCN 为 Text Attention。
    """
    def __init__(self, classnames, clip_model):
        super().__init__()
        
        # 1. CLIP Encoders 和参数
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
        # 2. 固定文本嵌入 (Fixed Text Embeddings)
        # 模仿 PromptLearner 的目标，但使用固定文本模板
        classnames = classnames[0:cfg.child_num]
        classnames = [name.replace("_", " ") for name in classnames]
        
        # 使用配置中的 child_ctx_init 作为前缀，或者默认使用 CLIP 的 'a photo of a'
        if hasattr(cfg, 'child_ctx_init') and cfg.child_ctx_init:
            template = cfg.child_ctx_init.strip() + " {}."
        else:
            template = "a photo of a {}."
            
        prompts = [template.format(name) for name in classnames]
        
        self.tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        
        # 在初始化时计算固定文本嵌入的 Tensor (固定前缀+后缀)
        # 编码器参数和位置嵌入都是固定的，所以只保留 token embedding 中间的部分
        with torch.no_grad():
            prompts_full_embedding = clip_model.token_embedding(self.tokenized_prompts).type(self.dtype)
        
        # 模仿 CLIP_for_train 中的 tokenized_prompts 作为 prompts 输入
        self.register_buffer("fixed_prompts_tensor", prompts_full_embedding)
        
        # 3. Text Attention Module
        embed_dim = clip_model.text_projection.shape[1] 
        self.text_attention_module = TextSelfAttention(
            embed_dim=embed_dim, 
            num_layers=2, 
            heads=8
        )

    def forward(self, image):
        # 1. 图像特征编码
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = F.normalize(image_features, p=2, dim=-1)

        # 2. 文本特征编码 (使用固定的 prompts tensor)
        # 这里的 fixed_prompts_tensor 包含了 SOS, EOS 等 token 嵌入
        text_features = self.text_encoder(self.fixed_prompts_tensor, self.tokenized_prompts)
        
        # 3. 文本特征增强 (Self-Attention)
        # text_features shape: [N_cls, D]
        text_features = self.text_attention_module(text_features) 
        
        # 4. Final Logits
        text_features = F.normalize(text_features, p=2, dim=-1)
        logit_scale = self.logit_scale.exp()
        
        logits = logit_scale * image_features @ text_features.t()
        return logits
    
class CLIP_TextAttentionCoOp(nn.Module):
    """
    集成 CoOp (可学习提示) 到 CLIP_TextAttention。
    - 替换固定的文本嵌入为可学习的 CoOp 提示。
    - TextSelfAttention 作用于经过 CoOp 优化的特征，建模类别关系。
    """
    def __init__(self, classnames, clip_model):
        super().__init__()
        
        # 1. CLIP Encoders 和参数 (冻结或训练，取决于外部配置)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
        # 2. **核心修改：使用 CoOpPromptLearner 替换固定文本**
        # 实例化您已经定义的 CoOpPromptLearner
        self.prompt_learner = CoOpPromptLearner(classnames, clip_model) 
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        
        # 3. Text Attention Module (保持不变，作用于编码后的特征)
        # embed_dim 应该与 TextEncoder 输出特征的维度一致 (text_projection 的维度)
        embed_dim = clip_model.text_projection.shape[1]
        self.text_attention_module = TextSelfAttention(
            embed_dim=embed_dim, 
            num_layers=2, 
            heads=8
        )

    def forward(self, image):
        # 1. 图像特征编码
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = F.normalize(image_features, p=2, dim=-1)

        # 2. **生成可学习的 prompts embedding**
        # 调用 CoOpPromptLearner 的 forward 方法，获取 prompts embedding
        prompts = self.prompt_learner() # prompts shape: [N_cls, N_tkn, D]
        
        # 3. 文本特征编码 (使用可学习 prompts)
        # TextEncoder 将 prompts embedding (N_cls, N_tkn, D) 编码为最终特征 (N_cls, D)
        text_features = self.text_encoder(prompts, self.tokenized_prompts)
        
        # 4. 文本特征增强 (TextSelfAttention 作用于编码后的特征)
        # TextSelfAttention 输入 [N_cls, D] -> 输出 [N_cls, D]
        text_features = self.text_attention_module(text_features) 
        
        # 5. Final Logits
        text_features = F.normalize(text_features, p=2, dim=-1)
        logit_scale = self.logit_scale.exp()
        
        logits = logit_scale * image_features @ text_features.t()
        return logits
    
class CLIPCoOpLoRA(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__(); 
        
        # *** 注入 LoRA 适配器并冻结基座 ***
        clip_model = add_lora_to_clip(clip_model)
        
        self.prompt_learner = CoOpPromptLearner(classnames, clip_model); 
        
        # FIX: 使用新的 LoRATextEncoder
        self.text_encoder = LoRATextEncoder(clip_model) 
        
        self.image_encoder = clip_model.visual; 
        self.logit_scale = clip_model.logit_scale; 
        self.dtype = clip_model.dtype
        
    def forward(self, image):
        # FIX 1: 图像编码器调用 (直接访问 PeftModel 内部的 base_model)
        # 这解决了 VisionTransformer.forward() 的 TypeError
        visual_module = self.image_encoder.base_model
        image_features = visual_module(image.type(self.dtype)); 

        prompts = self.prompt_learner()
        
        # FIX 2: 文本编码器调用 (现在由修正后的 LoRATextEncoder.forward 负责处理)
        text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
        
        image_features = F.normalize(image_features, p=2, dim=-1); 
        text_features = F.normalize(text_features, p=2, dim=-1)
        return self.logit_scale.exp() * image_features @ text_features.t()
    
class CLIP_TextAttentionCoOp(nn.Module):
    """
    集成 CoOp (可学习提示) 到 CLIP_TextAttention。
    - 替换固定的文本嵌入为可学习的 CoOp 提示。
    - TextSelfAttention 作用于经过 CoOp 优化的特征，建模类别关系。
    """
    def __init__(self, classnames, clip_model):
        super().__init__()
        clip_model = add_lora_to_clip(clip_model)
        # 1. CLIP Encoders 和参数 (冻结或训练，取决于外部配置)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
        # 2. **核心修改：使用 CoOpPromptLearner 替换固定文本**
        # 实例化您已经定义的 CoOpPromptLearner
        self.prompt_learner = CoOpPromptLearner(classnames, clip_model) 
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        
        # 3. Text Attention Module (保持不变，作用于编码后的特征)
        # embed_dim 应该与 TextEncoder 输出特征的维度一致 (text_projection 的维度)
        embed_dim = clip_model.text_projection.shape[1]
        self.text_attention_module = TextSelfAttention(
            embed_dim=embed_dim, 
            num_layers=2, 
            heads=8
        )

    def forward(self, image):
        # 1. 图像特征编码
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = F.normalize(image_features, p=2, dim=-1)

        # 2. **生成可学习的 prompts embedding**
        # 调用 CoOpPromptLearner 的 forward 方法，获取 prompts embedding
        prompts = self.prompt_learner() # prompts shape: [N_cls, N_tkn, D]
        
        # 3. 文本特征编码 (使用可学习 prompts)
        # TextEncoder 将 prompts embedding (N_cls, N_tkn, D) 编码为最终特征 (N_cls, D)
        text_features = self.text_encoder(prompts, self.tokenized_prompts)
        
        # 4. 文本特征增强 (TextSelfAttention 作用于编码后的特征)
        # TextSelfAttention 输入 [N_cls, D] -> 输出 [N_cls, D]
        text_features = self.text_attention_module(text_features) 
        
        # 5. Final Logits
        text_features = F.normalize(text_features, p=2, dim=-1)
        logit_scale = self.logit_scale.exp()
        
        logits = logit_scale * image_features @ text_features.t()
        return logits
    
    
    
# ==============================================================================
# 1. GPU 上的批量增强模块 (BatchAugmentation)
#    保持不变
# ==============================================================================
class BatchAugmentation(nn.Module):
    def __init__(self, input_size=224):
        super().__init__()
        # 定义强增强策略 (SimCLR/MoCo 风格)
        self.aug = nn.Sequential(
            T.RandomHorizontalFlip(p=0.5),
            # 颜色抖动: 亮度、对比度、饱和度、色相
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            # T.RandomErasing(p=0.2), # 可选：随机擦除
        )

    @torch.no_grad()
    def forward(self, x):
        return self.aug(x)

# ==============================================================================
# 2. 结构化先验提示器 (StructuredPriorPrompter, SPP)
#    [核心修改]：实现了大类内互斥、大类间共现，并用阈值替代了 Top-K
# ==============================================================================
class StructuredPriorPrompter(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        
        # 参数读取
        s_reweight = getattr(cfg, 'reweight_p', 0.2) 
        t_smooth = getattr(cfg, 't_smooth', 0.07) 
        sim_threshold = getattr(cfg, 'sim_threshold', 0.05) 
        
        if hasattr(cfg, 'child_num') and cfg.child_num > 0: 
            classnames = classnames[0:cfg.child_num]
        self.n_cls = len(classnames)
        dtype = clip_model.dtype
        
        # 1. 计算 CLIP 特征
        template = "a photo of a {}."
        classnames_proc = [name.replace("_", " ") for name in classnames]
        prompts = [template.format(name) for name in classnames_proc]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        
        with torch.no_grad():
            z_static = clip_model.encode_text(tokenized_prompts).type(dtype)
        z_static = F.normalize(z_static, p=2, dim=-1)
        
        # 原始相似度 A
        A_raw = torch.matmul(z_static, z_static.t())
        
        # ======================================================================
        # 步骤 1: 必须先构建并应用互斥掩码 (Mask)
        # 目的：在归一化之前，就把 Phase vs Phase 彻底杀死，防止它们干扰 row_max 计算
        # ======================================================================
        
        structure_mask = torch.ones_like(A_raw, dtype=torch.bool)
        
        # 0-7: Phase, 7-10: View, 10-110: Action
        
        # Phase vs Phase -> 互斥 (设为0)
        structure_mask[0:7, 0:7] = False
        
        # View vs View -> 互斥 (设为0)
        structure_mask[7:10, 7:10] = False
        
        # 对角线暂时保留 (后面会单独处理 Self-loop)
        structure_mask.fill_diagonal_(True)
        
        # 应用掩码：彻底清零互斥区
        A_masked = A_raw * structure_mask.float()
        
        # ======================================================================
        # 步骤 2: 分块最大值归一化 (Block-wise Max Norm)
        # 现在的输入已经是被 Mask 干净的 A_masked 了
        # ======================================================================
        
        A_norm = torch.zeros_like(A_masked)
        ranges = [(0, 7), (7, 10), (10, 110)]
        
        for src_start, src_end in ranges:          
            for tgt_start, tgt_end in ranges:      
                
                # 提取子块
                block = A_masked[src_start:src_end, tgt_start:tgt_end]
                
                # 特殊处理：如果是 Phase-Phase 这种已经被 Mask 成全 0 的块
                # 直接跳过，保持 A_norm 里的 0
                if block.sum() == 0:
                    continue
                
                # 1. 阈值过滤
                block_thresh = torch.where(block > sim_threshold, block, torch.zeros_like(block))
                
                # 2. 计算该块每行的最大值
                block_max, _ = block_thresh.max(dim=1, keepdim=True)
                
                # 3. 归一化 (让该块最强关联变为 1.0)
                block_norm = block_thresh / (block_max + 1e-12)
                
                # 4. 填回
                A_norm[src_start:src_end, tgt_start:tgt_end] = block_norm

        # ======================================================================
        # 步骤 3: 调整对角线权重 (Self-loop)
        # ======================================================================
        
        A_final = A_norm.to(A_raw.device)
        dialog_indices = torch.eye(self.n_cls, dtype=torch.bool).to(A_raw.device)
        
        # 先清空对角线 (因为之前 block norm 可能会把对角线也变成 1)
        A_final[dialog_indices] = 0.0 
        
        # 缩放邻居权重
        A_final = A_final * (1.0 - s_reweight) 
        
        # 填回固定的自身权重
        A_final[dialog_indices] = s_reweight
        
        self.register_buffer("A_star", A_final)

    def forward(self):
        return self.A_star
class SemanticAssociationModule(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        num_layers = getattr(cfg, 'gcn_layers', 3)
        mid_features = in_features * 2 
        
        gcn_layers = nn.ModuleList()
        # Input Layer
        gcn_layers.append(GraphConvolution(in_features, mid_features))
        # Hidden Layers
        for _ in range(num_layers - 2):
            gcn_layers.append(GraphConvolution(mid_features, mid_features))
        # Output Layer
        gcn_layers.append(GraphConvolution(mid_features, out_features))
            
        self.gcn_layers = gcn_layers
        self.relu = nn.LeakyReLU(0.2)
        
    def forward(self, H0, A_star):
        H_l = H0.float()
        
        for i, layer in enumerate(self.gcn_layers):
            # 确保邻接矩阵和特征在同一设备
            A_star = A_star.to(H_l.device)
            H_l = layer(H_l, A_star)
            if i < len(self.gcn_layers) - 1:
                H_l = self.relu(H_l) 
                
        # 残差连接 (Residual Connection)
        Z_star = H0 + H_l
        return Z_star

# ==============================================================================
# 4. [最终模型] MMLSurgAdaptSCPNet
# ==============================================================================
class MMLSurgAdaptSCPNetConsistency(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        # 1. 基础组件
        # 假设 PromptLearner 是你自己定义的类，如果需要可以在这里实例化
        self.prompt_learner = PromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        # 2. SCPNet 组件 (SPP & SAM)
        self.spp = StructuredPriorPrompter(classnames, clip_model)
        # 缓存计算好的 A_star
        self.register_buffer("A_star", self.spp()) 
        
        # 获取特征维度
        try:
            feat_dim = clip_model.text_projection.shape[1] 
        except:
            feat_dim = 1024 # ResNet50 default
            
        self.sam = SemanticAssociationModule(feat_dim, feat_dim)
        
        # 3. 一致性增强组件
        self.augmentor = BatchAugmentation()

    def encode_image(self, image, visual_adapter_func=None):
        if visual_adapter_func is not None:
            image_features = self.image_encoder([image.type(self.dtype), visual_adapter_func])
        else:
            image_features = self.image_encoder(image.type(self.dtype))
        return image_features

    def forward(self, image, text_features=None):
        # ---------------------------------------------------------
        # Step 1: 图像编码
        # ---------------------------------------------------------
        image_input = image 
        image_features = self.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # ---------------------------------------------------------
        # Step 2: 文本特征编码与 GCN 优化
        # ---------------------------------------------------------
        # 使用 Prompt Learner 生成动态 Prompts
        child_prompts = self.prompt_learner()
        
        # A. 获取初始文本特征 H^0 (Z)
        Z = self.text_encoder(child_prompts, self.tokenized_prompts)
        
        # B. SAM: 使用 A* (结构化掩码后的) 精炼标签特征
        # 这里的 self.A_star 保证了互斥大类间不会传播信息
        text_features_refined = self.sam(Z, self.A_star) 
        text_features_refined = text_features_refined / text_features_refined.norm(dim=-1, keepdim=True)
        
        # ---------------------------------------------------------
        # Step 3: 计算 Logits
        # ---------------------------------------------------------
        # 缩放因子通常设为 100 或 clip_model.logit_scale.exp()
        # 这里使用你代码中的 10
        logits = 10 * image_features @ text_features_refined.t()
        
        # 返回 logits 用于主 Loss，返回 A_star 用于正则化 Loss
        return logits, self.A_star
    
class MMLSurgAdaptSCPNet(nn.Module):
    """
    与 MMLSurgAdaptSCPNetConsistency 结构保持一致，但**不包含一致性增强**（BatchAugmentation）
    并且 forward 只返回 logits（不会返回 A_star），因此上层训练时不会计算额外的 consistency loss。
    """
    def __init__(self, classnames, clip_model):
        super().__init__()
        # CLIP 组件
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        # Prompt Learner (可学习 prompt)
        self.prompt_learner = PromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        # --- 使用 SCPNet 的 SPP + SAM（与 consistency 版本一致） ---
        # Structured Prior Prompter（构建并缓存 A_star）
        self.spp = StructuredPriorPrompter(classnames, clip_model)
        # 缓存 A_star，但非一致性版本不会额外使用 augmentor 或返回 A_star
        self.register_buffer("A_star", self.spp())

        # 获取特征维度（兼容多种 clip_model）
        try:
            feat_dim = clip_model.text_projection.shape[1]
        except Exception:
            feat_dim = 1024

        # Semantic Association Module (GCN)
        self.sam = SemanticAssociationModule(feat_dim, feat_dim)

        # 注意：**不**创建 BatchAugmentation 或一致性相关组件

    def encode_image(self, image, visual_adapter_func=None):
        """
        与 consistency 版本保持一致的图像编码接口，便于复用上游代码。
        """
        if visual_adapter_func is not None:
            image_features = self.image_encoder([image.type(self.dtype), visual_adapter_func])
        else:
            image_features = self.image_encoder(image.type(self.dtype))
        return image_features

    def forward(self, image, text_features=None):
        """
        返回 logits（用于主监督 loss）。
        不返回 A_star，也不执行任何额外的一致性计算。
        """
        # 1) 图像编码
        image_input = image
        image_features = self.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # 2) 文本 Prompt ... (保持不变)
        child_prompts = self.prompt_learner()
        Z = self.text_encoder(child_prompts, self.tokenized_prompts)

        # ======================================================================
        # [防爆处理] GCN 专用归一化
        # ======================================================================
        # self.A_star 是我们要传给 Loss 的那个"数值大"的矩阵 (Max Norm)
        # 但 GCN 不能吃这个，必须吃"行和为1"的矩阵 (Row-Sum Norm)
        
        # 计算行和
        row_sum = self.A_star.sum(dim=1, keepdim=True)
        # 临时归一化 (不会修改 self.A_star 本身)
        A_gcn = self.A_star / (row_sum + 1e-12)
        
        # 3) 使用 SAM（GCN）
        # [关键] 传 A_gcn 进去，保证数值稳定
        text_features_refined = self.sam(Z, A_gcn) 
        
        text_features_refined = text_features_refined / text_features_refined.norm(dim=-1, keepdim=True)

        # 4) 计算 logits
        logits = 10 * image_features @ text_features_refined.t()

        # [关键] 返回原始的 self.A_star 给 Loss 使用
        # 这样 Soft Target 拿到的就是 0.8 这样的大数值，而不是被稀释后的 0.02
        if isinstance(self, MMLSurgAdaptSCPNetConsistency):
             return logits, self.A_star 
        else:
             return logits
