import torch
import torch.nn as nn
import numpy as np
from src.core.vit.vit import ViT
from src.core.transformer import Transformer
import math
import yaml
import torch.nn.functional as F


import sys
from transformers.modeling_outputs import BaseModelOutput
from typing import Optional, Tuple, Union
from hubert_ecg import HuBERTECG, HuBERTECGConfig

class ActivationFunction(nn.Module):
    def __init__(self, activation : str):
        super(ActivationFunction, self).__init__()
        self.activation = activation
        
        if activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            raise ValueError('Activation function not supported')
    
    def forward(self, x):
        return self.act(x)

class HuBERTForECGClassification(nn.Module):

    def __init__(
        self,
        hubert_ecg : HuBERTECG,
        num_labels : int,
        classifier_hidden_size : int = None,
        activation : str = 'tanh',
        use_label_embedding : bool = False,
        classifier_dropout_prob : float = 0.1):
        super(HuBERTForECGClassification, self).__init__()
        self.hubert_ecg = hubert_ecg
        self.hubert_ecg.config.mask_time_prob = 0.0 # as we load pre-trained models that used to mask inputs, resetting masking probs prevents masking
        self.hubert_ecg.config.mask_feature_prob = 0.0 # as we load pre-trained models that used to mask inputs, resetting masking probs prevents masking
        
        self.num_labels = num_labels
        self.config = self.hubert_ecg.config
        self.classifier_hidden_size = classifier_hidden_size
        self.activation = ActivationFunction(activation)
        self.use_label_embedding = use_label_embedding 
        self.classifier_dropout = nn.Dropout(classifier_dropout_prob)
        
        del self.hubert_ecg.label_embedding # not needed
        del self.hubert_ecg.final_proj # not needed
        
        if use_label_embedding: # for classification only
            self.label_embedding = nn.Embedding(num_labels, self.config.hidden_size) 
        else:
            if classifier_hidden_size is None: # no hidden layer
                self.classifier = nn.Linear(self.config.hidden_size, num_labels)
            else:
                self.classifier = nn.Sequential(
                    nn.Linear(self.config.hidden_size, classifier_hidden_size),
                    self.activation,
                    nn.Linear(classifier_hidden_size, num_labels)
                )
        
    def set_feature_extractor_trainable(self, trainable : bool):
        '''Sets as (un)trainable the convolutional feature extractor of HuBERT-ECG'''
        self.hubert_ecg.feature_extractor.requires_grad_(trainable)
    
    def set_transformer_blocks_trainable(self, n_blocks : int):
        ''' Makes trainable only the last `n_blocks` of HuBERT-ECG transformer encoder'''
        
        assert n_blocks >= 0, f"n_blocks (inserted {n_blocks}) should be >= 0"
        assert n_blocks <= self.hubert_ecg.config.num_hidden_layers, f"n_blocks ({n_blocks}) should be <= {self.hubert_ecg.config.num_hidden_layers}"
        
        self.hubert_ecg.encoder.requires_grad_(False)
        for i in range(1, n_blocks+1):
            self.hubert_ecg.encoder.layers[-i].requires_grad_(True)
                
    def get_logits(self, pooled_output : torch.Tensor):
        '''Computes cosine similary between transfomer pooled output, referred to as input representation, and look-up embedding matrix, that is a dense representation of labels.
        In: pooled_output: (B, C) tensor
        Out: (B, num_labels) tensor of similarities/logits to be sigmoided and used in BCE loss
        '''
        logits = torch.cosine_similarity(pooled_output.unsqueeze(1), self.label_embedding.weight.unsqueeze(0), dim=-1)
        return logits
            
    def forward(
        self,
        x: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Union[Tuple, BaseModelOutput]]:
        
        return_dict = return_dict if return_dict is not None else self.hubert_ecg.config.use_return_dict
        output_hidden_states = True if self.hubert_ecg.config.use_weighted_layer_sum else output_hidden_states
               
        encodings = self.hubert_ecg(
                x,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
        
        x = encodings.last_hidden_state
        ecg_attentions = encodings.attentions
        # for attention in ecg_attentions:
        #     print(attention.shape)
        if attention_mask is None:
            x = x.mean(dim=1)
        else:
            padding_mask = self.hubert_ecg._get_feature_vector_attention_mask(x.shape[1], attention_mask)
            x[~padding_mask] = 0.0
            x = x.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)
            
        x = self.classifier_dropout(x) 
        # output = (
        #     self.get_logits(x) if self.use_label_embedding else self.classifier(x),
        #     encodings
        # )
        out = self.classifier(x)

        return out, x, ecg_attentions

class my_ecg_config:
    def __init__(self):
        self.load_path = './models/pretrained/hubert_ecg_small.pt'
        self.vocab_size = 2
        self.classifier_hidden_size = None
        self.use_label_embedding = False
        self.freezing_steps = None
        self.transformer_blocks_to_unfreeze = 8
        self.finetuning_layerdrop = 0.0
        self.model_dropout_mult = -1
        self.downsampling_factor = 5
    
EPS = 1e-9
MINIMAL_IMPROVEMENT = 1e-3
SUPERVISED_MODEL_CKPT_PATH = "./models/pretrained/"
DROPOUT_DYNAMIC_REG_FACTOR = 0.05

class SpaceTimeFactorizedViViT_ECG(nn.Module):
    def __init__(
        self,
        mode,
        patches,
        spatial_dropout_rate,
        spatial_hidden_size,
        spatial_num_layers,
        spatial_mlp_dim,
        spatial_num_heads,
        spatial_aggr_method,
        n_sampled_frames,
        temporal_dropout_rate,
        temporal_hidden_size,
        temporal_num_layers,
        temporal_mlp_dim,
        temporal_num_heads,
        temporal_aggr_method,
        vid_seq_len,
        vid_hidden_size,
        vid_dropout_rate,
        vid_num_layers,
        vid_mlp_dim,
        vid_num_heads,
        vid_aggr_method,
        output_dropout_rate=0.0,
        use_seg_labels=False,
        use_ed_es_locs=False,
        pretrained_patch_encoder_path=None,
        return_full_attn=False,
        use_classification_head=False,
        frame_size=224,
        use_ppnet=False,
        backbone=None,
        backbone_name=None
    ):

        super(SpaceTimeFactorizedViViT_ECG, self).__init__()

        

        n_temporal_tokens = n_sampled_frames
        if temporal_aggr_method == "cls":
            self.temporal_cls_token = nn.Parameter(
                torch.zeros((1, 1, temporal_hidden_size))
            )
            n_temporal_tokens = n_sampled_frames + 1
        n_view_tokens = vid_seq_len
        if vid_aggr_method == "cls":
            self.vid_cls_token = nn.Parameter(torch.zeros((1, 1, vid_hidden_size)))
            n_view_tokens = vid_seq_len + 1
        
        if vid_aggr_method == 'multi_cls':
            self.vid_cls_token = nn.Parameter(torch.zeros((1, 3, vid_hidden_size)))
            n_view_tokens = vid_seq_len + 3



        # If ViT's hidden dim doesn't match that of temporal transformer, we need to project
        self.ste_to_tte = None
        if spatial_hidden_size != temporal_hidden_size:
            self.ste_to_tte = nn.Linear(spatial_hidden_size, temporal_hidden_size)


        self.tte_to_vte = None
        if temporal_hidden_size != vid_hidden_size:
            self.tte_to_vte = nn.Linear(temporal_hidden_size, vid_hidden_size)
        # n_view_tokens = 2

        self.vid_trans_enc = Encoder(
            num_layers=vid_num_layers,
            mlp_dim=vid_mlp_dim,
            dropout_rate=vid_dropout_rate,
            hidden_size=vid_hidden_size,
            num_heads=vid_num_heads,
            seq_len=n_view_tokens,
            aggr_method=vid_aggr_method,
            last_layer_attn=False,
            return_full_attn=return_full_attn,
        )

        self.use_ppnet = use_ppnet
        if use_ppnet:
            # img_size = 224
            # prototype_shape = [32, 192, 1, 1]
            # num_classes = 4
            # reserve_layers = [11]
            # reserve_token_nums = [81]
            # proto_layer_rf_info = [14, 16, 16, 8.0]
            # use_global = True
            # use_ppc_loss = True
            # ppc_cov_thresh = 1
            # ppc_mean_thresh = 2
            # global_coe = 0.5
            # global_proto_per_class = 4
            # prototype_activation_function = "log"
            # add_on_layers_type = "regular"
            img_size = 128
            prototype_shape = [32, 192, 1, 1]
            num_classes = 1
            reserve_layers = [11]
            reserve_token_nums = [25]
            proto_layer_rf_info = [14, 16, 16, 8.0]
            use_global = True
            use_ppc_loss = True
            ppc_cov_thresh = 1
            ppc_mean_thresh = 2
            global_coe = 0.5
            global_proto_per_class = 1
            prototype_activation_function = "log"
            add_on_layers_type = "regular"

            self.prototype_layer = PPNet(
                features=self.spatial_trans_enc,
                img_size=img_size,
                prototype_shape=prototype_shape,
                proto_layer_rf_info=proto_layer_rf_info,
                num_classes=num_classes,
                reserve_layers=reserve_layers,
                reserve_token_nums=reserve_token_nums,
                use_global=use_global,
                use_ppc_loss=use_ppc_loss,
                ppc_cov_thresh=ppc_cov_thresh,
                ppc_mean_thresh=ppc_mean_thresh,
                global_coe=global_coe,
                global_proto_per_class=global_proto_per_class,
                init_weights=True,
                prototype_activation_function=prototype_activation_function,
                add_on_layers_type=add_on_layers_type,
            )

        if mode == "ef":
            self.output_mlp = nn.Sequential(
                nn.Linear(vid_hidden_size, vid_hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(p=output_dropout_rate),
                nn.Linear(vid_hidden_size // 2, vid_hidden_size // 4),
                nn.ReLU(inplace=True),
                nn.Dropout(p=output_dropout_rate),
                nn.Linear(vid_hidden_size // 4, 1),
                nn.Sigmoid(),
            )
            # set bias to 0.552
            self.output_mlp[-2].bias.data[0] = 0.552

            self.class_output_mlp = None
            if use_classification_head:
                self.class_output_mlp = nn.Sequential(
                    nn.Linear(
                        in_features=vid_hidden_size, out_features=vid_hidden_size // 2
                    ),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=output_dropout_rate),
                    nn.Linear(in_features=vid_hidden_size // 2, out_features=4),
                )

        elif mode == "as":
            self.output_mlp = nn.Sequential(
                nn.Linear(vid_hidden_size, vid_hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(p=output_dropout_rate),
                nn.Linear(vid_hidden_size // 2, vid_hidden_size // 4),
                nn.ReLU(inplace=True),
                nn.Dropout(p=output_dropout_rate),
                nn.Linear(vid_hidden_size // 4, 4),
            )

            self.class_output_mlp = None

        elif mode == "lge":
            self.output_mlp = nn.Sequential(
                nn.Linear(vid_hidden_size, vid_hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(p=output_dropout_rate),
                nn.Linear(vid_hidden_size // 2, vid_hidden_size // 4),
                nn.ReLU(inplace=True),
                nn.Dropout(p=output_dropout_rate),
                nn.Linear(vid_hidden_size // 4, 1),
            )

            self.output_mlp_perfusion = nn.Sequential(
                nn.Linear(vid_hidden_size, vid_hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(p=output_dropout_rate),
                nn.Linear(vid_hidden_size // 2, vid_hidden_size // 4),
                nn.ReLU(inplace=True),
                nn.Dropout(p=output_dropout_rate),
                nn.Linear(vid_hidden_size // 4, 1),
            )

            self.output_mlp_value = nn.Sequential(
                nn.Linear(vid_hidden_size, vid_hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(p=output_dropout_rate),
                nn.Linear(vid_hidden_size // 2, vid_hidden_size // 4),
                nn.ReLU(inplace=True),
                nn.Dropout(p=output_dropout_rate),
                nn.Linear(vid_hidden_size // 4, 16),
            )

            self.class_output_mlp = None
        elif mode == "pretrain":
            self.output_mlp = nn.Sequential(
                nn.Linear(vid_hidden_size, vid_hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(p=output_dropout_rate),
                nn.Linear(vid_hidden_size // 2, vid_hidden_size // 4),
                nn.ReLU(inplace=True),
                nn.Dropout(p=output_dropout_rate),
                nn.Linear(vid_hidden_size // 4, 400),
            )

            self.class_output_mlp = None
        else:
            self.output_mlp = None
            self.class_output_mlp = None

        self.mode = mode
        self.spatial_aggr_method = spatial_aggr_method
        self.temporal_aggr_method = temporal_aggr_method
        self.vid_aggr_method = vid_aggr_method
        self.use_ed_es_locs = use_ed_es_locs
        self.use_pretrained_patch_encoder = pretrained_patch_encoder_path is not None
        self.n_sampled_frames = n_sampled_frames
        self.return_full_attn = return_full_attn
        # ecg model
        self.ecg_args = my_ecg_config()
        checkpoint = torch.load(self.ecg_args.load_path, map_location = 'cpu')
        config = HuBERTECGConfig(**checkpoint['model_config'].to_dict())
        config.layerdrop = self.ecg_args.finetuning_layerdrop

        pretrained_hubert = HuBERTECG(config)
        
        # restore original p-dropout or set multipliers
        for name, module in pretrained_hubert.named_modules():
            if 'dropout' in name:
                module.p = 0.1 + DROPOUT_DYNAMIC_REG_FACTOR * self.ecg_args.model_dropout_mult
        
        self.ecg_model = HuBERTForECGClassification(pretrained_hubert, num_labels=self.ecg_args.vocab_size, classifier_hidden_size=self.ecg_args.classifier_hidden_size,  use_label_embedding=self.ecg_args.use_label_embedding)
        self.ecg_model.hubert_ecg.load_state_dict(checkpoint['model_state_dict'], strict=False) # load backbone weights
        self.concat_ecg = nn.Linear(512, 200)

        self.video_backbone = backbone
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # self.lge_value_mlp = nn.Sequential(
        #         nn.Linear(vid_hidden_size, vid_hidden_size // 2),
        #         #nn.ReLU(inplace=True),
        #         nn.Dropout(p=output_dropout_rate),
        #         nn.Linear(vid_hidden_size // 2, vid_hidden_size // 4),
        #         #nn.ReLU(inplace=True),
        #         nn.Dropout(p=output_dropout_rate),
        #         nn.Linear(vid_hidden_size // 4, 16),
        #     )


    def forward(self, data_dict):
        ecg = data_dict["ecg_data"]
        ecg_mask = data_dict["ecg_mask"]
        x = data_dict["vid"]
        mask = data_dict["mask"]
        ecg_concat_mask = data_dict["ecg_concat_mask"]
        ed_frames = data_dict["ed_frame"]
        ed_valid = data_dict["ed_valid"]
        es_frames = data_dict["es_frame"]
        es_valid = data_dict["es_valid"]
        #label = data_dict["label"] if data_dict["label"].dtype is torch.long else None
        label = data_dict["label"]
        ecg_out, ecg_embed, ecg_attn = self.ecg_model(ecg, attention_mask=ecg_mask, output_attentions=False, output_hidden_states=False, return_dict=True)
        num_frames = x.shape[2]
        # x, mask = self.temporal_encoder(x, mask)
        # Proto-related variables
        logits = None
        auxi_item = None
        ppc_loss = None

        # Return sampled video for visualization purposes
        with torch.no_grad():
            sampled_vid = x.detach() if self.return_full_attn else None

        # The frames are subsampled now, modify ED/ES locations accordingly
        if self.use_ed_es_locs and ed_valid is not None:
            ed_valid, es_valid, ed_frames, es_frames = sample_ed_es(
                ed_frames,
                ed_valid,
                es_frames,
                es_valid,
                num_frames,
                self.n_sampled_frames,
            )

        bs, nvid, t, h, w, c = x.shape
        x = x.permute(0, 1, 5, 2, 3, 4)
        x = x.contiguous().view(
            x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4], x.shape[5]
        )
        x, x_seg = self.video_backbone(x)
        if len(x.shape) == 2:
            x = x.view(bs, nvid, -1)
        else:
            x = self.avg_pool(x).view(bs, nvid, -1)

        if self.tte_to_vte is not None:
            x = self.tte_to_vte(x)

        # Add class tokens
        n, _, c = x.shape
        vid_mask = torch.any(mask, dim=2)
        if self.vid_aggr_method == "cls":
            vid_cls_tokens = torch.tile(self.vid_cls_token, [n, 1, 1])
            x = torch.cat((vid_cls_tokens, x), dim=1)
            vid_mask = torch.cat(
                (torch.ones((n, 1), dtype=torch.bool, device=mask.device), vid_mask),
                dim=1,
            )
        if self.vid_aggr_method == "multi_cls":
            vid_cls_tokens = torch.tile(self.vid_cls_token, [n, 1, 1])
            x = torch.cat((vid_cls_tokens, x), dim=1)
            vid_mask = torch.cat(
                (torch.ones((n, 3), dtype=torch.bool, device=mask.device), vid_mask),
                dim=1,
            )
            # print(vid_cls_tokens.shape, x.shape, vid_mask.shape)
        # concat ecg
        ecg_embed = ecg_embed.unsqueeze(1)
        ecg_embed = self.concat_ecg(ecg_embed)
        x = torch.cat((x, ecg_embed), dim=1)
        # vid_mask = torch.cat(
        #         (vid_mask, torch.ones((n, 1), dtype=torch.bool, device=mask.device)),
        #         dim=1,
        #     )
        # print(vid_mask, ecg_concat_mask)
        vid_mask = torch.cat((vid_mask, ecg_concat_mask),dim=1)


        # Video encoder
        # x[torch.bitwise_not(vid_mask)] = 0
        x, vid_pos_embed, vid_attn, _ = self.vid_trans_enc(x, vid_mask)

        if self.vid_aggr_method == "cls":
            x = x[:, 0]
        elif self.vid_aggr_method == "mean":
            sums = torch.sum(vid_mask, dim=1, keepdim=True)
            x = torch.sum(x * vid_mask.unsqueeze(-1), dim=1) / sums
        elif self.vid_aggr_method == "max":
            x = torch.nan_to_num(x)
            x = x.max(dim=1).values
        elif self.vid_aggr_method == 'multi_cls':
            x_lge = x[:, 0]
            x_perfusion = x[:, 1]
            x_value = x[:, 2]

        x_class = None
        if self.class_output_mlp is not None:
            x_class = self.class_output_mlp(x)

        if self.output_mlp is not None:
            if self.vid_aggr_method != 'multi_cls':
                out_lge = self.output_mlp(x)
                out_perfusion = self.output_mlp_perfusion(x)
                out_value = self.output_mlp_value(x)
            else:
                out_lge = self.output_mlp(x_lge)
                out_perfusion = self.output_mlp_perfusion(x_perfusion)
                out_value = self.output_mlp_value(x_value)

        
            
        return {
            "x": out_lge,
            "x_seg":x_seg,
            "lge_value":out_value,
            "perfusion":out_perfusion,
            "patch_pos_embed": None,
            "frame_pos_embed": None,
            "vid_pos_embed": vid_pos_embed,
            "patch_attn": None,
            "frame_attn": None,
            "vid_attn": vid_attn,
            "ecg_attn": ecg_attn[-1],
            "last_layer_patch_attn": None,
            "last_layer_frame_attn": None,
            "ed_valid": ed_valid,
            "es_valid": es_valid,
            "ed_frames": ed_frames,
            "es_frames": es_frames,
            "sampled_vid": sampled_vid,
            "x_class": x_class,
            "logits": logits,
            "ppc_loss": ppc_loss,
        }

    def extract_windows(
        self, x, bs, hidden_dim, n_patches, n_frames, windows_per_frame
    ):

        x = (
            x.permute(0, 2, 1)
            .contiguous()
            .view(
                bs * n_frames,
                hidden_dim,
                int(math.sqrt(n_patches)),
                int(math.sqrt(n_patches)),
            )
        )

        # Extract windows
        x = (
            x.unfold(1, hidden_dim, hidden_dim)
            .unfold(2, self.cross_attn_window, self.cross_attn_window)
            .unfold(3, self.cross_attn_window, self.cross_attn_window)
        )

        unfold_shape = x.size()

        x = x.contiguous().view(
            bs * n_frames,
            -1,
            hidden_dim,
            self.cross_attn_window,
            self.cross_attn_window,
        )

        x = x.contiguous().view(
            bs, n_frames, windows_per_frame, hidden_dim, self.cross_attn_window**2
        )
        x = x.permute(0, 2, 1, 4, 3)
        x = x.contiguous().view(
            bs * windows_per_frame, n_frames * self.cross_attn_window**2, hidden_dim
        )

        return x, unfold_shape

    def recons_frame(
        self, x, unfold_shape, bs, windows_per_frame, n_patches, n_frames, hidden_dim
    ):
        x = x.contiguous().view(
            bs, windows_per_frame, n_frames, self.cross_attn_window**2, hidden_dim
        )
        x = x.permute(0, 2, 1, 4, 3)
        x = x.contiguous().view(
            bs * n_frames,
            windows_per_frame,
            hidden_dim,
            self.cross_attn_window,
            self.cross_attn_window,
        )

        x = x.contiguous().view(unfold_shape)
        output_c = unfold_shape[1] * unfold_shape[4]
        output_h = unfold_shape[2] * unfold_shape[5]
        output_w = unfold_shape[3] * unfold_shape[6]
        x = x.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        x = x.view(-1, output_c, output_h, output_w)

        x = x.contiguous().view(bs * n_frames, hidden_dim, n_patches).permute(0, 2, 1)

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers,
        mlp_dim,
        dropout_rate,
        hidden_size,
        num_heads,
        seq_len,
        aggr_method="cls",
        last_layer_attn=False,
        return_full_attn=False,
    ):
        super(Encoder, self).__init__()

        self.positional_embedder = PositionEmbs(seq_len, hidden_size, dropout_rate)

        # self.pre_logits = nn.Linear(hidden_size, repr_dim)

        self.layer_norm = nn.LayerNorm(hidden_size)

        self.transformer = Transformer(
            num_layers=num_layers,
            dim=hidden_size,
            num_heads=num_heads,
            ff_dim=mlp_dim,
            dropout=dropout_rate,
            last_layer_attn=last_layer_attn,
            aggr_method=aggr_method,
            return_full_attn=return_full_attn,
        )

        # Initialize weights
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(
                    m.weight
                )  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)

        self.apply(_init)
        nn.init.normal_(
            self.positional_embedder.pos_emb, std=0.02
        )  # _trunc_normal(self.positional_embedding.pos_embedding, std=0.02)

    def forward(self, x, mask):
        x = self.positional_embedder(x)

        x, debug_attn, last_layer_attn = self.transformer(x, mask)

        x = self.layer_norm(x)  # b,d

        return x, self.positional_embedder.pos_emb.data, debug_attn, last_layer_attn


class PositionEmbs(nn.Module):
    def __init__(self, seq_len, hidden_size, dropout_rate):
        super(PositionEmbs, self).__init__()

        self.pos_emb = nn.Parameter(
            torch.zeros(1, seq_len, hidden_size), requires_grad=True
        )
        nn.init.trunc_normal_(self.pos_emb, std=0.2)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x + self.pos_emb
        return self.dropout(x)


class TemporalEncoder(nn.Module):
    def __init__(
        self, n_sampled_frames, patches, hidden_size, use_pretrained_patch_encoder
    ):

        super(TemporalEncoder, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=hidden_size,
            kernel_size=(patches[0], patches[1]),
            stride=(patches[0], patches[1]),
        )

        self.n_sampled_frames = n_sampled_frames
        self.patches = patches
        self.hidden_size = hidden_size
        self.use_pretrained_patch_encoder = use_pretrained_patch_encoder

    def forward(self, x, mask):

        # Choose frames
        x = sample_frames_uniformly(x, self.n_sampled_frames)
        mask = sample_frames_uniformly(mask, self.n_sampled_frames)
        bs, n, ts, in_h, in_w, c = x.shape

        if self.use_pretrained_patch_encoder:
            x = x.expand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4], 3)
        else:
            # Change to channels-second format
            x = x.permute(0, 5, 1, 2, 3, 4)

            # Reshape to an elongated frame
            x = x.contiguous().view(bs, c, n * ts * in_h, in_w)

            # Embed the patches
            x = self.conv(x)
            bs, c, nth, w = x.shape
            x = x.permute(0, 2, 3, 1)
            x = x.contiguous().view(bs, n, ts, -1, w, c)
        return x, mask


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    def __init__(self, in_dim, out_dim, hidden_size, dropout_rate):
        super(MlpBlock, self).__init__()

        self.dense1 = nn.Linear(in_dim, hidden_size)
        self.activation_func = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.dense2 = nn.Linear(hidden_size, out_dim)

    def forward(self, x):
        x = self.dense1(x)
        x = self.activation_func(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)

        return x


def sample_frames_uniformly(x, n_sampled_frames):
    num_frames = x.shape[2]
    temporal_indices = np.linspace(
        start=0, stop=num_frames, num=n_sampled_frames, endpoint=False, dtype=np.int32
    )
    return x[:, :, temporal_indices]


def sample_ed_es(
    ed_frames, ed_valid, es_frames, es_valid, num_frames, n_sampled_frames
):
    ed_valid = ed_valid.flatten()
    es_valid = es_valid.flatten()
    ed_frames = ed_frames.flatten()
    es_frames = es_frames.flatten()

    temporal_indices = np.linspace(
        start=0, stop=num_frames, num=n_sampled_frames, endpoint=False, dtype=np.int32
    )

    for i in range(ed_valid.shape[0]):
        if ed_valid[i]:
            if ed_frames[i].item() not in temporal_indices:
                ed_valid[i] = False
            else:
                ed_frames[i] = np.where(temporal_indices == ed_frames[i].item())[
                    0
                ].item()

        if es_valid[i]:
            if es_frames[i].item() not in temporal_indices:
                es_valid[i] = False
            else:
                es_frames[i] = np.where(temporal_indices == es_frames[i].item())[
                    0
                ].item()

    return ed_valid, es_valid, ed_frames, es_frames



