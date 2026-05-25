import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import sys
# sys.path.append('./LGE/gemtrans-main-ori-final-0407-ecgsupervised-value/')
from src.core.transformer import Transformer


class SegmentGNN(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.gnn = GATConv(in_dim, in_dim)

    def forward(self, x, edge_index=None, adj_matrix=None):
        B, N, C = x.shape
        if adj_matrix is not None:
            out = []
            for b in range(B):
                h = x[b]
                adj = adj_matrix[b]
                h_new = torch.matmul(adj, h)
                out.append(h_new)
            return torch.stack(out, dim=0)
        else:
            out = [self.gnn(x[b], edge_index) for b in range(B)]
            return torch.stack(out, dim=0)


class MultiViewLGEModel(nn.Module):
    def __init__(self, hidden_dim=256, num_views=5, num_segments=16, fuse_mode='prior_GNN', loc=False):
        super().__init__()
        # fuse_mode ['prior_GNN', 'GNN', 'segment_fc', 'transformer']
        self.hidden_dim = hidden_dim
        self.num_views = num_views
        self.num_segments = num_segments
        self.fuse_mode = fuse_mode

        if fuse_mode != 'global':
            self.view_embeddings = nn.Parameter(torch.randn(num_views, hidden_dim))
            self.pos_encoding = nn.Parameter(torch.randn(1, 500, hidden_dim))  # max 500 tokens per view

            self.segment_queries = nn.Parameter(torch.randn(num_segments, hidden_dim))  # (16, D)

            self.segment_proj = nn.Linear(hidden_dim, hidden_dim)
            self.segment_gnn = SegmentGNN(in_dim=hidden_dim)
            self.dynamic_gnn = SegmentGNN(in_dim=hidden_dim)
            self.alpha = nn.Parameter(torch.tensor(0.5))
        if fuse_mode == 'transformer' or fuse_mode == 'global':
            self.reg_head = nn.Linear(hidden_dim, 16)
        else:
            self.reg_head = nn.Linear(hidden_dim, 1)

        self.edge_index = self.build_segment_graph()

        if self.fuse_mode == 'transformer':
            self.transformer = Transformer(
                num_layers=2,
                dim=768,
                num_heads=16,
                ff_dim=512,
                dropout=0.3,
                last_layer_attn=False,
                aggr_method='cls',
                return_full_attn=False,
            )
            self.layer_norm = nn.LayerNorm(768)

        self.loc = loc
        if self.loc:
            if self.fuse_mode == 'transformer' or self.fuse_mode == 'global':
                self.loc_mlp = nn.Sequential(nn.Linear(hidden_dim, 16))
            else:
                self.loc_mlp = nn.Sequential(nn.Linear(hidden_dim, 1))

    # def build_segment_graph(self):
    #     # 医学先验连接（简化环状结构）
    #     edge_list = []
    #     for i in range(6):
    #         edge_list.append((i, (i+1)%6))
    #         edge_list.append(((i+1)%6, i))
    #     for i in range(6,12):
    #         edge_list.append((i, (i+1)%6+6))
    #         edge_list.append(((i+1)%6+6, i))
    #     for i in range(12,16):
    #         edge_list.append((i, 12 + (i+1)%4))
    #         edge_list.append((12 + (i+1)%4, i))
    #     return torch.tensor(edge_list).t().contiguous()
    def build_segment_graph(self):
        edge_list = []
        for i in range(16):
            edge_list.append((i, i))  # 添加自环
        for i in range(6):
            edge_list.append((i, (i+1)%6))
            edge_list.append(((i+1)%6, i))
        for i in range(6,12):
            edge_list.append((i, (i+1)%6+6))
            edge_list.append(((i+1)%6+6, i))
        for i in range(12,16):
            edge_list.append((i, 12 + (i+1)%4))
            edge_list.append((12 + (i+1)%4, i))
        return torch.tensor(edge_list).t().contiguous()


    def compute_learned_graph(self, segment_feats):
        norm_feats = F.normalize(segment_feats, dim=-1)  ##### 20251109改进版本进行L2归一化

        sim = torch.matmul(norm_feats, norm_feats.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        adj = F.softmax(sim, dim=-1)
        return adj
    
    # def compute_learned_graph(self, segment_feats):
    #     sim = torch.matmul(segment_feats, segment_feats.transpose(1, 2)) / (self.hidden_dim ** 0.5)
    #     sim = sim + torch.eye(sim.size(-1), device=sim.device).unsqueeze(0)  # 添加自环
    #     adj = F.softmax(sim, dim=-1)
    #     return adj


    def forward(self, views, masks=None):

        if self.fuse_mode != 'global':
            views = views.permute(1,0, 2, 3)
            all_tokens = []
            for i, tokens in enumerate(views):
                B, N, D = tokens.shape

                view_embed = self.view_embeddings[i].unsqueeze(0).unsqueeze(1)  # (1, 1, D)
                view_embed = view_embed.expand(B, N, D)

                pos_embed = self.pos_encoding[:, :N, :]  # (1, N, D)
                pos_embed = pos_embed.expand(B, N, D)

                tokens = tokens + view_embed + pos_embed  # 注入视图 & 位置信息
                all_tokens.append(tokens)

            patch_tokens = torch.cat(all_tokens, dim=1)  # (B, N_total, D)


            if self.fuse_mode != 'transformer' and self.fuse_mode != 'global':
                # === Segment Query Attention ===
                segment_queries = self.segment_queries.unsqueeze(0).expand(patch_tokens.size(0), -1, -1)  # (B, 16, D)
                attn_scores = torch.matmul(segment_queries, patch_tokens.transpose(1, 2)) / (self.hidden_dim ** 0.5)  # (B, 16, N)
                attn_weights = F.softmax(attn_scores, dim=-1)
                segment_feats = torch.matmul(attn_weights, patch_tokens)  # (B, 16, D)
                segment_feats = self.segment_proj(segment_feats)

                # === 图结构建模 医学先验+数据驱动===
                if self.fuse_mode == 'prior_GNN': 
                    learned_adj = self.compute_learned_graph(segment_feats)
                    static_out = self.segment_gnn(segment_feats, self.edge_index.to(segment_feats.device))
                    learned_out = self.dynamic_gnn(segment_feats, adj_matrix=learned_adj)

                    fused_feats = self.alpha * static_out + (1 - self.alpha) * learned_out
                    reg_preds = self.reg_head(fused_feats).squeeze(-1)  # (B, 16)
                    if self.loc:
                        loc_preds = self.loc_mlp(fused_feats).squeeze(-1)
                    else:
                        loc_preds = None

                # === 图结构建模 数据驱动===
                if self.fuse_mode == 'GNN': 
                    learned_adj = self.compute_learned_graph(segment_feats)
                    learned_out = self.dynamic_gnn(segment_feats, adj_matrix=learned_adj)
                    reg_preds = self.reg_head(learned_out).squeeze(-1)  # (B, 16)
                    if self.loc:
                        loc_preds = self.loc_mlp(learned_out).squeeze(-1)
                    else:
                        loc_preds = None

                # === 去掉图结构===
                if self.fuse_mode == 'segment_fc': 
                    reg_preds = self.reg_head(segment_feats).squeeze(-1)
                    if self.loc:
                        loc_preds = self.loc_mlp(segment_feats).squeeze(-1)
                    else:
                        loc_preds = None
                return reg_preds, loc_preds, attn_weights 

            # === 去掉query 直接 transformer建模所有patch===
            if self.fuse_mode == 'transformer':

                fea, _, _ = self.transformer(patch_tokens, masks)
                fea = self.layer_norm(fea)
                fea_cls = fea.mean(dim=1)
                reg_preds = self.reg_head(fea_cls).squeeze(-1)
                if self.loc:
                    loc_preds = self.loc_mlp(fea_cls).squeeze(-1)
                else:
                    loc_preds = None
                # print(reg_preds.shape, loc_preds.shape) 
                return reg_preds, loc_preds, loc_preds
        else:
            reg_preds = self.reg_head(views).squeeze(-1)
            if self.loc:
                loc_preds = self.loc_mlp(views).squeeze(-1)
            else:
                loc_preds = None
            return reg_preds, loc_preds, loc_preds

        


# ====== 模拟输入调用示例 ======
if __name__ == "__main__":
    B, N, D = 2, 393, 768  # batch, tokens per view, dim
    views = [torch.randn(B, N, D) for _ in range(5)]
    model = MultiViewLGEModel(hidden_dim=D, num_views=5)
    reg_preds = model(views)
    print("16段回归输出:", reg_preds.shape)  # [B, 16]
