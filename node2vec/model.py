import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
import dgl.function as fn


class PinSAGEModel(nn.Module):
    def __init__(self, g, ntype, hidden_dims_dict, n_layers):
        super().__init__()
        self.proj = LinearProjector(g, ntype, hidden_dims_dict)
        self.sage = SAGENet(hidden_dims_dict, n_layers)
        self.scorer = ItemToItemScorer(g, ntype)

    def forward(self, pos_graph, neg_graph, blocks):
        h_item = self.get_repr(blocks)
        pos_score = self.scorer(pos_graph, h_item)
        neg_score = self.scorer(neg_graph, h_item)
        return pos_score, neg_score

    def get_repr(self, blocks):
        h_item = self.proj(blocks[0].srcdata)
        h_item_dst = self.proj(blocks[-1].dstdata)
        return h_item_dst + self.sage(blocks, h_item)

    
class LinearProjector(nn.Module):
    def __init__(self, g, ntype, hidden_dims_dict):
        super().__init__()
        self.hidden_dims_dict = hidden_dims_dict
        self.inputs = _init_input_modules(g, ntype, hidden_dims_dict)

    def forward(self, ndata):
        projections = []
        for feature, data in ndata.items():
            if feature == dgl.NID:
                continue

            if feature not in self.hidden_dims_dict:
                continue
            
            module = self.inputs[feature]
            result = module(data)
            projections.append(result)

#         return torch.stack(projections, 1).sum(1)
        return torch.cat(projections, dim=-1)


def _init_input_modules(g, ntype, hidden_dims_dict):
    module_dict = nn.ModuleDict()
    for column, data in g.nodes[ntype].data.items():
        if column == dgl.NID:
            continue

        if column not in hidden_dims_dict:
            print("node feat", column, "not exist")
            continue
            
        hidden_dims = hidden_dims_dict[column]
        if data.dtype == torch.float32:
            assert data.ndim == 2
            m = nn.Linear(data.shape[1], hidden_dims)
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
            module_dict[column] = m
        elif data.dtype == torch.int64:
            assert data.ndim == 1
            m = nn.Embedding(data.max() + 2, hidden_dims, padding_idx=-1)
            nn.init.xavier_uniform_(m.weight)
            module_dict[column] = m

    return module_dict


class SAGENet(nn.Module):
    def __init__(self, hidden_dims_dict, n_layers):
        super().__init__()
        self.hidden_dims_sum = sum(hidden_dims_dict.values())
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(WeightedSAGEConv(self.hidden_dims_sum, self.hidden_dims_sum*2, self.hidden_dims_sum))

    def forward(self, blocks, h):
        for layer, block in zip(self.convs, blocks):
            h_dst = h[:block.number_of_dst_nodes()]
            h = layer(block, (h, h_dst), block.edata['weights'])
        return h


class WeightedSAGEConv(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, act=F.relu):
        super().__init__()

        self.act = act
        self.Q = nn.Linear(input_dims, hidden_dims)
        self.W = nn.Linear(input_dims + hidden_dims, output_dims)
        self.reset_parameters()
        self.dropout = nn.Dropout(0.5)

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.Q.weight, gain=gain)
        nn.init.xavier_uniform_(self.W.weight, gain=gain)
        nn.init.constant_(self.Q.bias, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, g, h, weights):
        h_src, h_dst = h
        with g.local_scope():
            g.srcdata['n'] = self.act(self.Q(self.dropout(h_src)))
            g.edata['w'] = weights.float()
            g.update_all(fn.u_mul_e('n', 'w', 'm'), fn.sum('m', 'n'))
            g.update_all(fn.copy_e('w', 'm'), fn.sum('m', 'ws'))
            n = g.dstdata['n']
            ws = g.dstdata['ws'].unsqueeze(1).clamp(min=1)
            z = self.act(self.W(self.dropout(torch.cat([n / ws, h_dst], 1))))
            z_norm = z.norm(2, 1, keepdim=True)
            z_norm = torch.where(z_norm == 0, torch.tensor(1.).to(z_norm), z_norm)
            z = z / z_norm
            return z

        
class ItemToItemScorer(nn.Module):
    def __init__(self, g, ntype):
        super().__init__()
        n_nodes = g.number_of_nodes(ntype)
        self.bias = nn.Parameter(torch.zeros(n_nodes))

    def _add_bias(self, edges):
        bias_src = self.bias[edges.src[dgl.NID]]
        bias_dst = self.bias[edges.dst[dgl.NID]]
        return {'s': edges.data['s'] + bias_src + bias_dst}

    def forward(self, item_item_graph, h):
        with item_item_graph.local_scope():
            item_item_graph.ndata['h'] = h
            item_item_graph.apply_edges(fn.u_dot_v('h', 'h', 's'))
            item_item_graph.edata['s'] = item_item_graph.edata['s'].view(-1)
            item_item_graph.apply_edges(self._add_bias)
            pair_score = item_item_graph.edata['s']
        return pair_score
