import numpy as np
import pandas as pd

import torch
from torch import nn, Tensor, LongTensor, tensor
import torch.nn.functional as F

from transformers import AutoConfig, AutoModel

import dgl
from dgl.nn import GATConv


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class GNN_2_Model(nn.Module):

    @staticmethod
    def get_all_pair(ar):
        return np.array(np.meshgrid(ar, ar)).T.reshape(-1, 2)

    @staticmethod
    def get_edges(r, c, ret_extra_edges=False, ret_caption_edges=False):
        edges = np.empty((0, 2), dtype=int)
        row_edges = GNN_2_Model.get_all_pair(np.arange(c))
        for i in range(r):
            edges = np.concatenate((edges, row_edges + i * c), axis=0)
        col_edges = GNN_2_Model.get_all_pair(np.arange(0, r * c, c))
        for i in range(c):
            edges = np.concatenate((edges, col_edges + i), axis=0)
        edges = np.unique(edges, axis=0)
        table_edges = LongTensor(edges[np.lexsort((edges[:, 1], edges[:, 0]))])
        if ret_extra_edges:
            table_cells = torch.arange(r * c)
            row_edges = torch.stack([table_cells, r * c + table_cells // c]).T
            col_edges = torch.stack([table_cells, r * c + r + table_cells % c]).T
            row_self_edges = torch.stack([r * c + torch.arange(r), r * c + torch.arange(r)]).T
            col_self_edges = torch.stack([r * c + r + torch.arange(c), r * c + r + torch.arange(c)]).T
            if not ret_caption_edges:
                return table_edges, torch.cat([row_edges, col_edges, row_self_edges, col_self_edges])
            row_col_edges = torch.cat([row_edges, col_edges, torch.flip(row_edges, (1, )), torch.flip(col_edges, (1, )), row_self_edges, col_self_edges])
            caption_edges = torch.stack([(r * c + r + c) * torch.ones(r + c + 1).long(), r * c + torch.arange(r + c + 1)]).T
            return table_edges, torch.cat([row_col_edges, caption_edges])
        return table_edges

    @staticmethod
    def create_edge_labels(all_edges, edge_list):
        df = pd.DataFrame(all_edges.tolist())
        df['merge'] = list(zip(df[0], df[1]))
        edge_labels = LongTensor(df['merge'].isin(edge_list))
        return edge_labels

    @staticmethod
    def get_all_pairs_torch(n, ordered=False):
        if ordered:
            return torch.cat([torch.combinations(torch.arange(n)), torch.combinations(torch.arange(n-1, -1, -1))])
        return torch.combinations(torch.arange(n))

    @staticmethod
    def get_block(h_in, h_out):
        return nn.Sequential(
            nn.Linear(h_in, h_out),
            nn.BatchNorm1d(h_out),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    @staticmethod
    def validate_args(args):
        assert isinstance(args['hidden_layer_sizes'], list)
        assert isinstance(args['num_heads'], list)
        assert len(args['hidden_layer_sizes']) == len(args['num_heads'])
        assert isinstance(args['use_caption'], bool)
        assert isinstance(args['add_constraint'], bool)
        assert isinstance(args['use_max_freq_feat'], bool)
        assert isinstance(args['max_freq_emb_size'], int)
        return args

    def __init__(self, args: dict):
        super(GNN_2_Model, self).__init__()
        self.args = self.validate_args(args)

        config = AutoConfig.from_pretrained(args['lm_name'], cache_dir=args['cache_dir'])
        self.encoder = AutoModel.from_pretrained(args['lm_name'], config=config, cache_dir=args['cache_dir'])

        in_dim = config.hidden_size
        self.default_embedding = nn.Embedding(1, in_dim)
        self.positional_embeddings = nn.Embedding(4, config.hidden_size)

        if self.args['use_max_freq_feat']:
            self.max_freq_feat_embedding = nn.Embedding(6, self.args['max_freq_emb_size'])
            in_dim += self.args['max_freq_emb_size']

        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(
            GATConv(in_dim, self.args['hidden_layer_sizes'][0], num_heads=self.args['num_heads'][0], residual=False))

        for l in range(1, len(self.args['hidden_layer_sizes'])):
            self.gat_layers.append(
                GATConv(self.args['hidden_layer_sizes'][l-1] * self.args['num_heads'][l-1], self.args['hidden_layer_sizes'][l], \
                num_heads=self.args['num_heads'][l], residual=True))

        out_dim = self.args['hidden_layer_sizes'][-1] * self.args['num_heads'][-1]
        self.dropout = nn.Dropout(0.2)
        self.comp_and_gid_layer = nn.Sequential(self.get_block(out_dim, 256), nn.Linear(256, 4))
        self.edge_layer = nn.Sequential(self.get_block(2 * out_dim, 256), nn.Linear(256, 1))

    def _encoder_forward(self, input_ids, attention_mask):
        lm_inp = {'input_ids': input_ids, 'attention_mask': attention_mask}
        max_len = max(len(s) for s in lm_inp['input_ids'])
        for k in lm_inp.keys():
            lm_inp[k] = [s + [0] * (max_len - len(s)) for s in lm_inp[k]]
            lm_inp[k] = LongTensor(lm_inp[k]).to(device)
        return self.encoder(**lm_inp)[0][:, 0]

    def concat_max_freq_feat_embedding(self, inps, h):
        all_row_max_freq, all_col_max_freq = [], []
        for x in inps:
            row_max_freq, col_max_freq = x['max_freq_feat'][:x['num_rows']], x['max_freq_feat'][-x['num_cols']:]
            row_max_freq = LongTensor(row_max_freq).unsqueeze(1).expand(x['num_rows'], x['num_cols']).reshape(-1)
            col_max_freq = LongTensor(col_max_freq).unsqueeze(0).expand(x['num_rows'], x['num_cols']).reshape(-1)
            all_row_max_freq += row_max_freq.tolist() + x['max_freq_feat'][:x['num_rows']] + [0] * x['num_cols']
            all_col_max_freq += col_max_freq.tolist() + [0] * x['num_rows'] + x['max_freq_feat'][-x['num_cols']:]
            if self.args['use_caption']:
                all_row_max_freq.append(0)
                all_col_max_freq.append(0)

        all_row_max_freq, all_col_max_freq = LongTensor(all_row_max_freq), LongTensor(all_col_max_freq)
        all_row_max_freq[all_row_max_freq > 5] = 5
        all_col_max_freq[all_col_max_freq > 5] = 5
        max_freq_feats = self.max_freq_feat_embedding(all_row_max_freq.to(device)) + self.max_freq_feat_embedding(all_col_max_freq.to(device))
        h = torch.cat([h, max_freq_feats], dim=1)
        return h

    def calc_constraint_loss(self, inps, row_col_gid_logits):
        comp_gid_probs = F.softmax(row_col_gid_logits, dim=1)
        base = 0
        constraints = {'1_2': [], '1_3': [], '2_3': [], '3_3': []}
        for x in inps:
            row_probs = comp_gid_probs[base:base+x['num_rows']].unsqueeze(1)
            base += x['num_rows']
            col_probs = comp_gid_probs[base:base+x['num_cols']].unsqueeze(0)
            base += x['num_cols']

            constraints['1_2'].append((row_probs[:, :, 1:3] + col_probs[:, :, 1:3] - 1).flatten())

            row_pairs = self.get_all_pairs_torch(x['num_rows'], ordered=True)
            col_pairs = self.get_all_pairs_torch(x['num_cols'], ordered=True)
            constraints['1_3'] += [
                (row_probs[:, 0, 1][row_pairs[:, 0]] + row_probs[:, 0, 3][row_pairs[:, 1]] - 1).flatten(),
                (col_probs[0, :, 1][col_pairs[:, 0]] + col_probs[0, :, 3][col_pairs[:, 1]] - 1).flatten(),
            ]

            constraints['2_3'] += [
                (row_probs[:, :, 2] + col_probs[:, :, 3] - 1).flatten(),
                (row_probs[:, :, 3] + col_probs[:, :, 2] - 1).flatten(),
            ]

            row_col_pairs = self.get_all_pairs_torch(x['num_rows'] + x['num_cols'])
            gid_probs = torch.cat([row_probs[:, 0, 3], col_probs[0, :, 3]])
            constraints['3_3'].append(gid_probs[row_col_pairs[:, 0]] + gid_probs[row_col_pairs[:, 1]] - 1)

        constraint_loss = tensor(0.0).to(device)
        for c in constraints.keys():
            constraint_loss += F.relu(torch.cat(constraints[c])).mean()
        return constraint_loss

    def forward(self, inps):
        lm_inp = {'input_ids': [], 'attention_mask': []}
        for x in inps:
            for k in lm_inp.keys():
                lm_inp[k] += x[k]
        embs = []
        for idx in range(0, len(lm_inp['input_ids']), 160):
            embs.append(self._encoder_forward(lm_inp['input_ids'][idx:idx+160], lm_inp['attention_mask'][idx:idx+160]))
        del lm_inp
        cell_h = torch.cat(embs)
        del embs

        if self.args['use_caption']:
            lm_inp = {'input_ids': [], 'attention_mask': []}
            for x in inps:
                for k in lm_inp.keys():
                    lm_inp[k] += x[f'caption_{k}']
            caption_h = self._encoder_forward(lm_inp['input_ids'], lm_inp['attention_mask'])
            del lm_inp

        mask_keys = ['cell', 'row', 'col']
        if self.args['use_caption']:
            mask_keys.append('caption')
        mask = {k: [] for k in mask_keys}

        row_positional_idxs, col_positional_idxs = [], []
        for x in inps:
            mask['cell'] += [1] * x['num_cells'] + [0] * (x['num_rows'] + x['num_cols'] )
            mask['row'] += [0] * x['num_cells'] + [1] * x['num_rows'] + [0] * (x['num_cols'])
            mask['col'] += [0] * (x['num_cells'] + x['num_rows']) + [1] * x['num_cols']

            table_cells = np.arange(x['num_cells']).reshape(x['num_rows'], x['num_cols'])
            row_nums = table_cells // x['num_cols']
            row_nums[row_nums > 2] = 2
            row_nums += 1
            row_positional_idxs += row_nums.flatten().tolist() + row_nums[:, 0].tolist() + [0] * x['num_cols']

            col_nums = table_cells % x['num_cols']
            col_nums[col_nums > 2] = 2
            col_nums += 1
            col_positional_idxs += col_nums.flatten().tolist() + [0] * x['num_rows'] + col_nums[0].tolist()

            if self.args['use_caption']:
                mask['caption'] += [0] * (x['num_cells'] + x['num_rows'] + x['num_cols'])
                for k in mask_keys:
                    mask[k].append(1 if k == 'caption' else 0)
                row_positional_idxs.append(0)
                col_positional_idxs.append(0)

        for k in mask_keys:
            mask[k] = Tensor(mask[k]).bool().to(device)

        h = self.default_embedding(LongTensor([0] * len(mask['cell'])).to(device))
        h[mask['cell']] = cell_h
        del cell_h
        if self.args['use_caption']:
            h[mask['caption']] = caption_h
            del caption_h
        
        h += self.positional_embeddings(LongTensor(row_positional_idxs).to(device)) + \
        self.positional_embeddings(LongTensor(col_positional_idxs).to(device))

        if self.args['use_max_freq_feat']:
            h = self.concat_max_freq_feat_embedding(inps, h)

        base = 0
        batch_all_edges, batch_table_edges, batch_edge_labels = [], [], []
        for x in inps:
            table_edges, extra_edges = self.get_edges(x['num_rows'], x['num_cols'], ret_extra_edges=True, ret_caption_edges=self.args['use_caption'])
            batch_all_edges.append(torch.cat([table_edges, extra_edges]) + base)
            batch_table_edges.append(table_edges + base)
            batch_edge_labels.append(self.create_edge_labels(table_edges, x['edge_list']))
            base += x['num_cells'] + x['num_rows'] + x['num_cols']
            if self.args['use_caption']:
                base += 1

        batch_all_edges = torch.cat(batch_all_edges)
        batch_table_edges = torch.cat(batch_table_edges).to(device)
        batch_edge_labels = torch.cat(batch_edge_labels).to(device)

        batch_g = dgl.graph((batch_all_edges[:, 0], batch_all_edges[:, 1])).to(device)

        for l in range(len(self.gat_layers)):
            h = F.elu(self.gat_layers[l](batch_g, h)).flatten(1)
        h = self.dropout(h)

        batch_row_col_gid_labels = []
        for x in inps:
            batch_row_col_gid_labels += x['row_label'] + x['col_label']
        batch_row_col_gid_labels = LongTensor(batch_row_col_gid_labels).to(device)

        row_col_gid_logits = self.comp_and_gid_layer(h[mask['row'] | mask['col']])
        assert len(batch_row_col_gid_labels) == len(row_col_gid_logits)

        edge_embs = torch.cat([h[batch_table_edges[:, 0]], h[batch_table_edges[:, 1]]], dim=1)
        edge_logits = self.edge_layer(edge_embs).squeeze(-1)
        assert len(batch_edge_labels) == len(edge_logits)

        ret = (row_col_gid_logits, batch_row_col_gid_labels), (edge_logits, batch_edge_labels),

        if self.training:
            if self.args['add_constraint']:
                constraint_loss = self.calc_constraint_loss(inps, row_col_gid_logits)
            else:
                constraint_loss = tensor(0.0).to(device)
            ret += (constraint_loss, ),

        return ret

