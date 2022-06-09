import numpy as np
import pandas as pd

import torch
from torch import nn, Tensor, LongTensor
import torch.nn.functional as F

from transformers import AutoConfig, AutoModel

import dgl
from dgl.nn import GATConv


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class GATModel(nn.Module):

    @staticmethod
    def get_all_pair(ar):
        return np.array(np.meshgrid(ar, ar)).T.reshape(-1, 2)

    @staticmethod
    def get_edges(r, c, ret_extra_edges=False):
        edges = np.empty((0, 2), dtype=int)
        row_edges = GATModel.get_all_pair(np.arange(c))
        for i in range(r):
            edges = np.concatenate((edges, row_edges + i * c), axis=0)
        col_edges = GATModel.get_all_pair(np.arange(0, r * c, c))
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
            return table_edges, torch.cat([row_edges, col_edges, row_self_edges, col_self_edges])
        return table_edges

    @staticmethod
    def create_edge_labels(all_edges, edge_list):
        df = pd.DataFrame(all_edges.tolist())
        df['merge'] = list(zip(df[0], df[1]))
        edge_labels = LongTensor(df['merge'].isin(edge_list))
        return edge_labels

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
        return args

    def __init__(self, args: dict):
        super(GATModel, self).__init__()
        self.args = self.validate_args(args)

        config = AutoConfig.from_pretrained(args['lm_name'], cache_dir=args['cache_dir'])
        self.encoder = AutoModel.from_pretrained(args['lm_name'], config=config, cache_dir=args['cache_dir'])

        in_dim = config.hidden_size
        self.default_embedding = nn.Embedding(1, in_dim)
        self.positional_embeddings = nn.Embedding(4, config.hidden_size)

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

    def forward(self, inps):
        lm_inp = {'input_ids': [], 'attention_mask': []}
        for x in inps:
            for k in lm_inp.keys():
                lm_inp[k] += x[k]
        embs = []
        for idx in range(0, len(lm_inp['input_ids']), 200):
            embs.append(self._encoder_forward(lm_inp['input_ids'][idx:idx+200], lm_inp['attention_mask'][idx:idx+200]))
        del lm_inp
        table_h = torch.cat(embs)
        del embs

        mask = {k: [] for k in ['table', 'row', 'col']}
        row_positional_idxs, col_positional_idxs = [], []
        for x in inps:
            mask['table'] += [1] * x['num_cells'] + [0] * (x['num_rows'] + x['num_cols'])
            mask['row'] += [0] * x['num_cells'] + [1] * x['num_rows'] + [0] * x['num_cols']
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

        for k in mask:
            mask[k] = Tensor(mask[k]).bool().to(device)

        h = self.default_embedding(LongTensor([0] * len(mask['table'])).to(device))
        h[mask['table']] = table_h
        del table_h
        h += self.positional_embeddings(LongTensor(row_positional_idxs).to(device)) + \
        self.positional_embeddings(LongTensor(col_positional_idxs).to(device))

        batch_all_edges, batch_table_edges, batch_edge_labels = [], [], []
        base = 0
        for x in inps:
            table_edges, extra_edges = self.get_edges(x['num_rows'], x['num_cols'], ret_extra_edges=True)
            batch_all_edges.append(torch.cat([table_edges, extra_edges]) + base)
            batch_table_edges.append(table_edges + base)
            batch_edge_labels.append(self.create_edge_labels(table_edges, x['edge_list']))
            base += x['num_cells'] + x['num_rows'] + x['num_cols']

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

        return (row_col_gid_logits, batch_row_col_gid_labels), (edge_logits, batch_edge_labels)

