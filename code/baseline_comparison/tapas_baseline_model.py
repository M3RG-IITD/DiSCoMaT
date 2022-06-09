import numpy as np
import pandas as pd

import torch
from torch import nn, Tensor, LongTensor

from transformers import TapasConfig, TapasModel, TapasTokenizer


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class TapasBaselineModel(nn.Module):

    @staticmethod
    def get_all_pair(ar):
        return np.array(np.meshgrid(ar, ar)).T.reshape(-1, 2)

    @staticmethod
    def get_edges(r, c):
        edges = np.empty((0, 2), dtype=int)
        row_edges = TapasBaselineModel.get_all_pair(np.arange(c))
        for i in range(r):
            edges = np.concatenate((edges, row_edges + i * c), axis=0)
        col_edges = TapasBaselineModel.get_all_pair(np.arange(0, r * c, c))
        for i in range(c):
            edges = np.concatenate((edges, col_edges + i), axis=0)
        edges = np.unique(edges, axis=0)
        table_edges = LongTensor(edges[np.lexsort((edges[:, 1], edges[:, 0]))])
        assert len(table_edges) == r * c * (r + c - 1)
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

    def __init__(self, args: dict):
        super(TapasBaselineModel, self).__init__()
        self.args = args

        self.tokenizer = TapasTokenizer.from_pretrained(args['lm_name'], cache_dir=args['cache_dir'], strip_column_names=True, cell_trim_length=50)
        config = TapasConfig.from_pretrained(args['lm_name'], cache_dir=args['cache_dir'])
        self.encoder = TapasModel(config)

        out_dim = config.hidden_size
        self.comp_and_gid_layer = nn.Sequential(nn.Dropout(0.2), self.get_block(out_dim, 256), nn.Linear(256, 4))
        self.edge_layer = nn.Sequential(nn.Dropout(0.2), self.get_block(2 * out_dim, 256), nn.Linear(256, 1))

    def truncate_caption(self, caption, max_len=100):
        caption_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(caption)[:max_len])
        return self.tokenizer.decode(caption_ids)

    def forward(self, inps):
        batch_first_cell_idxs = []
        batch_inputs = {'input_ids': [], 'token_type_ids': [], 'attention_mask': []}

        for x in inps:
            inputs = self.tokenizer(
                table=pd.DataFrame(x['act_table']).applymap(lambda x: '[EMPTY]' if x == '' else x),
                queries=[self.truncate_caption(x['caption'].replace('\n', ' '))],
                padding='max_length',
                truncation='drop_rows_to_fit',
                return_tensors='pt',
            )
            inputs['token_type_ids'][:, :, 3:] = 0 # set prev_labels, col ranks, inv col ranks, numeric_relations to 0

            if x['num_cols'] > 1:
                first_cell_idxs = (np.where(np.diff(inputs['token_type_ids'][0, :, 1].numpy()) != 0)[0] + 1).tolist()
                if inputs['token_type_ids'][0, -1, 1] == 0:
                    first_cell_idxs = first_cell_idxs[:-1]
            else:
                first_cell_idxs = (np.where(np.diff(inputs['token_type_ids'][0, :, 2].numpy()) != 0)[0] + 1).tolist()
                if inputs['token_type_ids'][0, -1, 2] == 0:
                    first_cell_idxs = first_cell_idxs[:-1]

            batch_first_cell_idxs.append(first_cell_idxs)
            for k in batch_inputs.keys():
                batch_inputs[k].append(inputs[k])

        for k in batch_inputs.keys():
            batch_inputs[k] = torch.cat(batch_inputs[k]).to(device)
            assert batch_inputs[k].shape[0] == len(inps)

        h = self.encoder(**batch_inputs).last_hidden_state
        assert h.shape[0] == len(inps)
        del batch_inputs

        row_col_embs, edge_embs, batch_edge_labels = [], [], []
        extra_rows_mask, extra_edges_mask = [], []
        for i, x in enumerate(inps):
            r, c = x['num_rows'], x['num_cols']
            first_cell_embs = h[i][batch_first_cell_idxs[i]]
            assert len(first_cell_embs) % c == 0
            table_cell_embs = first_cell_embs.reshape(-1, c, h.shape[-1])
            row_embs, col_embs = table_cell_embs.mean(1), table_cell_embs.mean(0)
            r_ = len(row_embs)
            assert r_ <= r
            row_col_embs += [row_embs, torch.zeros(r - r_, h.shape[-1]).to(device), col_embs]
            extra_rows_mask += [0] * r_ + [1] * (r - r_) + [0] * c

            table_edges = self.get_edges(r_, c)
            act_table_edges = self.get_edges(r, c)
            curr_edge_embs = torch.zeros(len(act_table_edges), 2 * h.shape[-1]).to(device)
            curr_edge_embs[act_table_edges.max(1)[0] < r_ * c] = torch.cat([first_cell_embs[table_edges[:, 0]], first_cell_embs[table_edges[:, 1]]], dim=1)
            edge_embs.append(curr_edge_embs)
            batch_edge_labels.append(self.create_edge_labels(act_table_edges, x['edge_list']))
            extra_edges_mask.append(act_table_edges.max(1)[0] >= r_ * c)

        del h, first_cell_embs, table_cell_embs, curr_edge_embs
        extra_rows_mask = Tensor(extra_rows_mask).bool().to(device)
        row_col_gid_logits = self.comp_and_gid_layer(torch.cat(row_col_embs))
        row_col_gid_logits[extra_rows_mask, 0] = 1.0
        row_col_gid_logits[extra_rows_mask, 1:] = 0.0
        extra_edges_mask = torch.cat(extra_edges_mask)
        edge_logits = self.edge_layer(torch.cat(edge_embs)).squeeze(-1)
        edge_logits[extra_edges_mask] = float('-inf')
        del row_col_embs, edge_embs

        batch_row_col_gid_labels = []
        for x in inps:
            batch_row_col_gid_labels += x['row_label'] + x['col_label']
        batch_row_col_gid_labels = LongTensor(batch_row_col_gid_labels).to(device)
        assert len(batch_row_col_gid_labels) == len(row_col_gid_logits)

        batch_edge_labels = torch.cat(batch_edge_labels).to(device)
        assert len(batch_edge_labels) == len(edge_logits)

        if self.training:
            return (row_col_gid_logits[~extra_rows_mask], batch_row_col_gid_labels[~extra_rows_mask]), \
            (edge_logits[~extra_edges_mask], batch_edge_labels[~extra_edges_mask])
        else:
            return (row_col_gid_logits, batch_row_col_gid_labels), (edge_logits, batch_edge_labels)

