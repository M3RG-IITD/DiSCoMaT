import numpy as np
import pandas as pd

import torch
from torch import nn, Tensor, LongTensor

from transformers import AutoConfig, AutoModel, AutoTokenizer


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class TapasAdaptedBaselineModel(nn.Module):

    @staticmethod
    def get_all_pair(ar):
        return np.array(np.meshgrid(ar, ar)).T.reshape(-1, 2)

    @staticmethod
    def get_edges(r, c):
        edges = np.empty((0, 2), dtype=int)
        row_edges = TapasAdaptedBaselineModel.get_all_pair(np.arange(c))
        for i in range(r):
            edges = np.concatenate((edges, row_edges + i * c), axis=0)
        col_edges = TapasAdaptedBaselineModel.get_all_pair(np.arange(0, r * c, c))
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

    @staticmethod
    def get_max_possible_cells(input_ids, curr_len):
        curr_len += 1 # one for last SEP in end
        num_cells = 0
        for l in input_ids:
            curr_len += max(len(l) - 2, 1)
            if curr_len > 512: break
            num_cells += 1
        return num_cells

    def __init__(self, args: dict):
        super(TapasAdaptedBaselineModel, self).__init__()
        self.args = args

        config = AutoConfig.from_pretrained(args['lm_name'], cache_dir=args['cache_dir'])
        self.tokenizer = AutoTokenizer.from_pretrained(args['lm_name'], cache_dir=args['cache_dir'], model_max_length=512)
        self.tokenizer.add_tokens(['[EMPTY]']) # for empty table cells
        self.empty_cell_input_id = self.tokenizer.convert_tokens_to_ids(['[EMPTY]'])

        self.encoder = AutoModel.from_pretrained(args['lm_name'], config=config, cache_dir=args['cache_dir'])
        self.encoder.resize_token_embeddings(len(self.tokenizer))

        self.row_embeddings = nn.Embedding(256, config.hidden_size)
        self.col_embeddings = nn.Embedding(256, config.hidden_size)

        out_dim = config.hidden_size
        self.comp_and_gid_layer = nn.Sequential(nn.Dropout(0.2), self.get_block(out_dim, 256), nn.Linear(256, 4))
        self.edge_layer = nn.Sequential(nn.Dropout(0.2), self.get_block(2 * out_dim, 256), nn.Linear(256, 1))

    def forward(self, inps):
        batch_first_cell_idxs, batch_input_ids, batch_attention_mask  = [], [], []
        batch_row_idxs, batch_col_idxs = [], []

        max_len = 0
        for x in inps:
            first_cell_idxs = []
            curr_input_ids = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)]
            curr_input_ids += x['caption_input_ids'][0][:100]
            curr_input_ids += [self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)]
            row_idxs, col_idxs = [0] * len(curr_input_ids), [0] * len(curr_input_ids)

            max_possible_rows = self.get_max_possible_cells(x['input_ids'], len(curr_input_ids)) // x['num_cols']

            for i in range(max_possible_rows):
                for j in range(x['num_cols']):
                    first_cell_idxs.append(len(curr_input_ids))
                    l = x['input_ids'][i * x['num_cols'] + j][1:-1]
                    if len(l) == 0: l = self.empty_cell_input_id
                    curr_input_ids += l
                    row_idxs += [i + 1] * len(l)
                    col_idxs += [j + 1] * len(l)

            curr_input_ids += [self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)]
            row_idxs.append(0)
            col_idxs.append(0)
            assert len(curr_input_ids) <= 512
            batch_first_cell_idxs.append(first_cell_idxs)
            batch_input_ids.append(curr_input_ids)
            batch_row_idxs.append(row_idxs)
            batch_col_idxs.append(col_idxs)
            batch_attention_mask.append([1] * len(curr_input_ids))
            max_len = max(max_len, len(curr_input_ids))

        batch_input_ids = LongTensor([l + [0] * (max_len - len(l)) for l in batch_input_ids]).to(device)
        batch_attention_mask = LongTensor([l + [0] * (max_len - len(l)) for l in batch_attention_mask]).to(device)
        batch_row_idxs = LongTensor([l + [0] * (max_len - len(l)) for l in batch_row_idxs]).to(device)
        batch_col_idxs = LongTensor([l + [0] * (max_len - len(l)) for l in batch_col_idxs]).to(device)

        h = self.encoder(input_ids=batch_input_ids, attention_mask=batch_attention_mask)[0]
        assert h.shape[0] == len(inps) and h.shape[1] == max_len and h.shape[2] == 768
        del batch_input_ids, batch_attention_mask
        h += self.row_embeddings(batch_row_idxs) + self.col_embeddings(batch_col_idxs)
        assert h.shape[0] == len(inps) and h.shape[1] == max_len and h.shape[2] == 768
        del batch_row_idxs, batch_col_idxs

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

