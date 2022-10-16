import numpy as np
import pandas as pd

import torch
from torch import nn, Tensor, LongTensor

from table_bert.utils import BertTokenizer
from table_bert.vertical.vertical_attention_table_bert import VerticalAttentionTableBert
from table_bert.table import DiscomatTable

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class TabertBaselineModel(nn.Module):

    @staticmethod
    def get_all_pair(ar):
        return np.array(np.meshgrid(ar, ar)).T.reshape(-1, 2)

    @staticmethod
    def get_edges(r, c):
        edges = np.empty((0, 2), dtype=int)
        row_edges = TabertBaselineModel.get_all_pair(np.arange(c))
        for i in range(r):
            edges = np.concatenate((edges, row_edges + i * c), axis=0)
        col_edges = TabertBaselineModel.get_all_pair(np.arange(0, r * c, c))
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
        super(TabertBaselineModel, self).__init__()
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer.add_tokens(['[EMPTY]']) # for empty table cells
        self.empty_cell_input_id = self.tokenizer.convert_tokens_to_ids(['[EMPTY]'])
        self.vertical_attention_table_bert_model = VerticalAttentionTableBert.from_pretrained('bert-base-uncased', cache_dir=args['cache_dir']) # initialize the vertical attention table bert model architecture with bert weights and not the pretrained weights

        out_dim = 768
        self.comp_and_gid_layer = nn.Sequential(nn.Dropout(0.2), self.get_block(out_dim, 256), nn.Linear(256, 4))
        self.edge_layer = nn.Sequential(nn.Dropout(0.2), self.get_block(2 * out_dim, 256), nn.Linear(256, 1))

    def truncate_caption(self, caption, max_len=100):
        caption_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(caption)[:max_len])
        return self.tokenizer.decode(caption_ids)

    def forward(self, inps):

        tables = []
        contexts = []
        max_r = 0
        max_c = 0

        for x in inps:
            max_r = max(max_r, x['num_rows'])
            max_c = max(max_c, x['num_cols'])
            table = DiscomatTable(id=str(x['pii'])+":"+str(x['t_idx']), data=x['act_table'])
            table.tokenize(self.vertical_attention_table_bert_model.tokenizer)
            context = self.vertical_attention_table_bert_model.tokenizer.tokenize(x['caption'])
            tables.append(table)
            contexts.append(context)

        context_encodings, schema_encodings, final_table_encodings, info_dict = self.vertical_attention_table_bert_model.encode_for_discomat(
            contexts=contexts,
            tables=tables
        )
        del tables, contexts
        del context_encodings, schema_encodings
        torch.cuda.empty_cache()
       
        assert final_table_encodings.shape[0] == len(inps)
        assert final_table_encodings.shape[1] == min(self.vertical_attention_table_bert_model.config.sample_row_num, max_r)
        assert final_table_encodings.shape[2] == max_c
        assert final_table_encodings.shape[3] == 768

        row_col_embs, edge_embs, batch_edge_labels = [], [], []
        extra_rows_mask, extra_edges_mask = [], []
        for i, x in enumerate(inps):
            r, c = x['num_rows'], x['num_cols']
            table_mask = info_dict['tensor_dict']['table_mask'][i]
            table_cell_embs = final_table_encodings[i][:r, :c, :]
            r_ = min(r, final_table_encodings.shape[1])

            if r > r_:
                # means some rows have been truncated 
                # append extra rows to table_cell_embs
                extra_rows = torch.zeros(r-r_, c, table_cell_embs.shape[-1], device=device)
                table_cell_embs = torch.cat((table_cell_embs.to(device), extra_rows), dim=0)

            assert table_cell_embs.shape[0] == r and table_cell_embs.shape[1] == c and table_cell_embs.shape[2] == 768
            first_cell_embs = table_cell_embs.reshape(r*c, table_cell_embs.shape[-1])
            row_embs, col_embs = table_cell_embs.mean(1), table_cell_embs.mean(0)
            assert row_embs.shape[0] == r and col_embs.shape[0] == c
            row_col_embs += [row_embs, col_embs]
            extra_rows_mask += [0] * r_ + [1] * (r - r_) + [0] * c

            table_edges = self.get_edges(r_, c)
            act_table_edges = self.get_edges(r, c)
            curr_edge_embs = torch.zeros(len(act_table_edges), 2 * table_cell_embs.shape[-1]).to(device)
            curr_edge_embs[act_table_edges.max(1)[0] < r_ * c] = torch.cat([first_cell_embs[table_edges[:, 0]], first_cell_embs[table_edges[:, 1]]], dim=1)
            edge_embs.append(curr_edge_embs)
            batch_edge_labels.append(self.create_edge_labels(act_table_edges, x['edge_list']))
            extra_edges_mask.append(act_table_edges.max(1)[0] >= r_ * c)

        keys = list(info_dict['tensor_dict'].keys())[:]
        for k in keys:
            del info_dict['tensor_dict'][k]
        del keys
        del first_cell_embs, table_cell_embs, curr_edge_embs, info_dict, table_edges, act_table_edges, final_table_encodings
        if r > r_:
            del extra_rows
        torch.cuda.empty_cache()
        extra_rows_mask = Tensor(extra_rows_mask).bool().to(device)

        row_col_gid_logits = self.comp_and_gid_layer(torch.cat(row_col_embs))
        row_col_gid_logits[extra_rows_mask, 0] = 1.0
        row_col_gid_logits[extra_rows_mask, 1:] = 0.0
        del row_col_embs
        torch.cuda.empty_cache()

        extra_edges_mask = torch.cat(extra_edges_mask)
        edge_logits = self.edge_layer(torch.cat(edge_embs)).squeeze(-1)
        edge_logits[extra_edges_mask] = float('-inf')
        del edge_embs
        torch.cuda.empty_cache()

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