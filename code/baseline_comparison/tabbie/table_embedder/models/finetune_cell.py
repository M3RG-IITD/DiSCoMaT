
from typing import Dict, Optional
from overrides import overrides
from sklearn.utils.class_weight import compute_class_weight

import os
import copy
import torch
import numpy as np
from pathlib import Path
from scripts.to_npy import ToNpy
import pdb
from torch import nn, Tensor, LongTensor
import pandas as pd
import torch.nn.functional as F
import logging

from allennlp.nn import util
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.modules import FeedForward, TextFieldEmbedder, Seq2VecEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics.categorical_accuracy import CategoricalAccuracy
from allennlp.training.metrics.fbeta_measure import FBetaMeasure
# from allennlp.training.metrics.mean_absolute_error import MeanAbsoluteError
# from allennlp.modules.token_embedders import PretrainedBertEmbedder
from table_embedder.models.lib.bert_token_embedder import PretrainedBertEmbedder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, precision_recall_fscore_support

from table_embedder.models.embedder_util import TableUtil
# from embedder_util import TableUtil
# from embedder_util import PredUtil
from table_embedder.models.lib.stacked_self_attention import StackedSelfAttentionEncoder
# from lib.stacked_self_attention import StackedSelfAttentionEncoder
from allennlp.models.archival import load_archive
import pickle
from pred_only_val import check_results


from table_embedder.models.cache_util import CacheUtil



@Model.register("finetune_cell")
class TableEmbedder(Model):
    


    def __init__(self, vocab: Vocabulary,
                 bert_embbeder: PretrainedBertEmbedder,
                 feedforward: FeedForward,
                 row_pos_embedding: Embedding,
                 col_pos_embedding: Embedding,
                 transformer_col1: StackedSelfAttentionEncoder,
                 transformer_col2: StackedSelfAttentionEncoder,
                 transformer_col3: StackedSelfAttentionEncoder,
                 transformer_col4: StackedSelfAttentionEncoder,
                 transformer_col5: StackedSelfAttentionEncoder,
                 transformer_col6: StackedSelfAttentionEncoder,
                 transformer_col7: StackedSelfAttentionEncoder,
                 transformer_col8: StackedSelfAttentionEncoder,
                 transformer_col9: StackedSelfAttentionEncoder,
                 transformer_col10: StackedSelfAttentionEncoder,
                 transformer_col11: StackedSelfAttentionEncoder,
                 transformer_col12: StackedSelfAttentionEncoder,
                 transformer_row1: StackedSelfAttentionEncoder,
                 transformer_row2: StackedSelfAttentionEncoder,
                 transformer_row3: StackedSelfAttentionEncoder,
                 transformer_row4: StackedSelfAttentionEncoder,
                 transformer_row5: StackedSelfAttentionEncoder,
                 transformer_row6: StackedSelfAttentionEncoder,
                 transformer_row7: StackedSelfAttentionEncoder,
                 transformer_row8: StackedSelfAttentionEncoder,
                 transformer_row9: StackedSelfAttentionEncoder,
                 transformer_row10: StackedSelfAttentionEncoder,
                 transformer_row11: StackedSelfAttentionEncoder,
                 transformer_row12: StackedSelfAttentionEncoder,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(TableEmbedder, self).__init__(vocab, regularizer)
        self.row_pos_embedding = row_pos_embedding
        self.col_pos_embedding = col_pos_embedding
        self.feedforward = feedforward
        self.bert_embedder = bert_embbeder

        # self.transformer_col = transformer_col
        self.transformer_col1 = transformer_col1
        self.transformer_col2 = transformer_col2
        self.transformer_col3 = transformer_col3
        self.transformer_col4 = transformer_col4
        self.transformer_col5 = transformer_col5
        self.transformer_col6 = transformer_col6
        self.transformer_col7 = transformer_col7
        self.transformer_col8 = transformer_col8
        self.transformer_col9 = transformer_col9
        self.transformer_col10 = transformer_col10
        self.transformer_col11 = transformer_col11
        self.transformer_col12 = transformer_col12
        # self.transformer_row = transformer_row
        self.transformer_row1 = transformer_row1
        self.transformer_row2 = transformer_row2
        self.transformer_row3 = transformer_row3
        self.transformer_row4 = transformer_row4
        self.transformer_row5 = transformer_row5
        self.transformer_row6 = transformer_row6
        self.transformer_row7 = transformer_row7
        self.transformer_row8 = transformer_row8
        self.transformer_row9 = transformer_row9
        self.transformer_row10 = transformer_row10
        self.transformer_row11 = transformer_row11
        self.transformer_row12 = transformer_row12
        self.loss = torch.nn.BCELoss()
        self.metrics = {
            # "accuracy": CategoricalAccuracy(),
            # "haccuracy": CategoricalAccuracy(),
            # "caccuracy": CategoricalAccuracy(),
            "fscore_rcid" : FBetaMeasure(),
            'fscore_edg' : FBetaMeasure(),
            'tuple_met_f' : FBetaMeasure(),
            'comp_met_f' : FBetaMeasure(),
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        all_train_comp_labels = pickle.load(open('./save_pickle/all_train_comp_labels.pkl', 'rb'))
        comp_gid_class_weights = Tensor(compute_class_weight('balanced', classes=[0, 1, 2, 3], y=all_train_comp_labels)).to(self.device)
        self.loss_func2 = torch.nn.CrossEntropyLoss(weight=comp_gid_class_weights)

        all_train_edge_labels = pickle.load(open('./save_pickle/all_train_edge_labels.pkl', 'rb'))
        self.neg_edge_wt, self.pos_edge_wt = compute_class_weight('balanced', classes=[0, 1], y=all_train_edge_labels)
        self.loss_func1 = torch.nn.BCEWithLogitsLoss(reduction='none')

        self.num_max_row_pos = 35
        self.num_max_col_pos = 25
        self.count = 0
        self.tot_count = 0

        self.y_comp_true, self.y_comp_pred = [], []
        self.y_edge_true, self.y_edge_pred = [], []
        self.tot_edge_loss = 0
        self.tot_ccid_loss = 0
        
        out_dim = 768
        self.comp_and_gid_layer = nn.Sequential(nn.Dropout(0.2), self.get_block(out_dim, 256), nn.Linear(256, 4)) 
        self.edge_layer = nn.Sequential(nn.Dropout(0.2), self.get_block(2 * out_dim, 256), nn.Linear(256, 1)) 

        self.cache_usage = os.getenv("cache_usage")
        self.cache_dir = os.getenv('cache_dir')
        self.cls_col = np.load(os.getenv("clscol_path"))
        self.cls_row = np.load(os.getenv("clsrow_path"))
        
        if self.cache_dir is not None:
            cache_dir = Path(self.cache_dir)
            self.cell_feats, self.cell_id = ToNpy.load_cid(cache_dir/'cell_feats.npy', cache_dir/'cell_id.txt')

        if os.getenv('model_path') is not None and os.getenv('learn_type') != 'pred':
            self.init_weight()

        self.opt_level = 'O0'
        initializer(self)
        
    @staticmethod
    def get_all_pair(ar):
        return np.array(np.meshgrid(ar, ar)).T.reshape(-1, 2)

    @staticmethod
    def get_edges(r, c):
        edges = np.empty((0, 2), dtype=int)
        row_edges = TableEmbedder.get_all_pair(np.arange(c))
        for i in range(r):
            edges = np.concatenate((edges, row_edges + i * c), axis=0)
        col_edges = TableEmbedder.get_all_pair(np.arange(0, r * c, c))
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
        


    def init_weight(self):
        model_path = os.getenv('model_path')
        archive = load_archive(model_path)
        # https://github.com/allenai/allennlp-semparse/blob/master/allennlp_semparse/models/wikitables/wikitables_erm_semantic_parser.py
        model_parameters = dict(self.named_parameters())
        archived_parameters = dict(archive.model.named_parameters())
        for name, weights in archived_parameters.items():
            if name in model_parameters:
                new_weights = weights.data
                model_parameters[name].data.copy_(new_weights)

    def pred_by_2d_transformer(self, row_embs, col_embs, table_mask_cls, bs, max_rows, max_cols):
        cells = torch.cat([row_embs, col_embs], dim=3)  
        out_prob_cell = self.feedforward(cells)  
        cell_mask_mod = table_mask_cls.reshape(bs, max_rows, max_cols, 1)
        out_prob_cell = util.masked_softmax(out_prob_cell, cell_mask_mod)  

        return out_prob_cell

    def get_labels(self, table_info, bs, n_rows, n_cols):
        header_labels = torch.zeros((bs, n_cols), device=self.device)
        cell_labels = torch.zeros((bs, n_rows, n_cols), device=self.device)
        for k, one_info in enumerate(table_info):
            if 'col_labels' in one_info and one_info['col_labels'] is not None:
                for label_idx in one_info['col_labels']:
                    header_labels[k][label_idx] = 1
            if 'cell_labels' in one_info and one_info['cell_labels'] is not None:
                for label_idx in one_info['cell_labels']:
                    #cell_labels[k][label_idx[0]][label_idx[1]] = 1
                    row_col_labels = label_idx[0]
                    edge_labels = label_idx[1]
                    
        r_c_l = []
        e_l = []
        if row_col_labels != '[]':
            for i in range(1, len(row_col_labels), 3):
                r_c_l.append(int(row_col_labels[i]))
        
        if edge_labels != '[]':
            import re
            aab = edge_labels[1:-1].replace('(', '').replace(')', '')
            aabr = re.split('\,\s', aab)
            for i in range(0, len(aabr), 2):
                e_l.append((int(aabr[i]), int(aabr[i+1])))
            
        return r_c_l, e_l


    def get_batch_edge_gid_pred_labels(self, comp_gid_logits, edge_logits, num_rows: list, num_cols: list):
        base_comp_gid, base_edge = 0, 0
        pred_edge_labels, pred_edges, pred_row_col_gid_labels = [], [], []
        comp_gid_labels = comp_gid_logits.argmax(1).tolist()
        for r, c in zip(num_rows, num_cols):
            num_comp_labels, num_edge_logits = r + c, r * c * (r + c - 1)
            row_col_gid_labels, edge_labels, edges = self.get_edge_gid_pred_labels(
                comp_gid_labels[base_comp_gid:base_comp_gid+num_comp_labels], 
                comp_gid_logits[base_comp_gid:base_comp_gid+num_comp_labels], 
                edge_logits[base_edge:base_edge+num_edge_logits], r, c)
            pred_edge_labels += edge_labels
            pred_edges.append(edges)
            pred_row_col_gid_labels += row_col_gid_labels
            base_comp_gid += num_comp_labels
            base_edge += num_edge_logits
        return pred_row_col_gid_labels, pred_edge_labels, pred_edges

    @staticmethod
    def rectify_comp_labels(row_labels, col_labels):
        row_1, row_2, col_1, col_2 = row_labels.count(1), row_labels.count(2), col_labels.count(1), col_labels.count(2)
        if row_1 + col_1 == 0 or row_2 + col_2 == 0 or row_1 + row_2 == 0 or col_1 + col_2 == 0:
            return [0] * len(row_labels), [0] * len(col_labels)
        if (row_2 + col_1 == 0) or (row_1 + col_2 == 0):
            return row_labels, col_labels
        if row_1 == 0:
            return row_labels, [0 if c == 2 else c for c in col_labels]
        if row_2 == 0:
            return row_labels, [0 if c == 1 else c for c in col_labels]
        if col_1 == 0:
            return [0 if r == 2 else r for r in row_labels], col_labels
        if col_2 == 0:
            return [0 if r == 1 else r for r in row_labels], col_labels
        if row_1 > row_2 and col_2 > col_1:
            return [0 if r == 2 else r for r in row_labels], [0 if c == 1 else c for c in col_labels]
        if row_2 > row_1 and col_1 > col_2:
            return [0 if r == 1 else r for r in row_labels], [0 if c == 2 else c for c in col_labels]
        return [0] * len(row_labels), [0] * len(col_labels)

    def get_edge_gid_pred_labels(self, comp_gid_labels, comp_gid_logits, edge_logits, num_rows: int, num_cols: int):
        row_col_labels = [0 if c == 3 else c for c in comp_gid_labels]
        row_labels, col_labels = TableEmbedder.rectify_comp_labels(row_col_labels[:num_rows], row_col_labels[-num_cols:])
        r, c = Tensor(row_labels).unsqueeze(1), Tensor(col_labels).unsqueeze(0)
        comp_cells = np.where((r * c).view(-1) == 2)[0]

        if sum(row_labels + col_labels) > 0:
            if 1 in row_labels:
                assert 2 in col_labels
                if 3 in comp_gid_labels[-num_cols:]:
                    col_gid_probs = F.softmax(comp_gid_logits[-num_cols:], dim=1)
                    idx = np.where(col_gid_probs[:, 3] == col_gid_probs[col_gid_probs.argmax(1) == 3, 3].max())[0][0]
                    col_labels[idx] = 3
            else:
                assert 1 in col_labels and 2 in row_labels
                if 3 in comp_gid_labels[:num_rows]:
                    row_gid_probs = F.softmax(comp_gid_logits[:num_rows], dim=1)
                    idx = np.where(row_gid_probs[:, 3] == row_gid_probs[row_gid_probs.argmax(1) == 3, 3].max())[0][0]
                    row_labels[idx] = 3

        df = pd.DataFrame(self.get_edges(num_rows, num_cols).tolist())
        df.columns = ['src', 'dst']
        df['wt'] = edge_logits
        idx = (df.groupby('src')['wt'].transform(max) == df['wt']) & df['src'].isin(comp_cells)
        df.drop('wt', inplace=True, axis=1)
        edges = df[idx].applymap(lambda x: (x // num_cols, x % num_cols)).values.tolist()
        return row_labels + col_labels, list(idx.astype(int).values), edges

        

    def get_tabemb(self, bert_header, bert_data, n_rows, n_cols, bs, table_mask, nrows, ncols):
        row_pos_ids = torch.arange(0, self.num_max_row_pos, device=self.device, dtype=torch.long)
        col_pos_ids = torch.arange(0, self.num_max_col_pos, device=self.device, dtype=torch.long)

        n_rows += 1  # row CLS
        n_cols += 1  # col CLS
        cls_col = torch.from_numpy(copy.deepcopy(self.cls_col)).to(device=self.device)
        cls_row = torch.from_numpy(copy.deepcopy(self.cls_row)).to(device=self.device)
        row_pos_embs = self.row_pos_embedding(row_pos_ids[:n_rows+1])
        col_pos_embs = self.col_pos_embedding(col_pos_ids[:n_cols])

        for i in range(1, 13):
            transformer_row = getattr(self, 'transformer_row{}'.format(str(i)))
            transformer_col = getattr(self, 'transformer_col{}'.format(str(i)))
            if i == 1:
                bert_data = TableUtil.add_cls_tokens(bert_header, bert_data, cls_row, cls_col, bs, n_rows, n_cols)
                bert_data += row_pos_embs.expand((bs, n_cols, n_rows + 1, 768)).permute(0, 2, 1, 3).expand_as(bert_data)
                bert_data += col_pos_embs.expand((bs, n_rows + 1, n_cols, 768)).expand_as(bert_data)
                table_mask_cls = TableUtil.add_cls_mask(table_mask, bs, n_rows, n_cols, self.device, nrows, ncols)
                col_embs = TableUtil.get_col_embs(bert_data, bs, n_rows, n_cols, table_mask_cls, transformer_col, self.opt_level)
                row_embs = TableUtil.get_row_embs(bert_data, bs, n_rows, n_cols, table_mask_cls, transformer_row, self.opt_level)
            else:
                row_embs = TableUtil.get_row_embs(ave_embs, bs, n_rows, n_cols, table_mask_cls, transformer_row, self.opt_level)
                col_embs = TableUtil.get_col_embs(ave_embs, bs, n_rows, n_cols, table_mask_cls, transformer_col, self.opt_level)
            ave_embs = (row_embs + col_embs) / 2.0

        return row_embs, col_embs, n_rows, n_cols, table_mask_cls



    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        fscore_dic = self.metrics["fscore_rcid"].get_metric(reset=reset)
        avgfscore = (fscore_dic['fscore'][1] + fscore_dic['fscore'][2] + fscore_dic['fscore'][3])/3
        edge_score = self.metrics['fscore_edg']['fscore']
        edg_loss = self.tot_edge_loss.item()
        ccid_loss = self.tot_ccid_loss.item()
        comp_m_f1 = self.metrics['comp_met_f']
        return {'compo' : fscore_dic['fscore'][1], 'consti': fscore_dic['fscore'][2], 'id':fscore_dic['fscore'][3], 'cid_avg_fscore':avgfscore, 'edge_score':edge_score, 'edge_loss':edg_loss, 'ccid_loss':ccid_loss, 'comp_f1':comp_m_f1}


    def get_meta(self, table_info):
        nrows = [one_info['num_rows'] for one_info in table_info]
        ncols = [one_info['num_cols'] for one_info in table_info]
        tids = [one_info['table_id'] for one_info in table_info]
        return nrows, ncols, tids

    @overrides
    def forward(self, table_info: Dict[str, str],
                indexed_headers: Dict[str, torch.LongTensor],#) -> Dict[str, torch.Tensor]:
                indexed_cells: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        


        row_col_embs = []
        edge_embs, batch_edge_labels = [], []
        extra_rows_mask, extra_edges_mask = [], []
        
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        
        # initialize
        self.bert_embedder.eval()
        bs, n_rows, n_cols = TableUtil.get_max_row_col(table_info)
        nrows, ncols, tids = self.get_meta(table_info)
        table_mask = TableUtil.get_table_mask(table_info, bs, n_rows, n_cols, self.device)

        # to row/col emb
        if self.cache_dir is None:
            bert_header, bert_cell = TableUtil.get_bert_emb(indexed_headers, indexed_cells, table_info, bs, n_rows, n_cols, self.cache_usage, self.bert_embedder, None, self.device)
        else:
            bert_header, bert_cell = TableUtil.to_bert_emb(table_info, bs, n_rows, n_cols, self.device, self.cell_id, self.cell_feats)
        # row_embs_cls, col_embs_cls, n_rows_cls, n_cols_cls, table_mask_cls = self.get_tab_emb(bert_header, bert_cell, n_rows, n_cols, table_info, bs, table_mask)
        row_embs, col_embs, n_rows_cls, n_cols_cls, table_mask_cls = self.get_tabemb(bert_header, bert_cell, n_rows, n_cols, bs, table_mask, nrows, ncols)
        # table_mask = TableUtil.add_cls_mask(table_mask, bs, n_rows, n_cols, self.device, nrows, ncols)
        
        #my required outputs
        name = table_info[0].get('table_id')
       

        train_tab_list = pickle.load(open('./data/table_list/train_list.pkl', 'rb'))
        val_tab_list = pickle.load(open('./data/table_list/val_list.pkl', 'rb'))
        test_tab_list = pickle.load(open('./data/table_list/test_list.pkl', 'rb'))

        if name in train_tab_list: split = 'train'
        elif name in val_tab_list: split = 'val'
        elif name in test_tab_list: split = 'test'

        tr_va = 3906 #total number of tables seen in each epochs = #train tables + #val tables
        if (self.tot_count%tr_va==0 and split=='train'):
            self.tot_count=0

        if (self.tot_count==0 and split=='train') or (self.count==3387 and split!='train'):
            self.y_comp_true, self.y_comp_pred = [], []
            self.count = 0
            self.y_edge_true, self.y_edge_pred = [], []
            print(name)


        self.count += 1
        self.tot_count += 1

        if self.count==1:
            if split=='train': 
                self.tot_ccid_loss = 0
                self.tot_edge_loss = 0
     
            elif split=='val': 
                self.tot_ccid_loss = 0
                self.tot_edge_loss = 0
     
            elif split=='test': 
                self.tot_ccid_loss = 0
                self.tot_edge_loss = 0
     

        if name=='S0022309303004794__1' or name=='S0022309309005572__3':
            row_embss = row_embs[:,1:2,1:,:]
            col_embss = col_embs[:,1:2,1:,:]

        else:
            row_embss = row_embs[:,1:,1:,:]
            col_embss = col_embs[:,1:,1:,:]
            

        df1 = pd.read_csv(f'./data/ft_cell/discomat_data/discomat_all_csv/{name}.csv', header=None)
        r = df1.shape[0]
        c = df1.shape[1]
        
        r_ = min(30, r) # table_info[0].get('num_rows') + 1
        c_ = min(20, c) # table_info[0].get('num_cols')
        
        e_ = 768

        
        temp_row_embs = row_embss.reshape(r_*c_, e_)
        temp_col_embs = col_embss.reshape(r_*c_, e_)
        
        first_cell_embs = (temp_row_embs+temp_col_embs)/2.0
        table_cell_embs = first_cell_embs.reshape(r_, c_, e_)
        
        if c_<c:
            table_cell_embs = F.pad(input=table_cell_embs, pad=(0,0,0,c-c_), mode='constant', value=0)
            first_cell_embs = table_cell_embs.reshape(r_*c, e_)
                
        row_embs, col_embs = table_cell_embs.mean(1), table_cell_embs.mean(0)
        
            
        row_col_embs += [row_embs, torch.zeros(r - r_, e_).to(device), col_embs]
        extra_rows_mask += [0] * r_ + [1] * (r - r_) + [0] * c
        
        row_col_labels, edge_labels = self.get_labels(table_info, bs, n_rows, n_cols)
                
        table_edges = self.get_edges(r_, c)
        act_table_edges = self.get_edges(r, c)
        curr_edge_embs = torch.zeros(len(act_table_edges), 2 * e_).to(device)
        curr_edge_embs[act_table_edges.max(1)[0] < r_ * c] = torch.cat([first_cell_embs[table_edges[:, 0]], first_cell_embs[table_edges[:, 1]]], dim=1)
        edge_embs.append(curr_edge_embs)
        batch_edge_labels.append(self.create_edge_labels(act_table_edges, edge_labels))
        extra_edges_mask.append(act_table_edges.max(1)[0] >= r_ * c)
        y_ed_true = batch_edge_labels[0].tolist()
        
        
        del first_cell_embs, table_cell_embs, curr_edge_embs
        extra_rows_mask = Tensor(extra_rows_mask).bool().to(device)
            
        row_col_gid_logits = self.comp_and_gid_layer(torch.cat(row_col_embs))
        row_col_gid_logits[extra_rows_mask, 0] = 1.0
        row_col_gid_logits[extra_rows_mask, 1:] = 0.0
        extra_edges_mask = torch.cat(extra_edges_mask)
        edge_logits = self.edge_layer(torch.cat(edge_embs)).squeeze(-1)
        edge_logits[extra_edges_mask] = 0 # float('-inf')
        del row_col_embs, edge_embs
            
        batch_row_col_gid_labels = []
        batch_row_col_gid_labels += row_col_labels
        batch_row_col_gid_labels = LongTensor(batch_row_col_gid_labels).to(device)
        assert len(batch_row_col_gid_labels) == len(row_col_gid_logits)
            
        batch_edge_labels = torch.cat(batch_edge_labels).to(device)
        assert len(batch_edge_labels) == len(edge_logits)
        
        ret_comp_pred, ret_edge_pred = [], []

        pred_comp_gid_labels, pred_edge_labels, batch_pred_edges = self.get_batch_edge_gid_pred_labels(row_col_gid_logits.cpu().detach(), edge_logits.cpu().detach(), [r], [c])
        ###compare pred_edge_labels and y_ed_true
        assert len(pred_edge_labels) == len(y_ed_true), 'Edge prediction and edge label dimensions should match'
        mod_pred_edge_labels, mod_y_edge_true = [], []
        for inde in range(len(pred_edge_labels)):
            if pred_edge_labels[inde]==0 and y_ed_true[inde]==0:
                continue
            else:
                mod_pred_edge_labels.append(pred_edge_labels[inde])
                mod_y_edge_true.append(y_ed_true[inde])
        self.y_edge_true += mod_y_edge_true
        self.y_edge_pred += mod_pred_edge_labels
        assert len(self.y_edge_true) == len(self.y_edge_pred), 'Dim for edge pred and edge level after true neg removal should be same'

        if len(self.y_edge_true) == 0:
            edge_metrics = {'precision': 0, 'recall': 0, 'fscore': 0}
        else:
            #if(split=='val'): print('In Val')
            prec, recall, fscore, _ = precision_recall_fscore_support(self.y_edge_true, self.y_edge_pred, average='binary')
            edge_metrics = {'precision': prec, 'recall': recall, 'fscore': fscore}


        edge_loss_vec = self.loss_func1(edge_logits, batch_edge_labels.to(dtype=torch.float32))
        class_wt = torch.ones(len(edge_loss_vec)) * self.neg_edge_wt
        class_wt[batch_edge_labels == 1] = self.pos_edge_wt
        edge_loss = (edge_loss_vec * class_wt.to(device)).mean()

        comp_gid_loss_fn = self.loss_func2(row_col_gid_logits, batch_row_col_gid_labels)
        
        torch.save(edge_logits, f'./edge_logit_17_ada/{name}.pt')
        torch.save(row_col_gid_logits, f'./row_col_gid_logit_17_ada/{name}.pt')
        
        prob_r_c_g_l = F.softmax(row_col_gid_logits, dim=1)
        mask1 = torch.ones(extra_rows_mask.size())
        self.metrics['fscore_rcid'](prob_r_c_g_l, batch_row_col_gid_labels)
        self.metrics['fscore_edg'] = edge_metrics

        
        loss = edge_loss + comp_gid_loss_fn

        output_dict = {'loss': loss}

        self.tot_ccid_loss += comp_gid_loss_fn
        self.tot_edge_loss += edge_loss

        
        if split=='val' and self.count==519:
            print(name)
            print(self.count)
            print(self.tot_count)
            print()
            self.metrics['comp_met_f'] = check_results()
        else:
            self.metrics['comp_met_f'] = 0
        

        return output_dict
        

    @staticmethod
    def add_metadata(table_info, output_dict):
        data_dict = {}
        for one_info in table_info:
            if 'table_id' in one_info:
                one_info['id'] = one_info['table_id']
            for k, v in one_info.items():
                data_dict[k] = data_dict.get(k, [])
                data_dict[k].append(v)
        output_dict.update(data_dict)
        return output_dict
