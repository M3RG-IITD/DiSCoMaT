## DiSCoMaT Evaluation

### Combined GNN1 and GNN2 results
```bash
bash compute_combined_results.sh
```

This prints the results for all 5 models: DiSCoMat, DiSCoMat w/o features, DiSCoMat w/o constraints, DiSCoMat w/o caption, and v-DiSCoMat. All five metrics (TT Acc., ID F1, TL F1, MatL F1, CV) are displayed.

### Table Type results
```bash
bash compute_table_type_results.sh
```

This prints the results for all table types (SCC, MCC-CI, and MCC-PI) for all 5 models. All four metrics (ID F1, TL F1, MatL F1, CV) are displayed.
