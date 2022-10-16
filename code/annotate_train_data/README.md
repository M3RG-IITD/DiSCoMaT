## DiSCoMaT Dataset

### Steps to annotate the training dataset using Distant Supervision

```bash
python distant_supervision_scc.py
```
This file reads train_data_new.pkl and SciGlass database files to annotate SCC tables and saves the annotations in train_data_scc.pkl

```bash
python distant_supervision_mcc_ci.py
```
This file reads train_data_scc.pkl and SciGlass files to annotate MCC-CI tables and saves the annotations in train_data_mcc_ci.pkl

```bash
python distant_supervision_mcc_pi.py
```
This file reads train_data_mcc_ci.pkl, SciGlass files, and train_val_test_paper_data.pkl to annotate MCC-PI tables and saves the annotations in train_data_mcc_pi.pkl

We used Interglad database files for annotation of our training set. Since `INTERGLAD` database is not publicly available, here we use `SciGlass` database as a proxy for `INTERGLAD`. 

`INTERGLAD` contains 12634 compositions corresponding to publications in our training set. However, `SciGlass` contains only 2347 compositions for these publications. Therefore, the above steps annotate a subset of training data only. We provide training data annoatated using `INTERGLAD` database in `train_data.pkl` and manually annotated dev & test data in `val_test_data.pkl`. We use these files to train and evaluate DiSCoMaT.
