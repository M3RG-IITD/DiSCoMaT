## DiSCoMaT Dataset

### Steps to download XMLs and create dataset

```bash
python donwload_xml_from_dois.py
```
This step requires elsevier API key. File reads dois from all_dois.pkl and saves XMLs in the piis folder.

```bash
python get_text_from_xmls.py
```
This file reads XMLs from piis folder and parse them to save text for all papers in train_val_test_paper_data.pkl


```bash
python get_table_from_xmls.py
```
This file reads XMLs from piis folder and parses them to save tables for all papers in pii_table_dict.pkl

```bash
python create_train_val_test_data.py
```
This files reads pii_table_dict.pkl and stores a separate dictionary for every table in train_data_new.pkl and val_test_data_new.pkl. The file also tokenizes tables using the MatSciBERT tokenizer. We provide train_data_new.pkl and val_test_data_new.pkl in the data folder.

```bash
python get_regex_from_texts.py
```
The file reads val_test_data_new.pkl, train_val_test_paper_data.pkl to save all the extracted regex compositions in extracted_regex.pkl. We provide extracted_regex.pkl in the data folder.
