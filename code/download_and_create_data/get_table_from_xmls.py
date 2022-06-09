import os
import pickle

from bs4 import BeautifulSoup
from tqdm import tqdm

from mit_table_extractor import TableExtractor


table_dir = '../../data'
piis_dir = os.path.join(table_dir, 'piis')

te = TableExtractor()
pii_table_dict = dict()

for pii in tqdm(sorted(os.listdir(piis_dir))):
    xml_path = os.path.join(piis_dir, pii, f'{pii}.xml')
    pkl_path = os.path.join(piis_dir, pii, 'table_info_new.pkl')
    if os.path.exists(pkl_path):
        os.remove(pkl_path)
    doi = None
    with open(xml_path, 'r', encoding='utf-8') as file:
        l = BeautifulSoup(file.read(), 'xml').find_all('xocs:doi')
        if len(l) > 0:
            doi = l[0].text
    assert doi is not None
    te.doi = doi
    tables, captions, footers = te.get_xml_tables(xml_path)
    assert len(tables) == len(captions) == len(footers)
    res = []
    for t, c, f in zip(tables, captions, footers):
        if len(t) == 1 and len(t[0]) == 1: 
            print(pii, t[0][0])
            continue
        max_cols = max(len(r) for r in t)
        for r in t:
            r += [''] * (max_cols - len(r))
        res.append({
            'doi': doi,
            'act_table': t,
            'caption': c if c is not None else '',
            'footer': f if f is not None else dict(),
        })

    assert len(res) > 0
    pickle.dump(res, open(pkl_path, 'wb'))
    pii_table_dict[pii] = res

pickle.dump(pii_table_dict, open(os.path.join(table_dir, 'pii_table_dict.pkl'), 'wb'))
