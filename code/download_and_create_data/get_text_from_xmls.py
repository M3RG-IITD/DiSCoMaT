import os
import pickle

from bs4 import BeautifulSoup
from tqdm import tqdm


table_dir = '../../data'

split_pii_t_idxs = pickle.load(open(os.path.join(table_dir, 'train_val_test_split.pkl'), 'rb'))
split_piis = dict()
for split in split_pii_t_idxs.keys():
    split_piis[split] = list(set(x[0] for x in split_pii_t_idxs[split]))


def get_contents(pii):
    path = os.path.join(table_dir, 'piis', pii, f'{pii}.xml')
    with open(path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file.read(), 'xml')

    sec = soup.find('xocs:item-toc')
    en = sec.findAll('xocs:item-toc-entry')

    snum, sname = [], []
    for s in en:
        try:
            snum.append(s.find('xocs:item-toc-label').contents[0])
            sname.append(s.find('xocs:item-toc-section-title').contents[0])
        except:
            pass

    paper = {
        'Title': ' '.join(soup.find('dc:title').text.split(',')).strip() if soup.find('dc:title') else '',
        'Abstract': soup.find('dc:description').text.replace('Abstract', '').replace('\n', '').strip() if soup.find('dc:description') else '',
    }

    sname.insert(0, 'Abstract')
    snum.insert(0, '')

    all_sections = soup.find_all('ce:section-title')
    for sec in all_sections:
        strr = ''
        if sec.text in sname:
            secid = sname.index(sec.text)
            if '.' not in snum[secid]:
                for tx in sec.find_next_siblings():
                    strr += tx.text.strip().replace('\n', ' ') + '\n'
                paper[f'{snum[secid]}_{sec.text}'] = strr.strip()

    if paper['Abstract'] == '':
        paper['Abstract'] = paper['_Abstract']
        paper.pop('_Abstract')

    return paper


text_data = dict()
for split in ['train', 'val', 'test']:
    for pii in tqdm(split_piis[split]):
        try:
            text_data[pii] = get_contents(pii)
        except AttributeError as e:
            continue

pickle.dump(text_data, open(os.path.join(table_dir, 'train_val_test_paper_data.pkl'), 'wb'))
