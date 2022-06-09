import requests
import os
import pickle

from bs4 import BeautifulSoup
from tqdm import tqdm


table_dir = '../../data'
api_key = '' # Elsevier api key. Obtain from https://dev.elsevier.com/


def article_downloader(doi_list, output_dir):
    '''
    Function to download XMLs from DOIs of Research Papers published in Elsevier journals.
            Parameters:
                    doi_list (list): List of doi strings
                    output_dir (str): Path of directory where papers will be downloaded
            Returns:
                    Downloads the research paper XML in a folder named with article pii inside output_dir
    '''
    os.makedirs(output_dir, exist_ok=True)
    for doi in tqdm(doi_list):
        xml_url = f'https://api.elsevier.com/content/article/doi/{doi}?APIKey={api_key}'
        headers = {'User-Agent': 'Mozilla/5.0'}
        url_get = requests.get(xml_url, headers=headers)
        if url_get.status_code == 200:
            soup = BeautifulSoup(url_get.content, 'lxml')
            pii = soup.find('xocs:pii-unformatted').text
            soup_path = os.path.join(output_dir, str(pii))
            xmlpath = os.path.join(soup_path, str(pii) + '.xml')
            os.makedirs(soup_path, exist_ok=True)
            with open(xmlpath, 'w', encoding='utf-8') as file:
                file.write(str(soup))



dois_list = pickle.load(open(os.path.join(table_dir, 'all_dois.pkl'), 'rb'))
output_dir = os.path.join(table_dir, 'piis')
article_downloader(dois_list, output_dir)

