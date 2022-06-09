"""
---------------------------------------------------------------------------------------------------------
The table extraction code has been adapted from the following repository: https://github.com/olivettigroup/table_extractor
The license for the same is provided below.
---------------------------------------------------------------------------------------------------------
MIT License

Copyright (c) 2018

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
---------------------------------------------------------------------------------------------------------
"""

import traceback
from bs4 import BeautifulSoup
import unidecode
import sys

from scipy import stats
from html.parser import HTMLParser
import traceback


class TableExtractor(object):
    def __init__(self):
        return

    def get_caption(self, table, format):
        if format == 'xml':
            if '10.1016' in self.doi:
                caption = table.find('caption')
                caption, ref = self._search_for_reference(caption, format)
                caption = unidecode.unidecode(HTMLParser().unescape(caption.text)).strip()
                return caption, ref
            elif '10.1021' in self.doi:
                caption = table.find('title')
                if caption is None:
                    up = table.parent
                    caption = table.find('title')
                    if caption is None:
                        caption = up.find('caption')
                caption, ref = self._search_for_reference(caption, format)
                caption = unidecode.unidecode(HTMLParser().unescape(caption.text)).strip()
                return caption, ref
        else:
            raise NotImplementedError      
        return '', []

    def get_footer(self, table, format):
        footer_dict = dict()
        if format == 'xml':
            if '10.1016' in self.doi:
                footer = table.find_all('table-footnote')
                if len(footer) > 0:
                    for f in footer:
                        sup = f.find('label')
                        if sup is not None:
                            dt = sup.text
                            f.label.decompose()
                        else:
                            dt = 'NA'
                        footer_dict[dt.strip()] = unidecode.unidecode(HTMLParser().unescape(f.text)).strip()
                else:
                    footer = table.find('legend')
                    if footer is None: return None
                    else:
                        all = footer.find_all('simple-para')
                        for a in all:
                            sup = a.find('sup')
                            if sup is not None:
                                dt = sup.text
                                a.sup.decompose()
                            else:
                                dt = 'NA'
                            footer_dict[dt.strip()] = unidecode.unidecode(HTMLParser().unescape(a.text)).strip()
            elif '10.1021' in self.doi:
                up = table.parent
                footer = up.find('table-wrap-foot')
                if footer is not None:
                    dts, dds = footer.find_all('label'), footer.find_all('p')
                    if len(dts) != len(dds):
                        ts = footer.find_all('sup')
                        dts = []
                        for t in ts:
                            if t.text != '':
                                dts.append(t)
                        if len(dds) == 1 and len(dts) > 1:
                            para = dds[0]
                            cont = para.contents
                            c = []
                            for co in cont:
                                try:
                                    c.append(co.text)
                                except:
                                    c.append(co)
                            ind = [i for i,x in enumerate(c) if x == '']
                            dts, dds = [], []
                            curr = ind[0]
                            for i in ind[1:]:
                                dts.append(c[curr-1])
                                dds.append(''.join(c[(curr+1):(i-1)]))
                                curr = i
                            dts.append(c[curr-1])
                            dds.append(''.join(c[(curr+1):]))
                            for d, t in zip(dds, dts):
                                footer_dict[t.strip()] = unidecode.unidecode(HTMLParser().unescape(d)).strip().replace('\n', ' ')

                        elif len(dts) != len(dds):
                            print('Problem in Footer: Keys and paragraphs len dont match')
                            return None
                        else:
                            for d, t in zip(dds, dts):
                                footer_dict[t.text.strip()] = unidecode.unidecode(HTMLParser().unescape(d.text)).strip().replace('\n', ' ')
                    else:
                        for d, t in zip(dds, dts):
                            footer_dict[t.text.strip()] = unidecode.unidecode(HTMLParser().unescape(d.text)).strip().replace('\n', ' ')
                else: return None
        else:
            raise NotImplementedError
        return footer_dict
                    
    def get_xml_tables(self, xml):
        all_tables = []
        all_captions = []
        all_footers = []
        
        soup = BeautifulSoup(open(xml, 'r+'), 'xml')
        tables = soup.find_all('table')
        for table in tables:
            try:
                caption, footer = None, None

                try:
                    caption = self.get_caption(table, format='xml')[0]
                except Exception as e:
                    print(e, 'Problem in caption')

                try:
                    footer = self.get_footer(table, format='xml')
                except Exception as e:
                    print(e, 'problem in footer')

                tab = []
                for t in range(400):
                    tab.append([None] * 400)
                rows = table.find_all('row')
                for i, row in enumerate(rows):
                    counter = 0
                    for ent in row:
                        curr_col, beg, end, more_row = 0, 0, 0, 0
                        if type(ent) == type(row):
                            if ent.has_attr('colname'):
                                try:
                                    curr_col = int(ent['colname'])
                                except:
                                    assert ent['colname'].startswith('col')
                                    curr_col = int(ent['colname'][3:])
                                    assert curr_col >= 0
                            if ent.has_attr('namest'):
                                try:
                                    beg = int(ent['namest'])
                                except:
                                    assert ent['namest'].startswith('col')
                                    beg = int(ent['namest'][3:])
                                    assert beg >= 0
                            if ent.has_attr('nameend'):
                                try:
                                    end = int(ent['nameend'])
                                except:
                                    assert ent['nameend'].startswith('col')
                                    end = int(ent['nameend'][3:])
                                    assert end >= 0
                            if ent.has_attr('morerows'):
                                try:
                                    more_row = int(ent['morerows'])
                                except:
                                    assert ent['morerows'].startswith('col')
                                    more_row = int(ent['morerows'][3:])
                                    assert more_row >= 0
                            ent = self._search_for_reference(ent, 'xml')[0]
                            if beg != 0 and end != 0 and more_row != 0:
                                for j in range(beg, end+1):
                                    for k in range(more_row+1):
                                        tab[i+k][j-1] = unidecode.unidecode(HTMLParser().unescape(ent.get_text())).strip().replace('\n', ' ')
                                        # sup_tab[i+k][j-1] = curr_ref
                            elif beg != 0 and end != 0:
                                for j in range(beg, end+1):
                                    tab[i][j-1] = unidecode.unidecode(HTMLParser().unescape(ent.get_text())).strip().replace('\n', ' ')
                            elif more_row != 0:
                                for j in range(more_row+1):
                                    tab[i+j][counter] = unidecode.unidecode(HTMLParser().unescape(ent.get_text())).strip().replace('\n', ' ')
                            elif curr_col != 0:
                                tab[i][curr_col-1] = unidecode.unidecode(HTMLParser().unescape(ent.get_text())).strip().replace('\n', ' ')
                            else:
                                counter_ent = counter
                                found = False
                                while not found:
                                    if tab[i][counter_ent] is None:
                                        tab[i][counter_ent] = unidecode.unidecode(HTMLParser().unescape(ent.get_text())).strip().replace('\n', ' ')
                                        found = True
                                    else:
                                        counter_ent+=1
                                counter = counter_ent
                            counter = counter + 1 + (end - beg)
                for t in tab:
                    for j in reversed(t):
                        if j is None:
                            t.remove(j)
                for t in reversed(tab):
                    if len(t) == 0:
                        tab.remove(t)
                lens = []
                for t in tab:
                    lens.append(len(t))
                size = stats.mode(lens)[0][0]
                for t in tab:
                    if len(t) != size:
                        for _ in range(len(t), size):
                            t.append('')
                all_tables.append(tab)
                all_captions.append(caption)
                all_footers.append(footer)
            except:
                print('Failed to extract XML table')
                tb = sys.exc_info()[-1]
                print(traceback.extract_tb(tb, limit=1)[-1][1])
        return all_tables, all_captions, all_footers

    def _search_for_reference(self, soup, format):
        if format == 'xml':
            ref = soup.find_all('xref')
            tags = []
            if len(ref) == 0:
                if soup.name == 'caption':
                    return soup, tags
                ref = soup.find_all('sup')
                for r in ref:
                    text = r.text.split(',')
                    for t in text:
                        if len(t) == 1 and t.isalpha():
                            tags.append(t)
                            soup.sup.decompose()
                return soup, tags
            else: 
                for r in ref:
                    if len(r.text) < 4:
                        tag = soup.xref.extract()
                        tags.append(tag.text)
                return soup, tags
        else:
            raise NotImplementedError

