import ast
import operator as op
import os
import pathlib
import pickle
import re

from normalize_text import normalize


elements_compounds_path = os.path.join(pathlib.Path(__file__).parent.resolve(), '../data/elements_compounds.pkl')
elements_compounds = pickle.load(open(elements_compounds_path, 'rb'))
all_elements, all_compounds = elements_compounds['elements'], elements_compounds['compounds']

operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv}


def eval_expr(expr):
    return eval_(ast.parse(expr, mode='eval').body)


def eval_(node):
    if isinstance(node, ast.Num):
        return node.n
    elif isinstance(node, ast.BinOp):
        return operators[type(node.op)](eval_(node.left), eval_(node.right))
    else:
        raise TypeError(node)


num = r'(\d*\.\d+|0|[1-9]\d*)'
comp_vars = ['x', 'y', 'z', 'X']
var = r'(' + r'|'.join(comp_vars) + r')'
enum = r'((' + num + var + r')|' + num + r'|' + var + r')'
expr = enum + r'(\s*[\+\-/]\s*' + enum + r')*'
expr = r'((' + num + r'\s*%)|(' + expr + r')|(\(' + expr + r'\)))'


def insert_mult(num_expr):
    if num_expr.endswith('%'):
        return re.match(num, num_expr).group()
    out = ''
    idx = 0
    if num_expr[0] == '(':
        assert num_expr[-1] == ')'
        num_expr = num_expr[1:-1]
    for m in re.finditer(enum, num_expr):
        out += num_expr[idx:m.span()[0]]
        t = num_expr[m.span()[0]:m.span()[1]]
        if t[0] == '0' and len(t) > 1: t = t[1:]
        a = re.match(num + var, t)
        idx = m.span()[1]
        if a is None:
            out += t
        else:
            out += re.search(num, t).group() + '*' + re.search(var, t).group()
    assert idx == len(num_expr)
    assert len(out) > 0
    return out


def norm_sum_to_1(composition, sum_perc):
    assert len(composition) > 0
    corr_comp = []
    for c in composition:
        try:
            val = eval_expr(c[1])
            if val < 0: return []
            if val > 0: corr_comp.append(c)
        except:
            corr_comp.append(c)
    norm_comp = [(c[0], f'({c[1]})/({sum_perc})') for c in corr_comp]
    d = dict()
    for c in norm_comp:
        if c[0] not in d:
            d[c[0]] = c[1].replace(' ', '')
        else:
            d[c[0]] += '+' + c[1].replace(' ', '')
    return [(k, v) for k, v in d.items()]


def process_arg(l):
    l = sorted(l, key=lambda x: -len(x))
    l = [c.replace('(', '\(').replace(')', '\)') for c in l]
    assert len(set(l) - set(all_elements + all_compounds)) == 0
    return l


def x_pattern_1(compounds=None):
    compounds = all_compounds if compounds is None else process_arg(set(compounds) - set(all_elements))
    if len(compounds) == 0: return None, None
    sub_pat = expr + r'?\s*' + r'(' + r'|'.join(compounds) + r')'
    pat = re.compile(r'(?:^|[^\w-])(' + sub_pat + r'(\s*[-*\:\,+]?\s*' + sub_pat + r')+)(?:[^\w-]|$)')
    return re.compile(sub_pat), pat


def x_parse_1(s, sub_pat, pat, check=True, specific=False):
    ress = []
    for m in re.finditer(pat, s):
        res, nums_found = [], 0
        comp = m.group(1)
        for l in re.findall(sub_pat, comp):
            perc = insert_mult(l[0]) if l[0] else '1.0'
            res.append((l[-1], perc))
            if l[0]: nums_found += 1
        if check and nums_found == 0: continue
        res = norm_sum_to_1(res, '+'.join([c[1] for c in res]))
        if len(res) == 0: continue
        percs = ''.join([x[1] for x in res])
        if not any(c in percs for c in comp_vars):
            res = [(x[0], eval_expr(x[1])) for x in res]
        ress.append((res, (m.start(1), m.end(1))))
    return ress


def x_pattern_2(elements=None):
    elements = all_elements if elements is None else process_arg(set(elements) & set(all_elements))
    if len(elements) == 0: return None, None
    sub_pat = r'(' + r'|'.join(elements) + r')\s*' + expr + r'?'
    pat = re.compile(r'(?:^|[^\w-])(' + sub_pat + r'(\s*[-*\:\,+]?\s*' + sub_pat + r')+)(?:[^\w-]|$)')
    return re.compile(sub_pat), pat


def x_parse_2(s, sub_pat, pat, check=True, specific=False):
    ress = []
    all_comp_pat = r'(' + r'|'.join(all_compounds) + ')([^\d\.]|$)'
    for m in re.finditer(pat, s):
        res, nums_found = [], 0
        comp = m.group(1)
        if not specific and re.search(all_comp_pat, comp): continue
        for l in re.findall(sub_pat, comp):
            perc = insert_mult(l[1]) if l[1] else '1.0'
            res.append((l[0], perc))
            if l[1]: nums_found += 1
        if check and nums_found == 0: continue
        res = norm_sum_to_1(res, '+'.join([c[1] for c in res]))
        if len(res) == 0: continue
        percs = ''.join([x[1] for x in res])
        if not any(c in percs for c in comp_vars):
            res = [(x[0], eval_expr(x[1])) for x in res]
        ress.append((res, (m.start(1), m.end(1))))
    return ress


def x_pattern_3(elements=None):
    elements = all_elements if elements is None else process_arg(set(elements) & set(all_elements))
    if len(elements) == 0: return None, None
    sub_pat_1 = r'(' + r'|'.join(elements) + r')\s*' + expr + r'?'
    sub_pat_2 = r'(' + sub_pat_1 + r'(\s*[-*\:\,+]?\s*' + sub_pat_1 + r')*)'
    sub_pat_2_ = sub_pat_1 + r'?(\s*[-*\:\,+]?\s*' + sub_pat_1 + r'?)*?'
    sub_pat_3 = r'((' + sub_pat_2_ + r'|\(\s*' + sub_pat_2_ + r'\s*\)|\[\s*' + sub_pat_2_ + r'\s*\])\s*' + expr + r'?)'
    pat = re.compile(r'(?:^|[^\w-])(' + sub_pat_3 + r'(\s*[-*\:\,+]?\s*' + sub_pat_3 + r')+)(?:[^\w-]|$)')
    sub_pat = [sub_pat_1, sub_pat_2, sub_pat_3]
    sub_pat = [re.compile(x) for x in sub_pat]
    return sub_pat, pat


def x_parse_3(s, sub_pat, pat, specific=False):
    if ('(' not in s or ')' not in s) and ('[' not in s or ']' not in s):
        return []
    sub_pat_1, sub_pat_2, sub_pat_3 = sub_pat
    ress = []
    for m in re.finditer(pat, s):
        comp = m.group(1)
        comp_s = re.sub(expr, ' ', comp)
        if ('(' not in comp_s or ')' not in comp_s) and ('[' not in comp_s or ']' not in comp_s): continue
        res, outer_coeffs = [], []
        fail = False
        for sub_comp in re.findall(sub_pat_3, comp):
            assert sub_comp[0].startswith(sub_comp[1])
            sub_comp_text = sub_comp[1].strip()
            bracket = sub_comp_text[0] in ['(', '[']
            if sub_comp_text[0] == '(':
                assert sub_comp_text[-1] == ')'
                sub_comp_text = sub_comp_text[1:-1].strip()
            elif sub_comp_text[0] == '[':
                assert sub_comp_text[-1] == ']'
                sub_comp_text = sub_comp_text[1:-1].strip()
            outer_coeff = sub_comp[0][len(sub_comp[1]):].strip()
            if bracket and outer_coeff == '':
                fail = True
                break
            outer_coeff = insert_mult(outer_coeff) if outer_coeff else '1.0'
            outer_coeffs.append(outer_coeff)
            parsed_sub_comp = x_parse_2(sub_comp_text, sub_pat_1, sub_pat_2, check=False, specific=specific)
            assert len(parsed_sub_comp) <= 1
            if len(parsed_sub_comp) == 0:
                fail = True
                break
            res += [(x[0], '(' + str(x[1]) +')*(' + outer_coeff + ')') for x in parsed_sub_comp[0][0]]
        if fail: continue
        res = norm_sum_to_1(res, '+'.join(outer_coeffs))
        if len(res) == 0: continue
        percs = ''.join([x[1] for x in res])
        if not any(c in percs for c in comp_vars):
            res = [(x[0], eval_expr(x[1])) for x in res]
        ress.append((res, (m.start(1), m.end(1))))
    return ress


def x_pattern_4(compounds=None):
    compounds = all_compounds if compounds is None else process_arg(set(compounds) - set(all_elements))
    if len(compounds) == 0: return None, None
    sub_pat_1 = expr + r'?\s*(' + r'|'.join(compounds) + r')'
    sub_pat_2 = r'(' + sub_pat_1 + r'(\s*[-*\:\,+]?\s*' + sub_pat_1 + r')*)'
    sub_pat_3 = r'((' + sub_pat_2 + r'|\(\s*' + sub_pat_2 + r'\s*\)|\[\s*' + sub_pat_2 + r'\s*\])\s*' + expr + r')'
    pat = re.compile(r'(?:^|[^\w-])(' + sub_pat_3 + r'(\s*[-*\:\,+]?\s*' + sub_pat_3 + r')+)(?:[^\w-]|$)')
    sub_pat = [sub_pat_1, sub_pat_2, sub_pat_3]
    sub_pat = [re.compile(x) for x in sub_pat]
    return sub_pat, pat


def x_parse_4(s, sub_pat, pat, specific=False):
    if ('(' not in s or ')' not in s) and ('[' not in s or ']' not in s):
        return []
    sub_pat_1, sub_pat_2, sub_pat_3 = sub_pat
    ress = []
    for m in re.finditer(pat, s):
        comp = m.group(1)
        comp_s = re.sub(expr, ' ', comp)
        if ('(' not in comp_s or ')' not in comp_s) and ('[' not in comp_s or ']' not in comp_s): continue
        res, outer_coeffs = [], []
        for sub_comp in re.findall(sub_pat_3, comp):
            assert sub_comp[0].startswith(sub_comp[1])
            sub_comp_text = sub_comp[1].strip()
            if sub_comp_text[0] == '[':
                assert sub_comp_text[-1] == ']'
                sub_comp_text = sub_comp_text[1:-1].strip()
            elif sub_comp_text[0] == '(' and sub_comp_text[-1] == ')':
                sub_comp_text = sub_comp_text[1:-1].strip()
            outer_coeff = insert_mult(sub_comp[0][len(sub_comp[1]):].strip())
            outer_coeffs.append(outer_coeff)
            parsed_sub_comp = x_parse_1(sub_comp_text, sub_pat_1, sub_pat_2, check=False)
            assert len(parsed_sub_comp) == 1
            res += [(x[0], '(' + str(x[1]) +')*(' + outer_coeff + ')') for x in parsed_sub_comp[0][0]]
        res = norm_sum_to_1(res, '+'.join(outer_coeffs))
        if len(res) == 0: continue
        percs = ''.join([x[1] for x in res])
        if not any(c in percs for c in comp_vars):
            res = [(x[0], eval_expr(x[1])) for x in res]
        ress.append((res, (m.start(1), m.end(1))))
    return ress


def x_pattern_5(elements=None):
    elements = all_elements if elements is None else process_arg(set(elements) & set(all_elements))
    if len(elements) == 0: return None, None
    sub_pat_1 = r'(' + r'|'.join(elements) + r')\s*' + expr + r'?'
    sub_pat_2 = r'(' + sub_pat_1 + r'(\s*[-*\:\,+]?\s*' + sub_pat_1 + r')*)'
    sub_pat_3 = r'(' + expr + r'\s*(\(\s*' + sub_pat_2 + r'\s*\)|\[\s*' + sub_pat_2 + '\s*\]))'
    pat = re.compile(r'(?:^|[^\w-])(' + sub_pat_3 + r'(\s*[-*\:\,+]?\s*' + sub_pat_3 + r')+)(?:[^\w-]|$)')
    sub_pat = [sub_pat_1, sub_pat_2, sub_pat_3]
    sub_pat = [re.compile(x) for x in sub_pat]
    return sub_pat, pat


def x_parse_5(s, sub_pat, pat, specific=False):
    if ('(' not in s or ')' not in s) and ('[' not in s or ']' not in s):
        return []
    sub_pat_1, sub_pat_2, sub_pat_3 = sub_pat
    ress, outer_coeffs = [], []
    for m in re.finditer(pat, s):
        comp = m.group(1)
        comp_s = re.sub(expr, ' ', comp)
        if ('(' not in comp_s or ')' not in comp_s) and ('[' not in comp_s or ']' not in comp_s): continue
        res, outer_coeffs = [], []
        fail = False
        for sub_comp in re.findall(sub_pat_3, comp):
            assert sub_comp[0].startswith(sub_comp[1])
            outer_coeff = insert_mult(sub_comp[1].strip())
            outer_coeffs.append(outer_coeff)
            sub_comp_text = sub_comp[0][len(sub_comp[1]):].strip()
            if sub_comp_text[0] == '(':
                assert sub_comp_text[-1] == ')'
                sub_comp_text = sub_comp_text[1:-1].strip()
            elif sub_comp_text[0] == '[':
                assert sub_comp_text[-1] == ']'
                sub_comp_text = sub_comp_text[1:-1].strip()
            parsed_sub_comp = x_parse_2(sub_comp_text, sub_pat_1, sub_pat_2, check=False, specific=specific)
            assert len(parsed_sub_comp) <= 1
            if len(parsed_sub_comp) == 0:
                fail = True
                break
            res += [(x[0], '(' + str(x[1]) +')*(' + outer_coeff + ')') for x in parsed_sub_comp[0][0]]
        if fail: continue
        res = norm_sum_to_1(res, '+'.join(outer_coeffs))
        if len(res) == 0: continue
        percs = ''.join([x[1] for x in res])
        if not any(c in percs for c in comp_vars):
            res = [(x[0], eval_expr(x[1])) for x in res]
        ress.append((res, (m.start(1), m.end(1))))
    return ress


def x_pattern_6(compounds=None):
    compounds = all_compounds if compounds is None else process_arg(set(compounds) - set(all_elements))
    if len(compounds) == 0: return None, None
    sub_pat_1 = expr + r'?\s*(' + r'|'.join(compounds) + r')'
    sub_pat_2 = r'(' + sub_pat_1 + r'(\s*[-*\:\,+]?\s*' + sub_pat_1 + r')*)'
    sub_pat_2_ = r'(' + sub_pat_1 + r'(\s*[-*\:\,+]?\s*' + sub_pat_1 + r')*?)'
    sub_pat_3 = r'(' + expr + '\s*(' + sub_pat_2_ + r'|\(\s*' + sub_pat_2_ + r'\s*\)|\[\s*' + sub_pat_2_ + '\s*\]))'
    pat = re.compile(r'(?:^|[^\w-])(' + sub_pat_3 + r'(\s*[-*\:\,+]?\s*' + sub_pat_3 + r')+)(?:[^\w-]|$)')
    sub_pat = [sub_pat_1, sub_pat_2, sub_pat_3]
    sub_pat = [re.compile(x) for x in sub_pat]
    return sub_pat, pat


def x_parse_6(s, sub_pat, pat, specific=False):
    if ('(' not in s or ')' not in s) and ('[' not in s or ']' not in s):
        return []
    sub_pat_1, sub_pat_2, sub_pat_3 = sub_pat
    ress, outer_coeffs = [], []
    for m in re.finditer(pat, s):
        comp = m.group(1)
        comp_s = re.sub(expr, ' ', comp)
        if ('(' not in comp_s or ')' not in comp_s) and ('[' not in comp_s or ']' not in comp_s): continue
        res, outer_coeffs = [], []
        for sub_comp in re.findall(sub_pat_3, comp):
            assert sub_comp[0].startswith(sub_comp[1])
            outer_coeff = insert_mult(sub_comp[1].strip())
            outer_coeffs.append(outer_coeff)
            sub_comp_text = sub_comp[0][len(sub_comp[1]):].strip()
            if sub_comp_text[0] == '(':
                assert sub_comp_text[-1] == ')'
                sub_comp_text = sub_comp_text[1:-1].strip()
            elif sub_comp_text[0] == '[':
                assert sub_comp_text[-1] == ']'
                sub_comp_text = sub_comp_text[1:-1].strip()
            parsed_sub_comp = x_parse_1(sub_comp_text, sub_pat_1, sub_pat_2, check=False)
            assert len(parsed_sub_comp) == 1
            res += [(x[0], '(' + str(x[1]) +')*(' + outer_coeff + ')') for x in parsed_sub_comp[0][0]]
        res = norm_sum_to_1(res, '+'.join(outer_coeffs))
        if len(res) == 0: continue
        percs = ''.join([x[1] for x in res])
        if not any(c in percs for c in comp_vars):
            res = [(x[0], eval_expr(x[1])) for x in res]
        ress.append((res, (m.start(1), m.end(1))))
    return ress


patterns = [x_pattern_1, x_pattern_2, x_pattern_4, x_pattern_6, x_pattern_3, x_pattern_5]
parses = [x_parse_1, x_parse_2, x_parse_4, x_parse_6, x_parse_3, x_parse_5]


def do_overlap(span_1, span_2):
    assert len(span_1) == 2 and len(span_2) == 2
    return span_2[0] < span_1[1] and span_1[0] < span_2[1]


def is_inside(sub_span, span):
    assert len(span) == 2 and len(sub_span) == 2
    return span[0] <= sub_span[0] and sub_span[1] <= span[1]


def parse_composition(s, compounds=None):
    s = s.replace('·', '*').replace('—', '-')
    s = normalize(s).replace('\n', ' ')
    res = []
    for pattern, parse in zip(patterns, parses):
        sub_pat, pat = pattern(compounds)
        if sub_pat is None and pat is None: continue
        curr_res = parse(s, sub_pat, pat, specific=compounds is not None)
        for x in curr_res:
            assert len(x) == 2 and len(x[1]) == 2
            st, en = x[1]
            while s[st] == ' ': st += 1
            while s[en-1] == ' ': en -= 1
            x = (x[0], (st, en))
            overlap_idxs = []
            for i, y in enumerate(res):
                if do_overlap(x[1], y[1]):
                    overlap_idxs.append(i)
            if len(overlap_idxs) == 0:
                res.append(x)
                continue
            inside = False
            for idx in overlap_idxs:
                if is_inside(x[1], res[idx][1]):
                    inside = True
                    break
            if inside: continue
            for idx in overlap_idxs:
                if not is_inside(res[idx][1], x[1]):
                    inside = True
                    break
            if inside: continue
            for idx in sorted(overlap_idxs, reverse=True):
                del res[idx]
            res.append(x)
    return res

