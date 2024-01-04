import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from tqdm import tqdm
import jsonlines
import math
import re

from modelscope import AutoModelForCausalLM, AutoTokenizer, snapshot_download
from modelscope import GenerationConfig
import thulac
from setting import *

def lcs_seq_num(indices):
    c = 0
    for i in range(1, len(indices)):
        if indices[i] - indices[i - 1] > 1:
            c += 1
    return len(indices) - c


def construct_lcs_with_indices(X, Y, L):
    lcs = []
    indices = []
    i, j = len(X), len(Y)

    while i > 0 and j > 0:
        if X[i - 1] == Y[j - 1]:
            lcs.append(X[i - 1])
            indices.append(j - 1)  # 添加当前字符在Y中的下标
            i -= 1
            j -= 1
        elif L[i - 1][j] > L[i][j - 1]:
            i -= 1
        else:
            j -= 1

    # 反转lcs和indices列表以获得正确的顺序
    return ''.join(reversed(lcs)), list(reversed(indices))


def lcs_with_character_indices(X, Y):
    m, n = len(X), len(Y)
    L = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    lcs_seq, indices = construct_lcs_with_indices(X, Y, L)

    return len(lcs_seq), lcs_seq, indices


def close_match(q, indic, n):
    indic_re = []
    for key in indic:
        r_len, lcs_seq, indices = lcs_with_character_indices(q, key)
        seq_num = lcs_seq_num(indices)
        indic_re.append((key, r_len + seq_num))
    indic_re.sort(key=lambda x: x[1], reverse=True)
    return indic_re[:n]


def extract_inc_name(question, docs):
    name = re.findall(r'(?:.*，)?(.*?)(?:股份)?有限公司', question)
    if len(name) == 0:
        name = re.findall(r'(?:.*，)?(.*?)股份', question)
    if len(name) == 0:
        name = question
    else:
        name = name[0]
    inc = close_match(name, docs.keys(), n=1)
    return inc[0][0]


def extrac_doc_page(thu, stop_words, doc, question):
    seg_list = thu.cut(question.replace('.', '点'))
    seg_list = [x[0] for x in seg_list]
    key_words = list(set(seg_list) - stop_words)
    key_word_tf = {key: 0 for key in key_words}
    key_word_idf = {key: [] for key in key_words}
    for idx, p in enumerate(doc):
        finds = re.findall('|'.join(key_words), p.page_content)
        for r in finds:
            key_word_tf[r] += 1
            key_word_idf[r].append(idx)
    key_tfidf = []
    for key in key_words:
        # tf = key_word_tf[key]
        tf = 1
        idf = math.log(len(doc) / (len(set(key_word_idf[key])) + 1))
        key_tfidf.append((key, tf * idf))
    key_tfidf.sort(key=lambda x: x[1], reverse=True)
    doc_p = [[i, 0] for i in range(len(doc))]
    for key in key_tfidf:
        p_id = list(set(key_word_idf[key[0]]))
        for p in p_id:
            doc_p[p][1] += key[1]
    doc_p.sort(key=lambda x: x[1], reverse=True)
    after_rank_doc = []
    for i in doc_p[0:20]:
        after_rank_doc.append(doc[i[0]].page_content)
    return '\n\n'.join(after_rank_doc), key_tfidf, doc_p


def read_jsonl(path):
    content = []
    with jsonlines.open(path, "r") as json_file:
        for obj in json_file.iter(type=dict, skip_invalid=True):
            content.append(obj)
    return content


def write_jsonl(path, content):
    with jsonlines.open(path, "w") as json_file:
        json_file.write_all(content)

### 文档分片 ####
def doc_faiss_create(fpath):
    files = os.listdir(fpath)
    docs = {}
    for fn in tqdm(files):
        with open(os.path.join(fpath, fn)) as f:
            txt = f.read()
            r = re.findall('(?:^|：)(.*?)股份有限公司', txt)
            if len(r) == 0:
                r = re.findall('(?:^|：|\n)(.*?)股份有限公司', txt)[:3]
            assert len(r) != 0
            new = []
            for n in r:
                if '证券' not in n:
                    new.append(n)
            r = new
        raw_documents = TextLoader(os.path.join(fpath, fn)).load()
        txt = []
        for line in raw_documents[0].page_content.split('\n'):
            if '......' in line:
                continue
            else:
                txt.append(line)

        raw_documents[0].page_content = ''.join(txt)

        text_splitter = CharacterTextSplitter('，', chunk_size=200, chunk_overlap=0)
        documents = text_splitter.split_documents(raw_documents)
        docs[','.join(r)] = documents
    return docs

def text_solution(fpath):
    model_dir = os.path.join(models_dir,'Tongyi-Finance-14B-Chat')
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda:0", trust_remote_code=True).eval()
    model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)
    docs = doc_faiss_create(os.path.join(data_dir,'bs_challenge_financial_14b_dataset/pdf_txt_file'))

    prompt = '''你是一个能精准提取文本信息并回答问题的AI。请你用下面提供的材料回答问题，材料是：```%s```。
    请根据以上材料回答问题："%s"。如果能根据给定材料回答，则提取出最合理的答案来回答问题,并回答出完整内容，如果不能找到答案，则回答“无法回答”，需要将问题复述一遍，现在请输出答案：'''
    answers = []
    stop_words = set(open('/app/baidu_stopwords.txt').read().split('\n') +
                     open('/app/hit_stopwords.txt').read().split('\n')
                     + ['，', '？', '。'])
    thu = thulac.thulac(seg_only=True)
    ques = read_jsonl(os.path.join(data_dir,qfpath))

    for q in ques:
        if re.search(r'(?:股票|基金|A股|港股)', q['question']):
            continue
        print(q['id'])
        name = extract_inc_name(q['question'], docs)
        doc = docs[name]

        prompt_inc = '''你是一个能精准提取公司名称的AI，公司名称类似`武汉兴图新科电子股份有限公司`，如果无法提取请输出`股份有限公司`。从下面文本中抽取公司名称:`{}`，
        注意文本只能来源于上述文本,现在请抽取：'''
        response, history = model.chat(tokenizer, prompt_inc.format(q['question']), history=None, temperature=0.0,
                                       do_sample=False)
        qu = q['question'].replace(response, '')
        if qu == '':
            qu = q['question'].replace('股份', '').replace('有限公司', '')
        refer_txt, key_tfidf, doc_p = extrac_doc_page(thu, stop_words, doc, qu)

        response, history = model.chat(tokenizer, prompt % (refer_txt[:3000], q['question']), history=None,
                                       temperature=0.0, do_sample=False)
        if '无法' in response:
            retry = 0
            while retry < 3:
                response, history = model.chat(tokenizer, prompt % (
                refer_txt[retry * 2000: (retry + 1) * 2000], q['question']), history=None,
                                               temperature=0.0, do_sample=False)
                if '无法' not in response:
                    break
                retry += 1

        q['answer'] = response
        answers.append(q)
        print(q['question'], '   回答：', response)


    write_jsonl(fpath, ques)
