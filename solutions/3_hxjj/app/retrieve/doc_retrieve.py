from typing import Dict, DefaultDict,  List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import re
import json
import glob
from utils.file_processor import read_jsonl, write_jsonl
from config import PDF_TXT_PATH

# bm25
def bm25_retrieve(query, contents):
    bm25 = BM25Okapi(contents)

    # 对于每个文档，计算结合BM25
    bm25_scores = bm25.get_scores(query)

    # 根据得分排序文档
    sorted_docs = sorted(zip(contents, bm25_scores), key=lambda x: x[1], reverse=True)
    return sorted_docs



    
class DocRetrieve:
    def __init__(self,
                 top_k: int = 15,
                 txt_folder: str = PDF_TXT_PATH
    ):
        self.top_k = top_k
        self.txt_folder = txt_folder
        self.doc_dict = dict()
        self.company_list = []
        self.get_all_company()

    def get_all_company(self):
        file_paths = glob.glob(f'{self.txt_folder}/*')
        for file_path in file_paths:
            file_name = re.split('\\\|/',file_path)[-1].replace(".txt","")
            self.company_list.append(file_name)

    def check_company(self, company, query):
        for cc in self.company_list:
            if cc in query:
                if len(cc) >= len(company):
                    return cc, query.replace(cc,'')
                else:
                    return cc, query.replace(company,'')

        query = query.replace(company, '')
        new_company, score = bm25_retrieve(company, self.company_list)[0]
        company_counts = []
        if score < 9:
            for cc in self.company_list:
                if self.doc_dict.get(cc) is None:
                    self.construct_docs(cc)
                c_count = 0
                for one_doc in self.doc_dict[cc]:
                    c_count += one_doc.page_content.count(company)
                company_counts.append((c_count, cc))
            company_counts.sort(reverse=True)
            return company_counts[0][1], query
        else:
            return new_company, query
        


    def construct_docs(self, company):
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n"],
                                                       keep_separator=True,
                                                       chunk_size=200,
                                                       chunk_overlap=100,
                                                       length_function=len)
        file_path = self.txt_folder + company + '.txt'
        
        content = read_jsonl(file_path)
        full_text = ""
        for line in content:
            text = line.get("inside", "")
            text_type = line.get("type", -2)

            if text_type in ("页眉", "页脚") or text == "":
                continue
            elif text_type == "excel":
                full_text += "\t".join(text) + '\n'
            else:
                full_text += text + "\n"
        self.doc_dict[company] = text_splitter.create_documents([full_text])


    def search(self, company, query, score_func=bm25_retrieve):
        if self.doc_dict.get(company) is None:
            self.construct_docs(company)
        contents = [doc.page_content for doc in self.doc_dict[company]]
        return "\n\n".join([f'检索内容{i+1}开始:\n. {retrieve_content[0]}\n检索内容{i+1}结束.' for i,retrieve_content in enumerate(score_func(query,contents)[:self.top_k])])
    
    def get_single_extra_knowledge(self, company, query):
        company, question = self.check_company(company, query)
        return self.search(company, question)
