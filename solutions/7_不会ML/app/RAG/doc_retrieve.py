import sys
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from bm25 import BM25
from model_api.call_api import call_embedding,call_rerank
import torch
import faiss
import time
from Config.set_config import Config
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
import pickle

def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def text2vector(text_list): #文本向量化函数,text_list是已经切分好的文本块
    result = dict()
    index = 1
    for doc in tqdm(text_list):
        # doc是索引字典{index,doc}
        result[ doc] = np.array(call_embedding(doc))
        # result.append(call_embedding(doc))
       
        # print('当前已经embedding处理完第{}块文本'.format(index))
        index+=1
            
    return result
    
### 采用BM25与embedding集成检索----双路检索==========>在对检索到的文本进行重排
### 采用父文档检索法
class doc_retrive():
    def __init__(self,top_k:int=15,parent_chunk_size:int=800,chunk_size:int=150,chunk_overlap:int=50,query:str='',use_embedding=True,rerank_top_k:int=6,use_api=False,is_first=True):
       
        self.top_k = top_k
        self.end_top_k = rerank_top_k
        self.company_list = []
        self.txt_path = Config.txt_path
        # self.text_ner_path = Config.text_ner_path
        self.query = query
        self.embedding_dict = dict()
        self.chunk_size = chunk_size
        self.chunk_overlap=chunk_overlap
        self.use_embedding = use_embedding    
        self.company_list_embedding = dict()
        self.embedding_doc_res = dict()
        self.use_api =use_api
        self.query_embedding = None
        self.parent_size =parent_chunk_size # 父文档chunk大小
        self.is_first = is_first #是否是第一次使用该参数（切分参数）
        self.use_bm25 = True # 默认使用bm25稀疏检索
        self.get_all_companies()

    def get_all_companies(self):
        self.company_list = [f for f in os.listdir(self.txt_path) if f.endswith('.txt') and not f.startswith('.')] # 避免出现 .为首的隐藏文件
        # 对每个txt名称进行embedding,防止每次query都进行embedding,节省时间与内存
        if self.use_embedding:
            print('*****************对用户输入问题进行embedding*************************')
            self.query_embedding = np.array(call_embedding(self.query))
            if os.path.exists(Config.embedding_company_name_store):
                print('当前向量库已经存在公司名称向量结果pkl,开始加载pkl文件')
                with open(Config.embedding_company_name_store,'rb') as file:
                    self.company_list_embedding = pickle.load(file)
            else:
                print('第一次存储公司名称向量化pkl文件')    
                for company_txt in self.company_list:
                    self.company_list_embedding[company_txt]=np.array(call_embedding(company_txt))
                # 将字典存储为 .pkl 文件
                with open(Config.embedding_company_name_store, 'wb') as file:
                    pickle.dump(self.company_list_embedding, file)
                print("字典已成功存储为 pkl 文件")
            if self.is_first:#如果是第一次使用类中的参数（包括切分文本细节的），则进行embedding，方便后续存储调用
                print('************第一次使用上述参数切分，开始对txt文件进行embedding**************')
                for txt_file in self.company_list:
                    
                    cur_file_path = os.path.join(self.txt_path,txt_file)
                    
                    with open(cur_file_path,'r',encoding='utf-8') as file:
                        cur_text = file.read()
                    #先切分为大文本
                    text_splitter=RecursiveCharacterTextSplitter(chunk_size=self.parent_size,chunk_overlap=100,separators=["\n"],keep_separator=True,length_function=len)
                    parent_docs = text_splitter.split_text(cur_text)
                    #再切分小文本
                    cur_text = []
                    index = 0
                    for doc in parent_docs:
                        small_doc = self.text_split(doc) #切分为子文本块
                        cur_text+=small_doc
                        index+=1
                    print('当前正在向量化存储文件：',cur_file_path)
                    cur_vector_dict = text2vector(cur_text)
                    # 将字典存储为 .pkl 文件
                    save_path = os.path.join(Config.embedding_vector_store,txt_file.split('.')[0]+'.pkl')
                    with open(save_path, 'wb') as file:
                        pickle.dump(cur_vector_dict, file)
                    print('当前向量化数据已经存储至：',save_path)
    def text_split(self,content): #子文本块大小切分准则
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,chunk_overlap=self.chunk_overlap,separators=["\n",],keep_separator=True,length_function=len)
        return text_splitter.split_text(content)
    def parent_split(self,content):# 构建父文本块,建立父节点查询索引
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=self.parent_size,chunk_overlap=100,separators=["\n"],keep_separator=True,length_function=len)
        return text_splitter.split_text(content)
        # parent_docs = text_splitter.split_text(content)
        # parent_doc_index = [(i,parent_docs[i]) for i in range(len(parent_docs))]
        # return parent_doc_index,parent_docs  #返回父文本索引以及对应的文本块
        
    def bm25_retrieve_txt(self,query,contents): # contents是列表，用于检索与用户问题匹配的txt文件名称
        bm_25 = BM25(contents)
        stop_words = ['有限公司','科技','；',';','.','。','哪些','的','年度','比例','股份','什么','？','报告期内','主营业务','收入','分别',
                      '根据','招股书','在景观亮化半导体照明领域的市场占有率为多少？']
        for word in stop_words:
            query.replace(word,'')
        bm25_scores = bm_25.get_scores(query)
         # 根据得分排序文档
        sorted_docs = sorted(zip(contents, bm25_scores), key=lambda x: x[1], reverse=True)
        return sorted_docs
    def embedding_retrieve_txt(self,query_embedding): # 利用稠密检索的方式获得与query匹配的txt文件
        similarity = {}
        for k,v in self.company_list_embedding.items():
            similarity[k] = v@query_embedding.T
        similarity = dict(sorted(similarity.items(),key = lambda x:x[1],reverse=True))
        return next(iter(similarity)) #取排名第一个的匹配结果
    
    def get_prompt_docs(self,ner_res): #ner_res是LORA模型进行命名实体识别后的结果
        # 首先根据query搜索最相关的company   系统默认使用BM25进行稀疏检索
        if self.use_embedding: #如果设置稠密检索为真，则默认进行双路检索
            sorted_file_name_embedding = self.embedding_retrieve_txt(self.query_embedding)
            sorted_file_name_bm25 = self.sorted_file_name = self.bm25_retrieve_txt(self.query,self.company_list)[0][0] #规定bm25算法搜索一个
            if sorted_file_name_bm25 == sorted_file_name_embedding: #如果匹配结果相同，则直接返回
                print('当前问题匹配txt文件为：',sorted_file_name_bm25)
                # 读取txt文件，构建父文本块
                with open(os.path.join(Config.txt_path,sorted_file_name_bm25),'r',encoding='utf-8') as file:
                    data = file.read()
                parent_docs = self.parent_split(data) #父文档
                # 开始对embedding向量进行相似度计算
                #step1,打开相应的pkl文件
                pkl_file = os.path.join(Config.embedding_vector_store,sorted_file_name_bm25.replace('txt','pkl'))
                with open(pkl_file,'rb') as file:
                    docs_dict = pickle.load(file)
                print('当前匹配向量数据库为：',pkl_file)
                self.chunk_docs = list(docs_dict.keys())
                index_file = os.path.join(Config.embedding_index_path,sorted_file_name_bm25.replace('txt','faiss'))
                if os.path.exists(index_file):
                    print('当前数据库中已存在faiss文件，无需再次生成faiss文件，开始读取index')
                    index = faiss.read_index(index_file)
                else:
                    # print('当前数据库中不存在相应的faiss文件，开始生成faiss文件')
                    # 将所有的向量从字典中提取出来，形成一个大的NumPy数组
                    cur_vectors = np.array(list(docs_dict.values()))
                    # 获取向量的维度
                    dim = cur_vectors.shape[1]
                    # 创建一个索引，这里使用L2距离的简单索引
                    index = faiss.IndexFlatL2(dim)
                    # 将所有向量添加到索引中
                    index.add(cur_vectors)
                    # faiss.write_index(index, index_file)
                # 进行向量化搜索
                
                _, I = index.search(self.query_embedding.reshape(1, -1), self.top_k) 
            
                embed_select_docs = [self.chunk_docs[i] for i in I[0]] #存储为列表
                company_name,keywords = self.get_question_key(ner_res)
                bm25_selected_docs = [ docs_tuple[0] for docs_tuple  in self.bm25_retrieve_txt(keywords,self.chunk_docs)[:self.top_k]]
                #合并双路检索文档，并去重
                first_docs_select = list(set(bm25_selected_docs+embed_select_docs))
                # 利用重排模型对检索结果进行重排
                end_similarity = {}
                for chunk in first_docs_select:
                    end_similarity[chunk] = call_rerank([self.query,chunk])
                # 根据重排的子文档确定最终选择的子文档
                end_select_docs = []
                for key,_ in sorted(end_similarity.items(),key=lambda x:x[1],reverse=True)[:self.end_top_k]:
                    for index,parent_doc in enumerate(parent_docs): # 遍历父文本块
                        if key in parent_doc and parent_doc not in end_select_docs: #根据匹配的子文本块找到父文本
                            end_select_docs.append(parent_doc)
                
                    
                    
            else:#如果匹配结果不同，则结合两种算法匹配的结果
                # 读取txt文件，构建父文本块
                with open(os.path.join(Config.txt_path,sorted_file_name_bm25),'r',encoding='utf-8') as file:
                    data = file.read()
                with open(os.path.join(Config.txt_path,sorted_file_name_embedding),'r',encoding='utf-8') as file:
                    data += file.read()
                parent_docs = self.parent_split(data) #父文档
                # 开始对embedding向量进行相似度计算
                #step1,打开相应的pkl文件
                pkl_file_bm25 = os.path.join(Config.embedding_vector_store,sorted_file_name_bm25.replace('txt','pkl'))
                pkl_file_embed = os.path.join(Config.embedding_vector_store,sorted_file_name_embedding.replace('txt','pkl'))
                with open(pkl_file_bm25,'rb') as file:
                    docs_dict_bm25 = pickle.load(file)
                
                with open(pkl_file_embed,'rb') as file:
                    docs_dict_embed = pickle.load(file)
                print('当前匹配向量数据库为：',pkl_file)
                self.chunk_docs = list(docs_dict_bm25.keys())+list(docs_dict_embed.keys())
                index_file = os.path.join(Config.embedding_index_path,sorted_file_name_bm25.replace('txt','faiss'))
                if os.path.exists(index_file):
                    print('当前数据库中已存在faiss文件，无需再次生成faiss文件，开始读取index')
                    index = faiss.read_index(index_file)
                else:
                    # print('当前数据库中不存在相应的faiss文件，开始生成faiss文件')
                    # 将所有的向量从字典中提取出来，形成一个大的NumPy数组
                    cur_vectors = np.array(list(docs_dict_bm25.values())+list(docs_dict_embed.values()))
                    # 获取向量的维度
                    dim = cur_vectors.shape[1]
                    # 创建一个索引，这里使用L2距离的简单索引
                    index = faiss.IndexFlatL2(dim)
                    # 将所有向量添加到索引中
                    index.add(cur_vectors)
                    # faiss.write_index(index, index_file)
                # 进行向量化搜索
                _, I = index.search(self.query_embedding.reshape(1, -1), self.top_k) 
            
                embed_select_docs = [self.chunk_docs[i] for i in I[0]] #存储为列表
                company_name,keywords = self.get_question_key(ner_res)
                bm25_selected_docs = [ docs_tuple[0] for docs_tuple  in self.bm25_retrieve_txt(keywords,self.chunk_docs)[:self.top_k]]
                
                #合并双路检索文档，并去重
                first_docs_select = list(set(bm25_selected_docs+embed_select_docs))
                
                # 利用重排模型对检索结果进行重排
                end_similarity = {}
                for chunk in first_docs_select:
                    end_similarity[chunk] = call_rerank([self.query,chunk])
                # 根据重排的子文档确定最终选择的子文档
                end_select_docs = []
                for key,_ in sorted(end_similarity.items(),key=lambda x:x[1],reverse=True)[:self.end_top_k]:
                    for index,parent_doc in enumerate(parent_docs): # 遍历父文本块
                        if key in parent_doc and parent_doc not in end_select_docs: #根据匹配的子文本块找到父文本
                            end_select_docs.append(parent_doc)
                
                
        else: #如果不使用向量搜索
            sorted_file_name_bm25 =  self.bm25_retrieve_txt(self.query,self.company_list)[0][0] #规定bm25算法搜索一个
            #将公司名称的txt文件名与main_path连接在一起
            company_file_name = os.path.join(self.txt_path,sorted_file_name_bm25 )
            with open(company_file_name,'r',encoding='utf-8') as file:
                docs_data = file.read()
            parent_docs = self.parent_split(docs_data) #父文档
            self.chunk_docs = []
            for doc in parent_docs:
                self.chunk_docs +=self.text_split(doc)
            # 接下来进行稀疏检索
            company_name,keywords = self.get_question_key(ner_res)
            bm25_selected_docs = [ docs_tuple[0] for docs_tuple  in self.bm25_retrieve_txt(keywords,self.chunk_docs)[:self.top_k]]
            # 利用重排模型对检索结果进行重排
            end_similarity = {}
            for chunk in bm25_selected_docs:
                end_similarity[chunk] = call_rerank([self.query,chunk])
            # 根据重排的子文档确定最终选择的子文档
            end_select_docs = []
            for key,_ in sorted(end_similarity.items(),key=lambda x:x[1],reverse=True)[:self.end_top_k]:
                for index,parent_doc in enumerate(parent_docs): # 遍历父文本块
                    if key in parent_doc and parent_doc not in end_select_docs: #根据匹配的子文本块找到父文本
                        end_select_docs.append(parent_doc)
                        
        return company_name,'\n'.join(end_select_docs),keywords
        
        
    def get_question_key(self,ner_res):
        ### 不使用API时进行命名实体识别    
        ner_res = ner_res.replace('：',':')
        ner_res = ner_res.replace('，',',')
        try:
            start_index = ner_res.index('公司名称:')
            end_index = ner_res.index('关键词:')
            # 提取公司名称和关键词信息
            company_name = ner_res[start_index+5:end_index-1]
            keywords = ner_res[end_index+4:]
        except:
            start_index = ner_res.index('公司名称:')
            company_name = ner_res[start_index+5:]
            keywords = self.query.replace(ner_res[:start_index+5],'')
        self.keywords = keywords
        
        return company_name,keywords
        
if __name__ =='__main__':
    demo = doc_retrive(query='截至2009年12月31日，兰州海默科技股份有限公司的正式在册员工人数是多少？')
    company,doc = demo.get_prompt_docs(ner_res='文本理解，公司名称：兰州海默科技股份有限公司，关键词：截至2009年12月31日，正式在册员工人数')
    print(company)
    print(doc)
    
