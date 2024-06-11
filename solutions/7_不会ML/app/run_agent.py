import torch
import random
import re
import os
import numpy as np 
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model_api.call_api import call_qwen,call_glm,call_llm_ner_lora,call_llm,call_llm_sql_lora
import pandas as pd
from Prompts.prompt_generate import Prompts
from RAG.doc_retrieve import doc_retrive
from Config.set_config import Config
import warnings
warnings.filterwarnings('ignore')
from SQL_base.get_sql_ans import Sqldb
from RAG.bm25 import BM25
from utils.check_llm_response import align_response
import json
from tqdm import tqdm
import jsonlines
import argparse

ner_error=0
sql_error=0
def read_jsonl(path):
    content = []
    with jsonlines.open(path, "r") as json_file:
        for obj in json_file.iter(type=dict, skip_invalid=True):
            content.append(obj)
    return content
class agent:
    def __init__(self,use_api=True,query='',api_name='glm',top_k:int=15,parent_chunk_size:int=800,chunk_size:int=150,chunk_overlap:int=50,use_embedding=True,rerank_top_k:int=6,is_first=True)-> None:
        self.use_api = use_api
        self.query = query
        self.api_name = api_name
        self.top_k = top_k
        self.parent_chunk_size = parent_chunk_size
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_embedding = use_embedding
        self.rerank_top_k = rerank_top_k
        self.is_first = is_first
        if self.api_name =='qwen':
            self.llm = call_qwen
        elif self.api_name =='glm':
            self.llm = call_glm
    
    def classify_task(self):
        if self.use_api:
            # 调用大模型进行意图识别，根据用户判断识别结果
            prompt = Prompts(company='',query=self.query,top_texts='',ner_key='',text_answer='').classify_task()
            content = [{"role": "user", "content": prompt}]
            response = self.llm(content)
            return response
        else:
            # 不使用私有API即为本地化部署
            return call_llm_ner_lora(prompt=self.query) #直接返回本地化LORA微调后的NER识别结果
        
            
    def text_task(self,ner_res):
        retrieve_tool  = doc_retrive(top_k=self.top_k,parent_chunk_size=self.parent_chunk_size, query = self.query,use_embedding=self.use_embedding,rerank_top_k=self.rerank_top_k,use_api=self.use_api,is_first=self.is_first) # 初始化文本检索器
        company_name,selected_docs,keywords = retrieve_tool.get_prompt_docs(ner_res)
        if self.use_api: #如果使用私有化部署API
            prompt = Prompts(company=company_name,query= self.query,top_texts= selected_docs).api_text_generate_prompt() #获取该问题相关的prompt示例
            content =  [{"role": "user", "content": prompt}] 
            cur_response = self.llm(content) # 获得当前问题的初步结果
        else: #如果是本地化部署大模型
            
            prompt =Prompts(company=company_name,query= self.query,top_texts= selected_docs,ner_key=keywords).text_generate_prompt() #获取该问题相关的prompt示例
            cur_response = call_llm(prompt)
            # 这里加上回溯检查
            
        return  cur_response
    
    def get_sql_prompt_bm25(self):
        exp_answer_path = Config.sql_answer_template_path #读取SQL问题的模版
        exp_answer_df = pd.read_csv(exp_answer_path)
        exp_querys = list(exp_answer_df['问题'].values)
        bm_25 = BM25(exp_querys) #利用prompt工程
        bm25_scores = bm_25.get_scores(self.query)
        sorted_docs = sorted(zip(exp_querys, bm25_scores), key=lambda x: x[1], reverse=True)[:2]
        select_query = [docs_tuple[0] for docs_tuple in sorted_docs]
        select_sql = [list(exp_answer_df[exp_answer_df['问题']==select_query[0]]['SQL'].values)[0],list(exp_answer_df[exp_answer_df['问题']==select_query[1]]['SQL'].values)[0]]
        if self.use_api:
            sql_prompt = Prompts(query = self.query).api_sql_generate_prompt(select_query,select_sql)
        else:
            sql_prompt = Prompts(query=self.query).sql_generate_prompt(select_query,select_sql)
            
        return sql_prompt
    
    def sql_task(self):
        sqldb = Sqldb(Config.db_sqlite_url)
        if self.use_api:#如果使用私有化部署API
            sql_prompt = self.get_sql_prompt_bm25()
            content =  [{"role": "user", "content": sql_prompt}] 
            response = self.llm(content) # 获得生成的SQL语句执行结果
            print('初步生成sql结果：',response)
            num_repet = 1
            while "```sql" not in response and num_repet <=5:
                response = self.llm(content)
                num_repet+=1
            response = response.replace("B股票日行情表",'A股票日行情表')   
            response=  response.replace("创业板日行情表",'A股票日行情表')   
            if " 股票日行情表" in  response:
                 response =  response.replace(" 股票日行情表",' A股票日行情表')  
            if " 港股日行情表" in response:
                 response =  response.replace(" 港股日行情表",' 港股票日行情表')  
            response =  response.replace("”",'').replace("“",'') 
            try:
                pattern = re.compile(r'```sql\n(.*?)```', re.DOTALL)
                matches = pattern.findall(response)
                # print(matches)
                
                if matches:
                    sql_statement = matches[0].strip()
                #     print(sql_statement)
                else:
                    sql_statement = None
                #     print("No SQL statement found.")  
                sql_ans = sqldb.select_data(sql_statement)   
                print('当前SQL语句查询结果：',sql_ans)
            except:
                print('当前答案出错')
                sql_ans = self.query
            
            sql_ans_prompt =  Prompts().get_sql_answer(query=self.query,sql=sql_ans)
            ans = self.llm(message=[{"role": "user", "content": sql_ans_prompt}])
       
        
        else: #如果使用本地化部署大模型如 Qwen-Finance-14B-Chat
            # 首先使用Prompt工程得到初步结果
            sql_prompt = self.get_sql_prompt_bm25()
            response = call_llm(sql_prompt) #利用Qwen-14B-Chat响应结果
            response = response.replace("B股票日行情表",'A股票日行情表')   
            response=  response.replace("创业板日行情表",'A股票日行情表')   
            if " 股票日行情表" in  response:
                 response =  response.replace(" 股票日行情表",' A股票日行情表')  
            if " 港股日行情表" in response:
                 response =  response.replace(" 港股日行情表",' 港股票日行情表')  
            response =  response.replace("”",'').replace("“",'') 
            
            try: #首先尝试根据Prompt工程生成的SQL去数据库查询结果
                sql_ans = sqldb.select_data(response)
                if sql_ans ==[] or 'None' in str(sql_ans):
                    # 如果没有报错但是生成的答案是空值，则启用Lora进行生成
                    
                    re_response = call_llm_sql_lora(self.query)
                    try:
                        sql_ans = Sqldb.select_data(re_response)
                        
                    except:
                        sql_ans = self.query
            except:
                #如果查询失败，则使用prompt工程进行生成
                sql_prompt = self.get_sql_prompt_bm25()
                re_response = call_llm(sql_prompt) #利用Qwen-14B-Chat响应结果
                try:
                    sql_ans = sqldb.select_data(re_response)
                except:
                    #如果这时还是出现错误，则返回错误，不再进行纠正
                    sql_ans = self.query
            # 此时已经初步得到SQL的查询结果了
            if self.query == sql_ans:
                print('当前问题回答出错，最终答案设置为原问题描述')
                ans = sql_ans
            else:
                print('当前SQL语句查询结果：',sql_ans)
                end_prompt = Prompts().get_sql_answer(query=self.query,sql=sql_ans)
                ans = call_llm(end_prompt)
        return ans
                
    def run(self):
        global ner_error
        global sql_error
        if self.use_api:
            task = self.classify_task()
            pattern = re.compile(r'《(.*?)》', re.DOTALL)
            matches = pattern.findall(task)
            if matches:
                ner_res = matches[0].strip()
            print('当前任务类型：',ner_res)
            if '文本理解' in ner_res:
                # 执行文本理解任务
                ans = self.text_task(ner_res=ner_res)
            else:
                ans = self.sql_task()
                
                    
        else:
            #如果不使用API，则需要进行NER识别
            ner_res = call_llm_ner_lora(prompt=self.query)
            print('当前NER识别结果：',ner_res)
            if '文本理解' in ner_res:
                ans = self.text_task(ner_res=ner_res)
                check_ans = align_response(response=ans,query=self.query).preprocess_fail_ans(ner_res=ner_res)
                if check_ans =='请使用SQL查询！':
                    print('当前文本理解失败，转为SQL查询')
                    ner_error+=1
                    ans = self.sql_task()
                else:
                    ans = check_ans
            else:
                ans = self.sql_task()
                if ans == self.query:
                    sql_error+=1
        
        return ans
            
def main():
    parser = argparse.ArgumentParser(description="Run Agent")
    parser.add_argument("--use_api", action = 'store_true', default=False)
    parser.add_argument("--api_name", type=str, default='qwen')
    parser.add_argument("--top_k", type=int, default=15)
    parser.add_argument("--parent_chunk_size", type=int, default=800)
    parser.add_argument("--chunk_size", type=int, default=150)
    parser.add_argument("--chunk_overlap", type=int, default=50)
    parser.add_argument("--use_embedding", action='store_true', default=False)
    parser.add_argument("--rerank_top_k", type=int, default=6)
    parser.add_argument("--is_first", action='store_true', default=False)
    args = parser.parse_args()
    print(args)
    querys = read_jsonl(path=Config.question_json_path)
    with open(Config.res_json_path,'w',encoding='utf-8') as file:
        for item in tqdm(querys):
            
            query = item['question']
            print('当前问题为：',query)
            cur_task = agent(use_api=args.use_api,query=query,api_name=args.api_name,top_k=args.top_k,parent_chunk_size=args.parent_chunk_size,chunk_size = args.chunk_size,chunk_overlap=args.chunk_overlap,use_embedding=args.use_embedding,rerank_top_k=args.rerank_top_k,is_first=args.is_first)
            cur_res = cur_task.run()
            item['answer'] = cur_res
            print('当前问题回答为：',cur_res)
                
            json_record = json.dumps(item, ensure_ascii=False)
            file.write(json_record)
            file.write('\n')
        print('当前agent系统意图识别错误个数为：',ner_error)
        print('当前agent系统SQL生成错误个数为：',sql_error)
           
        
        
        
if __name__=='__main__':
    main()
 
        
        
    
    
        

