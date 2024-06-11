import jieba
import logging
import sys
import os
import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Prompts.prompt_generate import Prompts
from RAG.doc_retrieve import doc_retrive
from model_api.call_api import call_llm

class align_response():
    def __init__(self,response,query):
        self.response = response
        self.query = query
    
    def preprocess_fail_ans(self,ner_res):  ## 每次生成一个response去执行这个函数
        # 检查当前回答是否超出时长，即回答为空
        if not self.response:
            #如果回答为空，原因可归为文本太长，可适当减少输入的文本长度
            print('-------------当前回答为空，现在开始调整文本长度进行重新预测----------')
            call_llm_nums=1
            while call_llm_nums<=3:
                new_task =doc_retrive(rerank_top_k=5,use_embedding=False,is_first=False, query = self.query)
                company,selected_docs,ner_key= new_task.get_prompt_docs(ner_res=ner_res)
                prompt = Prompts(company,self.query,selected_docs,ner_key).text_generate_prompt()
                res = call_llm(prompt)
                if res:
                    break
                call_llm_nums+=1
        # 如果当前提供文本不能回答问题，需要重新修改文本块的长度以及top_k
        elif '对不起' in self.response or '抱歉' in self.response or '无法回答' in self.response or '请你提供' in self.response or '无法' in self.response:
            print('-----------------当前问题回答失败（回答字样包含《无法回答》等，开始重新回答----------------------------')
            call_llm_nums_new = 1
            while call_llm_nums_new<=1:
                new_task_ = doc_retrive(chunk_size=200,parent_chunk_size=900,rerank_top_k=6,use_embedding=False,is_first=False,query =self.query)
                company,selected_docs_,ner_key= new_task_.get_prompt_docs(ner_res=ner_res)
                prompt = Prompts(company,self.query,selected_docs_,ner_key).text_generate_prompt()
                res = call_llm(prompt)
                if '对不起' not in res and '抱歉' not in res and '无法回答' not in res and '请你提供' not  in res and '无法' not in res:
                    break
                call_llm_nums_new+=1
            if not res: #如果增加文本后超出限制或回应为None
                while res:
                    new_task =doc_retrive(chunk_size=200,parent_chunk_size=900,rerank_top_k=5,use_embedding=False,is_first=False,query =self.query)
                    company,selected_docs,ner_key= new_task.get_prompt_docs(ner_res=ner_res)
                    prompt = Prompts(company,self.query,selected_docs,ner_key).text_generate_prompt()
                    res = call_llm(prompt)
            if '对不起' in res or '抱歉' in res or '无法回答' in res or '请你提供' in res or '无法' in res:
                #如果还是无法回答，则转为SQL查询模块
                res = '请使用SQL查询！'
        else:
            res = self.response
        # 最后确定res  -------------如果还是含有无法回答字样基本确定是匹配错误----------------
        # if '对不起' in res or '抱歉' in res or '无法回答' in res or '请你提供' in res or '无法' in res:
        #     print('-----------------当前问题bm25算法文档检索不正确，开始使用embedding算法重新检索----------------------------')
        #     new_task_ = doc_retrive(top_k=5,chunk_size=300,chunk_overlap=100,query =self.query,use_embedding=True)
        #     company,selected_docs_,ner_key = new_task_.search_reletive_docs()
        #     prompt = Prompts(company,self.query,selected_docs_,ner_key).text_generate_prompt()
        #     res = call_llm(prompt)
        return res
        
if __name__=='__main__':
    res = '''\
    中国铁路通信信号股份有限公司的主要经营模式包括销售模式、生产模式、研发模式以及采购模式。

1. 销售模式：公司核心主业围绕轨道交通控制系统行业，因此整体销售模式按照主要客户中国铁路总公司、各客专公司及各城市轨道交通公司的公开招标模式进行投标。公司的销售模式主要为市场化公开投标，报告期内没有明显变化。

2. 生产模式：公司针对轨道交通控制系统行业的需求，进行产品的设计、生产和交付。公司建立了完善的设计、生产和交付流程，并严格执行ISO9001质量管理体系，确保产品质量。

3. 研发模式：公司注重技术创新和研发，建立了完善的研发体系和技术团队。公司紧跟行业发展需求，不断加大研发投入，推动技术和产品的创新和升级。

4. 采购模式：公司采用集中采购模式，由运营管理部作为采购活动的归口管理部门，负责制定采购管理相关制度，并监督和检查各级公司采购活动。公司对于主要原材料和设备采用长期合作协议和招标采购两种方式进行采购，以确保质量和稳定性。
    '''
    demo = align_response('请问您有什么问题需要帮助吗？','广东银禧科技股份有限公司设立时的发起人是谁？')
    a = demo.align_text_response()
    print(a)