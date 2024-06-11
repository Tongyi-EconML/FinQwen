
import os
import logging
WORK_DIR = ''
logger = logging.getLogger()
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler(WORK_DIR + "log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def get_base_file_path():
    return os.path.dirname(os.path.dirname(__file__))


class Config:
    base_path = get_base_file_path()
    pdf_path = os.path.join(base_path,"data/competition_data/pdf")
    txt_path = os.path.join(base_path, "data/competition_data/pdf2txt")
    company_name_res = os.path.join(base_path,"data/pdf2txt.csv")
    Qwen_Fin_14B_Chat = os.path.join(base_path,'models/llm_model/Qwen-Fin-14B-Chat')
    Qwen_7B_Chat = os.path.join(base_path,'models/llm_model/Qwen-7B-Chat')
    NER_lora_path = os.path.join(base_path,'models/lora_adpter/NER_lora')
    SQL_lora_path = os.path.join(base_path,'models/lora_adpter/sql_lora')
    question_ner_path = os.path.join(base_path,"out/NER_lora_res.csv")
    embedding_model_path = os.path.join(base_path,"models/embedding_model") #基础embedding模型
    rerank_model_path = os.path.join(base_path,"models/rerank_model") #重排embedding模型
    res_json_path=os.path.join(base_path,'out/answer_submit_qwen.jsonl')
    question_json_path = os.path.join(base_path,'data/competition_data/question.json')
    db_sqlite_url = os.path.join(base_path, "data/competition_data/博金杯比赛数据.db").replace("\\", "/")
    sql_cluster_path = os.path.join(base_path,"out/谱聚类结果")
    sql_answer_template_path = os.path.join(base_path,'data/ICL_EXP.csv')
    sql_answer_template_path1 = os.path.join(base_path,'data/SQL-template.xlsx')
    embedding_vector_store = os.path.join(base_path,'data/embedding_vector')
    embedding_company_name_store = os.path.join(base_path,'data/embedding_vector/embedding_company.pkl')
    embedding_index_path = os.path.join(base_path,'data/embedding_index')


    