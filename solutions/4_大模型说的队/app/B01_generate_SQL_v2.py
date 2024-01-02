table_name_list = ['基金基本信息','基金股票持仓明细','基金债券持仓明细','基金可转债持仓明细','基金日行情表','A股票日行情表','港股票日行情表','A股公司行业划分表','基金规模变动表','基金份额持有人结构']
table_info_dict = {}
n = 5
deny_list = ['0','1','2','3','4','5','6','7','8','9','，','？','。',
             '一','二','三','四','五','六','七','八','九','零','十',
            '的','小','请','.','?','有多少','帮我','我想','知道',
             '是多少','保留','是什么','-','(',')','（','）','：',
              '哪个','统计','且','和','来','请问','记得','有','它们']


import csv
import pandas as pd 
import numpy as np
import sqlite3
import re
import copy 
from langchain.utilities import SQLDatabase
from modelscope import AutoModelForCausalLM, AutoTokenizer, snapshot_download
from modelscope import GenerationConfig

db0 = SQLDatabase.from_uri("sqlite:////tcdata/bs_challenge_financial_14b_dataset/dataset/博金杯比赛数据.db", sample_rows_in_table_info=0)
dbd0 = db0.table_info

db2 = SQLDatabase.from_uri("sqlite:////tcdata/bs_challenge_financial_14b_dataset/dataset/博金杯比赛数据.db", sample_rows_in_table_info=2)
dbd2 = db2.table_info
list1 = dbd2.split('CREATE TABLE')
for cyc_piece in range(len(list1)):
    list1[cyc_piece] = 'CREATE TABLE' + list1[cyc_piece]
for piece in list1:
    for word in table_name_list:
        if word in piece:
            table_info_dict[word] = piece
question_csv_file_dir = "/app/intermediate/A01_question_classify.csv"
question_csv_file = pd.read_csv(question_csv_file_dir,delimiter = ",",header = 0)
model_dir = '/tcdata/models/Tongyi-Finance-14B-Chat'
# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda:0", trust_remote_code=True, bf16=True).eval()    
model.generation_config = GenerationConfig.from_pretrained(model_dir,
                                                           trust_remote_code=True,
                                                           temperature = 0.0001,
                                                           top_p = 1,
                                                           do_sample = False,
                                                           seed = 1234)

print('B01_model_loaded')

deny_token_list = list()
for word in deny_list:
    temp_tokens = tokenizer(word)
    temp_tokens = temp_tokens['input_ids']
    deny_token_list = deny_token_list + temp_tokens

def get_prompt_v33(question,index_list):
    
    Examples = '以下是一些例子：'
    for index in index_list:
        Examples = Examples + "问题：" + example_question_list[index] + '\n'
        Examples = Examples + "SQL：" + example_sql_list[index] + '\n'
    
    impt2 = """
        你是一个精通SQL语句的程序员。
        我会给你一个问题，请按照问题描述，仿照以下例子写出正确的SQL代码。
    """
    
                
    impt2 = impt2 + Examples

    impt2 = impt2 +  "问题：" + question + '\n'
    impt2 = impt2 +  "SQL：" 
    return impt2


SQL_examples_file_dir = "/app/data/files/ICL_EXP.csv"
SQL_examples_file = pd.read_csv(SQL_examples_file_dir,delimiter = ",",header = 0)

example_employ_list = list()
for cyc in range(len(SQL_examples_file)):
    example_employ_list.append(0)

example_question_list = list()
example_table_list = list()
example_sql_list = list()
example_token_list = list()

for cyc in range(len(SQL_examples_file)):
    example_question_list.append(SQL_examples_file[cyc:cyc+1]['问题'][cyc])
    example_sql_list.append(SQL_examples_file[cyc:cyc+1]['SQL'][cyc])
    temp_tokens = tokenizer(SQL_examples_file[cyc:cyc+1]['问题'][cyc])
    temp_tokens = temp_tokens['input_ids']
    temp_tokens2 = [x for x in temp_tokens if x not in deny_token_list]
    example_token_list.append(temp_tokens2)

g = open('/app/intermediate/question_SQL_V6.csv', 'w', newline='', encoding = 'utf-8-sig') 
csvwriter = csv.writer(g)
csvwriter.writerow(['问题id','问题','SQL语句','prompt'])

pattern1 = r'\d{8}'

for cyc in range(len(question_csv_file)):
    if cyc % 50 == 0:
        print(cyc)
    response2 = 'N_A'
    prompt2 = 'N_A'

    if question_csv_file['分类'][cyc] == 'SQL' and cyc not in [174]:
        temp_question = question_csv_file[cyc:cyc+1]['问题'][cyc]
        date_list =  re.findall(pattern1,temp_question)
        temp_question2_for_search = temp_question
        for t_date in date_list:
            temp_question2_for_search.replace(t_date,' ')
        temp_tokens = tokenizer(temp_question2_for_search)
        temp_tokens = temp_tokens['input_ids']
        temp_tokens2 = [x for x in temp_tokens if x not in deny_token_list]
        temp_tokens = temp_tokens2
        #计算与已有问题的相似度
        similarity_list = list()
        for cyc2 in range(len(SQL_examples_file)):
            similarity_list.append(len(set(temp_tokens) &set(example_token_list[cyc2]))/ (len(set(temp_tokens))+len(set(example_token_list[cyc2])) ))

        #求与第X个问题相似的问题

        t = copy.deepcopy(similarity_list)
        # 求m个最大的数值及其索引
        max_number = []
        max_index = []
        for _ in range(n):
            number = max(t)
            index = t.index(number)
            t[index] = 0
            max_number.append(number)
            max_index.append(index)
        t = []
        
        temp_length_test = ""
        short_index_list = list()
        for index in max_index:
            temp_length_test_1 = temp_length_test
            temp_length_test = temp_length_test + example_question_list[index]
            temp_length_test = temp_length_test + example_sql_list[index]
            if len(temp_length_test) > 2300:
                break
            short_index_list.append(index)
        
        prompt2 = get_prompt_v33(question_csv_file['问题'][cyc],short_index_list)
        response2, history = model.chat(tokenizer, prompt2, history=None)
    else:
        pass
    csvwriter.writerow([str(question_csv_file[cyc:(cyc+1)]['问题id'][cyc]),
                str(question_csv_file[cyc:(cyc+1)]['问题'][cyc]),
                response2,prompt2])







