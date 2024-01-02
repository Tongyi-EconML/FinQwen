import csv
import pandas as pd
import numpy as np
import re
import copy
n = 4
deny_list = ['0','1','2','3','4','5','6','7','8','9','，','？','。',
             '一','二','三','四','五','六','七','八','九','零','十',
            '的','小','请','.','?','有多少','帮我','我想','知道',
             '是多少','保留','是什么','-','(',')','（','）','：',
              '哪个','统计','且','和','来','请问','记得','有','它们']
pattern1 = r'\d{8}'
data_file_dir = '/app/intermediate/question_SQL_V6_exed.csv'
data_file = pd.read_csv(data_file_dir,delimiter = ",",header = 0)

data_file2_dir = '/app/intermediate/A01_question_classify.csv'
data_file2 = pd.read_csv(data_file2_dir,delimiter = ",",header = 0)
from modelscope import AutoModelForCausalLM, AutoTokenizer, snapshot_download
from modelscope import GenerationConfig

model_dir = '/tcdata/models/Tongyi-Finance-14B-Chat'

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda:0", trust_remote_code=True, bf16=True).eval()

model.generation_config = GenerationConfig.from_pretrained(model_dir,
                                                           trust_remote_code=True,
                                                           temperature = 0.00001,
                                                           top_p = 1,
                                                           do_sample = False,
                                                           seed = 1234)


print('B03_model_loaded')

deny_token_list = list()
for word in deny_list:
    temp_tokens = tokenizer(word)
    temp_tokens = temp_tokens['input_ids']
    deny_token_list = deny_token_list + temp_tokens

def get_prompt_v33(question,data,index_list):
    
    Examples = '以下是一些例子：'
    for index in index_list:
        Examples = Examples + "问题：" + example_question_list[index] + '\n'
        Examples = Examples + "资料：" + example_data_list[index] + '\n'
        Examples = Examples + "答案：" + example_FA_list[index] + '\n'
    impt2 = """
        你要进行句子生成工作，根据提供的资料来回答对应的问题。下面是一些例子。注意问题中对小数位数的要求。+ '\n'
    """
    
                
    impt2 = impt2 + Examples

    impt2 = impt2 +  "问题：" + question + '\n'
    impt2 = impt2 +  "资料：" + data + '\n'
    impt2 = impt2 +  "答案："
    return impt2




SQL_examples_file_dir = "/app/data/files/ICL_EXP.csv"
SQL_examples_file = pd.read_csv(SQL_examples_file_dir,delimiter = ",",header = 0)

example_employ_list = list()
for cyc in range(len(SQL_examples_file)):
    example_employ_list.append(0)

example_question_list = list()
example_data_list = list()
example_FA_list = list()
example_token_list = list()

for cyc in range(len(SQL_examples_file)):
    example_question_list.append(SQL_examples_file[cyc:cyc+1]['问题'][cyc])
    example_data_list.append(SQL_examples_file[cyc:cyc+1]['资料'][cyc])
    example_FA_list.append(SQL_examples_file[cyc:cyc+1]['FA'][cyc])
    temp_tokens = tokenizer(SQL_examples_file[cyc:cyc+1]['问题'][cyc])
    temp_tokens = temp_tokens['input_ids']
    temp_tokens2 = [x for x in temp_tokens if x not in deny_token_list]
    example_token_list.append(temp_tokens2)






g = open('/app/intermediate/FA_V5_SQL.csv', 'w', newline='', encoding = 'utf-8-sig') 
csvwriter = csv.writer(g)
csvwriter.writerow(['问题id','问题','FA','SQL结果'])





for cyc in range(1000):
    if cyc % 50 == 0:
        print(cyc)
    temp_question = data_file[cyc:cyc+1]['问题'][cyc]
    class_ans = data_file2[cyc:cyc+1]['分类'][cyc]
    SQL_search_result = data_file[cyc:cyc+1]['执行结果'][cyc]
    temp_FA = temp_question
    if class_ans != 'SQL':
        temp_FA = 'N_A'
    elif SQL_search_result != 'N_A':
        if len(SQL_search_result) > 0:
            if len(SQL_search_result) > 250:
                SQL_search_result = SQL_search_result[0:250]
            temp_question = data_file[cyc:cyc+1]['问题'][cyc]
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
                
                
                
            prompt2 = get_prompt_v33(data_file['问题'][cyc],data_file['执行结果'][cyc],max_index)
            temp_FA, history = model.chat(tokenizer, prompt2, history=None)

            
    else:
        SQL_search_result = 'SQL未能成功执行！'
        
    
    csvwriter.writerow([str(data_file[cyc:(cyc+1)]['问题id'][cyc]),
                    str(data_file[cyc:(cyc+1)]['问题'][cyc]),temp_FA,SQL_search_result])
g.close()
exit()
