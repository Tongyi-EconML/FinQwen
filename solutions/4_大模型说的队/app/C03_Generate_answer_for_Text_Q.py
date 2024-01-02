import json
import csv
import pandas as pd
import copy 
n1 = 20
n2 = 20
num_of_answer = 10

csv_file_dir = '/app/intermediate/AB01_question_with_related_text_ot_normalized.csv'
q_file =  pd.read_csv(csv_file_dir,delimiter = ",",header = 0)

csv_file_dir_2 = '/app/intermediate/AB01_question_with_related_text_rp.csv'
q_file_2 =  pd.read_csv(csv_file_dir_2,delimiter = ",",header = 0)

unknown_words = ["不知道","无法直接","无法确定","没有找到",'未知','与问题无关','无法直接得出','没有具体说明',
                 '没有看到','没有被提及','没有发现','不清楚','没有明确','无法得出','尚未公布','●●●','不在提供的资料',
                 '无法回答','可以推算','无法找到','没有提供','无法直接','未知的','未在资料中','没有在资料中',
                 '未明确列出','没有被明确','没有被提供','无法计算','我找不到','无法浏览网页',']]','[[','并未给出',
                 '无法判断','没有直接给出','没有提到','我不清楚','无法准确回答','没有直接说明','未被提及','无法访问','无法给出','无法得出','无法预测','无法提供']

from modelscope import AutoModelForCausalLM, AutoTokenizer, snapshot_download
from modelscope import GenerationConfig

model_dir = '/tcdata/models/Tongyi-Finance-14B-Chat'

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

# use bf16
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda:0", trust_remote_code=True, bf16=True).eval()

model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True, temperature = 0.01, top_p = 1, seed = 1234)

def answer_generator(piece_list,n,m):
    temp_index = 0
    return_response_list = list()

    while temp_index < n:
        next_piece = ''
        prompt = "你是一个人工智能文字秘书。下面是一段资料，不要计算，不要计算，直接从资料中寻找问题的答案，使用完整的句子回答问题。\n 如果资料不包含问题的答案，回答“不知道。”如果从资料无法得出问题的答案，回答“不知道。”如果答案未在资料中说明，回答“不知道。”如果资料与问题无关或者在资料中找不到问题的答案，回答“不知道。”如果资料没有明确说明问题答案，回答“不知道。”资料："
        if 'text' in piece_list[temp_index]:
            if piece_list[temp_index]['text'] == piece_list[temp_index]['text']:
                next_piece = next_piece + piece_list[temp_index]['text']

        if 'table' in piece_list[temp_index]:
            if piece_list[temp_index]['table'] == piece_list[temp_index]['table']:
                next_piece = next_piece + piece_list[temp_index]['table']
        temp_index = temp_index + 1

        prompt = prompt + '资料' + '：' + next_piece
        prompt = prompt + ' \n 问题：' + temp_q 

        response, history = model.chat(tokenizer, prompt, history=None)

        response = response.replace('\n','')
        #response = response.replace(' ','')

	
        add_flag = 1
        if len(response) <500:
            for word in unknown_words:
                if word in response:
                    add_flag = 0

        if add_flag == 1 and response not in return_response_list:
            return_response_list.append(response)

        if len(return_response_list) >= m:
            break
    return return_response_list


g = open('/app/intermediate/FA_V5_Text_cap4_4_nt.csv', 'w', newline='', encoding = 'utf-8-sig') 
csvwriter = csv.writer(g)
csvwriter.writerow(['问题id','问题','实体答案','final_ans1','ans_list'])

print('C03_Started')

for cyc in range(0,1000):
    entity_name = q_file[cyc:(cyc+1)]["对应实体"][cyc]
    answer = 'N_A'
    answer2 = 'N_A'
    temp_q = q_file[cyc:(cyc+1)]['问题'][cyc]
    response_list = list()
    if entity_name != 'N_A':
        
        temp_pdf_pieces = eval(q_file[cyc:cyc+1]['top_n_pages'][cyc])
        
        
        temp_index = 0
        
        response_list = answer_generator(temp_pdf_pieces,n1,num_of_answer)

        if len(response_list) <= 7:
            response_list_2 = list()
            temp_pdf_pieces_2 = eval(q_file_2[cyc:cyc+1]['top_n_pages'][cyc])
            response_list_2 = answer_generator(temp_pdf_pieces_2,n2,num_of_answer-len(response_list))
            for resp in response_list_2:
                response_list.append(resp)       

        
        
        answer = ""
        for resp in response_list:
            answer = answer + resp + '\n'

            
    csvwriter.writerow([q_file[cyc:(cyc+1)]['问题id'][cyc],
                        temp_q,
                        entity_name,
                        answer,str(response_list)])
    
g.close()
exit()
