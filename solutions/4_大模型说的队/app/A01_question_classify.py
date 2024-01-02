import csv
import pandas as pd
import numpy as np
import re
from modelscope import AutoModelForCausalLM, AutoTokenizer, snapshot_download
from modelscope import GenerationConfig

model_dir = '/tcdata/models/Tongyi-Finance-14B-Chat'

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

new_question_file_dir = '/app/intermediate/question_csv.csv'
new_question_file = pd.read_csv(new_question_file_dir,delimiter = ",",header = 0)
company_file_dir = '/app/data/files/AF0_pdf_to_company.csv'
company_file = pd.read_csv(company_file_dir,delimiter = ",",header = 0)
company_list = list()
for cyc in range(len(company_file)):
    company_list.append(company_file[cyc:cyc+1]['公司名称'][cyc])
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda:0", trust_remote_code=True, bf16=True).eval()
model.generation_config = GenerationConfig.from_pretrained(model_dir,
                                                           trust_remote_code=True,
                                                           temperature = 0.0000001,
                                                           top_p = 1,
                                                           do_sample = False,
                                                           seed = 1234)

print('A01_model_loaded')

g = open('/app/intermediate/A01_question_classify.csv', 'w', newline='', encoding = 'utf-8-sig') 
csvwriter = csv.writer(g)
csvwriter.writerow(['问题id','问题','答案','分类'])
prompt = """
    你是一个问题分类器。对于每个提供给你的问题，你需要猜测答案是在该公司的招股说明书中还是在基金股票数据库里。以下是一些例子：

    问题：“在2019年的中期报告里，XX基金管理有限公司管理的基金中，有多少比例的基金是个人投资者持有的份额超过机构投资者？希望得到一个精确到两位小数的百分比。”
    回答：“基金股票数据库”
    
    问题：“XXXX股份有限公司变更设立时作为发起人的法人有哪些？”
    回答：“该公司的招股说明书”
    
    问题：“我想知道XXXXXX债券A基金在20200930的季报中，其可转债持仓占比最大的是哪个行业？用申万一级行业来统计。”
    回答：“基金股票数据库”
    
    问题：“XXXXXX股份有限公司2020年增资后的投后估值是多少？”
    回答：“该公司的招股说明书”
    
    问题：“根据XXXXXX股份有限公司招股意向书，全球率先整体用LED路灯替换传统路灯的案例是？”
    回答：“该公司的招股说明书”
    
    问题：“什么公司、在何时与XXXXXX股份有限公司发生了产品争议事项？产品争议事项是否已经解决？”
    回答：“该公司的招股说明书”
    
    问题：“请帮我查询下股票代码为XXXXXX的股票在2021年内最高日收盘价是多少？”
    回答：“基金股票数据库”
    
    问题：“XXXXXX股份有限公司的中标里程覆盖率为多少？”
    回答：“该公司的招股说明书”
    
    问题：“根据中国证监会颁布的《上市公司行业分类指导》的规定，XXXXXX有限公司所属的行业大类、中类、小类是什么？”
    回答：“该公司的招股说明书”
    
    问题：“请问XXXX年一季度有多少家基金是净申购?它们的净申购份额加起来是多少?请四舍五入保留小数点两位。”
    回答：“基金股票数据库”
    
    问题：“XXXXXX有限公司和合肥翰林是否按规定为员工缴纳了社会保险？”
    回答：“该公司的招股说明书”
    
    问题：“我想知道XXXXXX有限公司在2020年成立了多少只管理费率小于0.8%的基金？”
    回答：“基金股票数据库”
    
    问题：“根据《CRCC产品认证实施规则》，《铁路产品认证证书》有效期为多久？XXXXXX有限公司取得 《铁路产品认证证书》后，至少多久需要接受一次监督？”
    回答：“该公司的招股说明书”
    
    问题：“我想知道XXXXXX基金管理有限公司在2019年成立了多少只管理费率小于0.8%的基金？”
    回答：“基金股票数据库”
    
    问题：“请问XXXX年一季度有多少家基金是净申购?它们的净申购份额加起来是多少?请四舍五入保留小数点两位。”
    回答：“基金股票数据库”
    
    问题：“我想知道XXXXXX有限公司在2019年成立了多少只管理费率小于0.8%的基金？”
    回答：“基金股票数据库”
    
    问题：“我想知道股票XXXXXX在申万行业分类下的二级行业是什么？用最新的数据。”
    回答：“基金股票数据库”
    
    问题：“请帮我查询下股票代码为XXXXXX的股票在2019年内最高日收盘价是多少？”
    回答：“基金股票数据库”
    
    问题：“股票XXXXXX在20200227日期中的收盘价是多少?（小数点保留3位）”
    回答：“基金股票数据库”
    
    问题：“截至2009年底，中海达、南方测绘合计占有国产品牌销售额的多大比例？”
    回答：“该公司的招股说明书”
    
    问题：“截止2005年12月31日，南岭化工厂的总资产和净资产分别是多少？”
    回答：“该公司的招股说明书”
    
    问题：“股票XXXXXX在20200227日期中的收盘价是多少?（小数点保留3位）”
    回答：“基金股票数据库”

    根据上面提供的例子对以下问题进行分类。
    问题：“
    """
for cyc in range(len(new_question_file)):
    temp_question = new_question_file[cyc:cyc+1]['问题'][cyc]

    prompt1 = prompt + temp_question + """？"""

    response_new, history_new = model.chat(tokenizer, prompt1, history=None)
    if cyc % 100 == 0:
        print(str(new_question_file[cyc:(cyc+1)]['问题id'][cyc]))

    if '招股说明书' in response_new and '股票数据库' not in response_new:
        temp_class = 'Text'
    elif '招股说明书' not in response_new and '股票数据库' in response_new:
        temp_class = 'SQL'
        for company_name in company_list:
            if company_name in temp_question:
                temp_class = 'Text'
    else:
        temp_class = 'SQL'
        for company_name in company_list:
            if company_name in temp_question:
                temp_class = 'Text'
    if cyc in [166,174]:
        temp_calss = 'Text'


    csvwriter.writerow([str(new_question_file[cyc:(cyc+1)]['问题id'][cyc]),
                    str(new_question_file[cyc:(cyc+1)]['问题'][cyc]),
                    response_new,temp_class])
g.close()



exit()
