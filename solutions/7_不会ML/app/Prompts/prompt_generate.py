from langchain.prompts import ChatPromptTemplate

SCHEME_STRUCTURE_DICT = {
'A股公司行业划分表': 
'''
字段 类型
股票代码 TEXT 
交易日期 TEXT
行业划分标准 TEXT
一级行业名称 TEXT
二级行业名称 TEXT
''',
'A股票日行情表': 
'''
字段 类型
股票代码 TEXT
交易日 TEXT
[昨收盘(元)] REAL
[今开盘(元)] REAL
[最高价(元)] REAL
[最低价(元)] REAL
[收盘价(元)] REAL
[成交量(股)] REAL
[成交金额(元)] REAL
''',
'基金份额持有人结构':
'''
字段 类型
基金代码 TEXT
基金简称 TEXT
公告日期 TIMESTAMP
截止日期 TIMESTAMP
机构投资者持有的基金份额 REAL
机构投资者持有的基金份额占总份额比例 REAL
个人投资者持有的基金份额 REAL
个人投资者持有的基金份额占总份额比例 REAL
定期报告所属年度 INTEGER
报告类型 TEXT
''',
'基金债券持仓明细':
'''
字段 类型
基金代码 TEXT
基金简称 TEXT
持仓日期 TEXT
债券类型 TEXT
债券名称 TEXT
持债数量 REAL
持债市值 REAL
持债市值占基金资产净值比 REAL
第N大重仓股 INTEGER
所在证券市场 TEXT
[所属国家(地区)] TEXT
报告类型TEXT TEXT
''',
'基金可转债持仓明细':
'''
字段 类型
基金代码 TEXT
基金简称 TEXT
持仓日期 TEXT
对应股票代码 TEXT
债券名称 TEXT
数量 REAL
市值 REAL
市值占基金资产净值比 REAL
第N大重仓股 INTEGER
所在证券市场 TEXT
[所属国家(地区)] TEXT
报告类型 TEXT
''',
'基金基本信息':
'''
字段 类型
基金代码 TEXT
基金全称 TEXT
基金简称 TEXT
管理人 TEXT
托管人 TEXT
基金类型 TEXT
成立日期 TEXT
到期日期 TEXT
管理费率 TEXT
托管费率 TEXT
''',
'基金日行情表':
'''
字段 类型
基金代码 TEXT
交易日期 TEXT
单位净值 REAL
复权单位净值 REAL
累计单位净值 REAL
资产净值 REAL
''',
'基金股票持仓明细':
'''
字段 类型
基金代码 TEXT
基金简称 TEXT
持仓日期 TEXT
股票代码 TEXT
股票名称 TEXT
数量 REAL
市值 REAL
市值占基金资产净值比 REAL
第N大重仓股 INTEGER
所在证券市场 TEXT
[所属国家(地区)] TEXT
报告类型 TEXT
''',
'基金规模变动表':
'''
字段 类型
基金代码 TEXT
基金简称 TEXT
公告日期 TIMESTAMP
截止日期 TIMESTAMP
报告期期初基金总份额 REAL
报告期基金总申购份额 REAL
报告期基金总赎回份额 REAL
报告期期末基金总份额 REAL
定期报告所属年度 INTEGER
报告类型 TEXT
''',
'港股票日行情表':
'''
字段 类型
股票代码 TEXT
交易日 TEXT
[昨收盘(元)] REAL
[今开盘(元)] REAL
[最高价(元)] REAL
[最低价(元)] REAL
[收盘价(元)] REAL
[成交量(股)] REAL
[成交金额(元)] REAL
'''
}
class Prompts():
    def __init__(self,company:str='',query:str='',top_texts:str='',ner_key:str='',text_answer:str=''):
        self.query = query
        self.top_texts = top_texts
        self.company = company
        self.ner_key = ner_key
        self.text_answer = text_answer
    
    def classify_task(self): #定义意图识别函数
        prompt = ChatPromptTemplate.from_template(
            "你是一个问题分类器。对于每个提供给你的问题，你需要猜测出该问题是属于文本理解任务还是在SQL查询任务,如果是文本理解任务，则需要返回判断类型结果以及问题中涉及的公司名称以及该问题的语义关键词；如果是SQL查询任务，则只需要返回判断类型结果即可，无需返回公司名称以及关键词。以下是一些例子：\n"
            "问题：“在2019年的中期报告里，XX基金管理有限公司管理的基金中，有多少比例的基金是个人投资者持有的份额超过机构投资者？希望得到一个精确到两位小数的百分比。回答：《SQL查询任务》\n"
            "问题：“XXXX股份有限公司变更设立时作为发起人的法人有哪些？回答：《文本理解任务，公司名称：XXXX股份有限公司，关键词：变更设立时作为发起人的法人有哪些》\n”"
            "问题：“我想知道XXXXXX债券A基金在20200930的季报中，其可转债持仓占比最大的是哪个行业？用申万一级行业来统计。”回答：《SQL查询任务》\n"
            "问题：“XXXXXX股份有限公司2020年增资后的投后估值是多少？”回答：《文本理解任务，公司名称：XXXXXX股份有限公司，关键词：2020年增资后的投后估值是多少》\n"
            "问题：根据XXXXXX股份有限公司招股意向书，全球率先整体用LED路灯替换传统路灯的案例是？”回答：《文本理解任务，公司名称：XXXXXX股份有限公司，关键词：全球率先整体用LED路灯替换传统路灯的案例》\n"
            "问题：什么公司、在何时与XXXXXX股份有限公司发生了产品争议事项？产品争议事项是否已经解决？回答：《文本理解任务，公司名称：XXXXXX股份有限公司，关键词：什么公司、在何时与之发生了产品争议事项，是否已经解决》\n"
            "问题：请问XXXX年一季度有多少家基金是净申购?它们的净申购份额加起来是多少?请四舍五入保留小数点两位。回答：《SQL查询任务》\n"
            "问题：我想知道股票XXXXXX在申万行业分类下的二级行业是什么？用最新的数据。回答：《SQL查询任务》\n"
            "问题：截止2005年12月31日，南岭化工厂的总资产和净资产分别是多少？回答：《文本理解任务，公司名称：南岭化工厂，关键词：截止2005年12月31日的总资产和净资产》\n"
            "问题：XXXXXX股份有限公司的中标里程覆盖率为多少？回答：《文本理解任务，公司名称：XXXXXX股份有限公司，关键词：中标里程覆盖率》\n"
            "问题：各报告期末，XXXXX股份有限公司存货分别为多少万元？占XX资产的比例分别多少?回答：《文本理解任务，公司名称：XXXXX股份有限公司，关键词：各报告期末的存货以及占XX资产的比例》\n"
            "请你根据上面提供的例子，对当前用户问题类型进行分类，输出标准请严格按照上面提示要求，\n"
            "当判断问题为文本理解类型时，你的严格输出格式为:《任务类型结果，公司名称：，关键词：》；当判断问题为SQL查询任务类型时，你的严格输出格式为:《任务类型结果》用户问题：{question}\n"
            "请直接输出结果，不可以输出其他的文字。一般情况下，如果问题中涉及到基金以及股票时都是SQL查询任务。"
            ).format_messages(question=self.query)
        return prompt[0].content
    
    def api_text_generate_prompt(self): ###-----分数为74.07
        prompt = ChatPromptTemplate.from_template(
            "你是一个能够精准提取文本信息并能根据文本信息准确回答问题的智能小助手。\n"
            "下面我会给你几段招股书文本和一个问题，请你根据提供的文本对所给的问题进行回答。\n"
            "已知相关的部分招股书资料是：\n"
            "{text}\n\n"
            "请你牢记以下五条回答准则，并且根据上文所给的招股书资料以及所给的问题关键词回答问题：{question}\n"
            "注意在回答的时候请遵守以下准则：\n"
            "(1).你只能根据我提供给你的招股书资料回答问题，不可以利用文本资料以外的知识进行回答,不可以回答招股书资料以外的答案！\n"
            "(2).你可以智能的处理表格中的信息，你的回答中不可以直接输出表格。\n"
            "(3).问题的答案基本可以在所给的招股书资料中可以明确找到，但是如果你明确在已有的资料里面查找不到答案，请你严格按照如下格式输出：对不起，我不能根据所给的资料回答问题，请你提供更多的资料！\n"
            "(4).你的回答应围绕所给的关键词进行回答，并且要尽可能地具体且正确。\n"
            "(5).要确保答案的完整性,请不要直接输出答案，输出的格式请一定要结合原问题，以下是几个输出示例：\n"
            "问题：A公司的盈利是多少？错误回答格式：2000万元；正确回答格式：A公司的盈利是2000万元；\n问题：B公司的发行股票的数量是多少？错误回答格式：回答格式：1000股；正确回答格式：B公司的发行股票的数量是1000股;\n问题：C公司的主要产品是什么？错误回答格式：xx疫苗与xx药品；正确回答格式：C公司的主要产品是xx疫苗与xx药品。\n\n"
            ).format_messages(text=self.top_texts, question=self.query) 
        return prompt[0].content
    def text_generate_prompt(self): ###-----分数为74.07
        prompt = ChatPromptTemplate.from_template(
            "你是一个能够精准提取文本信息并能根据文本信息准确回答问题的智能小助手。\n"
            "下面我会给你几段招股书文本和一个问题以及问题中的重要关键词，请你根据提供的文本对所给的问题进行回答。\n"
            "关于{company}的招股书资料是：\n\n"
            "{text}\n\n"
            "请你牢记以下五条回答准则，并且根据上文所给的招股书资料以及所给的问题关键词回答问题：{question}\n"
            "注意在回答的时候请遵守以下准则：\n"
            "(1).你只能根据我提供给你的招股书资料回答问题，不可以利用文本资料以外的知识进行回答,不可以回答招股书资料以外的答案！\n"
            "(2).你可以智能的处理表格中的信息，你的回答中不可以直接输出表格。\n"
            "(3).问题的答案基本可以在所给的招股书资料中可以明确找到，但是如果你明确在已有的资料里面查找不到答案，请你严格按照如下格式输出：对不起，我不能根据所给的资料回答问题，请你提供更多的资料！\n"
            "(4).你的回答应围绕所给的关键词进行回答，并且要尽可能地具体且正确。\n"
            "(5).要确保答案的完整性,请不要直接输出答案，输出的格式请一定要结合原问题，以下是几个输出示例：\n"
            "问题：A公司的盈利是多少？错误回答格式：2000万元；正确回答格式：A公司的盈利是2000万元；\n问题：B公司的发行股票的数量是多少？错误回答格式：回答格式：1000股；正确回答格式：B公司的发行股票的数量是1000股;\n问题：C公司的主要产品是什么？错误回答格式：xx疫苗与xx药品；正确回答格式：C公司的主要产品是xx疫苗与xx药品。\n\n"
            "本次问题的关键词是：{key}\n"
            ).format_messages(company=self.company,text=self.top_texts, question=self.query,key = self.ner_key) 
        return prompt[0].content   
    def sql_generate_prompt(self,query_tmp,answer_tmp):
        prompt =  ChatPromptTemplate.from_template(
            "你是一名高级SQL工程师，请你根据我提供的表结构说明以及用户问题，生成sql语句，数据库为sqlite，你生成的sql语句格式必须符合sqlite格式。\n"
            "你可以选择的表结构为：\n"
            "{table_structure_introduction}\n\n"
            "请你在生成SQL的时候要符合上述所给的表结构，不能出现遗漏或制作不存在的表名和字段名。\n"
            "生成SQL时请一定要保证格式正确，不能出现错误： cur.execute(p_sql_str) sqlite3.OperationalError: near “)”: syntax error\n"
            "生成Sql时要根据上文提供的表结构去生成相应的SQL代码。\n"
            "以下是两个示例：\n"
            "问题：{query1};SQL语句：{sql1}\n"
            "问题：{query2};SQL语句：{sql2}\n"
            "请你根据上述提供的表结构并参考提供的两个示例取生成用户问题的答案。\n"
            "请你仔细检查生成的sql格式，不允许有格式错误，如：sqlite3.OperationalError: near “(”: syntax error\n"
            "用户问题：{user_question}\n"
            "请只输出sql语句即可，不可以输出其他文字内容").format_messages(table_structure_introduction= SCHEME_STRUCTURE_DICT,user_question= self.query,query1= query_tmp[0],sql1=answer_tmp[0],query2=query_tmp[1],sql2=answer_tmp[1])
        return prompt[0].content
    def api_sql_generate_prompt(self,query_tmp,answer_tmp):
        prompt =  ChatPromptTemplate.from_template(
            "你是一名高级SQL工程师，请你根据我提供的表结构说明以及用户问题，生成sql语句，数据库为sqlite，你生成的sql语句格式必须符合sqlite格式。\n"
            "请你生成SQL时请一定要保证格式正确，不能出现错误： cur.execute(p_sql_str) sqlite3.OperationalError: near “)”: syntax error\n"
            "以下是两个示例：\n"
            "问题：{query1};SQL语句：\n{sql1}\n"
            "问题：{query2};SQL语句：\n{sql2}\n"
            "请你参考上面两个示例去生成问题的答案。\n"
            "所给的示例以及问题的答案格式是一致的，一般情况下你只需要根据问题修改示例中的相应的时间以及表结构名称即可，不可以随意更改原来示例的代码结构。\n"
            "请你仔细检查生成的sql格式，不允许有格式错误，如：sqlite3.OperationalError: near “(”: syntax error\n"
            "用户问题：{user_question}\n"
            "注意：你的输出答案只需要sql语句，不需要其他任何输出。\n"
            "你的输出格式为：```sql\n 具体生成的sql语句```\n\n"
            "请严格按照以上格式输出。"
            ).format_messages(table_structure_introduction= SCHEME_STRUCTURE_DICT,user_question= self.query,query1= query_tmp[0],sql1=answer_tmp[0],query2=query_tmp[1],sql2=answer_tmp[1])
        return prompt[0].content

    def get_sql_answer(self,query,sql):
        prompt = ChatPromptTemplate.from_template(
            "你是一个可以整合问题和答案为完整的一句话的AI，你可以进行小数点保留与取整操作。\n"
            "我会给你一个问题以及该问题的回答（回答是sql的查询结果）。请你按照下面的要求进行整合输出：\n"
            "首先你需要根据问题去正确提取sql输出中的答案，然后总结到一起成为一句通顺的语句输出。你可以参考下面的示例：\n"
            "示例：【问题：请查询：在2099的年度报告中，个人投资者持有基金份额大于机构投资者持有基金份额的基金属于货币型类型的有几个；sql答案：[(10,)]；输出格式：在2099的年度报告中，个人投资者持有基金份额大于机构投资者持有基金份额的基金属于货币型类型的有10个。】\n"
            "你要保证输出结果中的小数点与问题要求的小数点保持一致,如果问题中要求取整，那么你的答案也要保留整数，不可以有小数,以下是几个例子：\n"
            "类似的：如果在问题含有百分数保留两位小数字样，若sql为：[0.0%],你应该在回答中含有0.00%；如果在问题含有百分数保留三位小数字样，若sql为：[3.51%],你应该在回答中含有3.510%\n"
            "现在请你将问题：{q}与sql答案：{sql}整合输出。\n"
            "只输出最后整合的答案，不可以输出其他不相关的文本，也不可以直接输出sql答案。要额外注意小数点问题，输出的结果要符合问题要求的小数点位数,不可以随意进行单位转换，请保持原查询结果的数据形式，只需注意问题的小数点要求即可。").format_messages(q=query,sql=sql)
        return prompt[0].content
        
    def align_sql_prompt(self,query_tmp,answer_tmp,answer_cur,except_str):
        prompt =  ChatPromptTemplate.from_template(
            "你是一名高级数据库工程师，请你根据所提供的表结构说明以及用户问题，生成sql语句，数据库为sqlite，你生成的sql语句格式必须符合sqlite格式。你可以检查并修正你的sql代码。\n"
            "------表结构说明开始------\n"
            "{table_structure_introduction}\n"
            "------表结构说明结束------\n"
            "以下是两个示例：\n"
            "问题：{query1};SQL语句：{sql1}\n"
            "问题：{query2};SQL语句：{sql2}\n"
            "请你根据上述提供的表结构并参考提供的两个示例取生成用户问题的答案。\n"
            "用户问题：{user_question}\n"
            "你当前的输出sql语句为:{ans_cur}\n"
            "你当前代码的错误信息为：{except_str}\n"
            "你需要根据代码的错误信息以及上文提供的表结构对你当前的代码进行修正。\n"
            "注意：答案只需要sql语句，不需要其他任何输出。").format_messages(table_structure_introduction= SCHEME_STRUCTURE_DICT,user_question= self.query,query1= query_tmp[0],sql1=answer_tmp[0],query2=query_tmp[1],sql2=answer_tmp[1],ans_cur=answer_cur,except_str=except_str)
        return prompt[0].content
    

if __name__ =="__main__":
    a = Prompts('1+1=？','2222222222222')
    print(a.text_generate_prompt())