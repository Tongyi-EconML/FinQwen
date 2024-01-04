from .prompt import PromptGenerator

BS_PLAN_DEFAULT_PROMPT = "你是一名高级智能助手，你可以先对问题进行分类，问题类型只有公司招股书咨询和股票基金数据查询两类，然后根据所给的信息列出回答该问题的任务列表。股票基金数据查询提供的表如下：A股公司行业划分表, A股票日行情表, 基金份额持有人结构, 基金债券持仓明细, 基金可转债持仓明细, 基金基本信息, 基金日行情表, 基金股票持仓明细, 基金规模变动表, 港股票日行情表。"
BS_TASK_DEFAULT_PROMPT = "你是一名高级智能助手，你需要根据当前提供的信息，执行当前任务。"
BS_CHAIN_PROMPT = "你是一名高级智能助手, 你需要针对用户问题，选择使用合适的插件。"
BS_TASK_INSTRUCTION_TEMPLATE = """当前任务可以使用的插件信息如下，请自行判断是否需要调用插件来解决当前用户问题。若需要调用插件，则需要将插件调用请求按照json格式给出，必须包含api_name、parameters字段，并在其前后使用<|startofthink|>和<|endofthink|>作为标志。\
若无需调用插件，直接执行任务，结果无需标志。
{tool_list}"""

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

BS_USER_QUESTION_TEMPLATE = "用户问题：{user_question}"
BS_CURRENT_TASK_TEMPLATE = "当前任务：{current_task}"

BS_CHAT_KNOWLEDGE_TEMPLATE="""------检索内容开始------
{extra_knowledge}
------检索内容结束------

用户问题：{user_question}。
完全根据检索内容结合问题回答用户问题，将问题和答案结合后输出。注意不要输出“根据检索”。
"""

# BS_CHAT_KNOWLEDGE_TEMPLATE="""------检索内容开始------
# {extra_knowledge}
# ------检索内容结束------

# 用户问题：{user_question}。
# 完全根据检索内容结合问题回答用户问题，将问题和答案结合后输出；
# 若在检索内容中无答案，输出“问题” + “并未在招股意向书中详细说明”，如用户问题：“上海华铭智能终端设备股份有限公司的首发战略配售结果如何？”，输出：“上海华铭智能终端设备股份有限公司的首发战略配售具体情况并未在招股意向书中详细说明。”。"""

BS_SQL_GENERATOR_TEMPLATE="""你是一名高级数据库工程师，请你根据所提供的表结构说明以及用户问题，生成sql语句，数据库为sqlite，你生成的sql语句格式必须符合sqlite格式。
------表结构说明开始------
{table_structure_introduction}
------表结构说明结束------

用户问题：{user_question}。
注意：答案只需要sql语句，不需要其他任何输出。
"""

BS_SQL_GENERATOR_TEMPLATE_1="你是一名sqlite数据库开发人员，精通sql语言，你需要根据已知的10张表的表名、字段名和用户输入的问题编写sql\n\n" \
             "{'表名': '基金基本信息', '字段名': ['基金代码', '基金全称', '基金简称', '管理人', '托管人', '基金类型', '成立日期', '到期日期', '管理费率', '托管费率']}\n" \
             "{'表名': '基金股票持仓明细', '字段名': ['基金代码', '基金简称', '持仓日期', '股票代码', '股票名称', '数量', '市值', '市值占基金资产净值比', '第N大重仓股', '所在证券市场', '[所属国家(地区)]', '报告类型']}\n" \
             "{'表名': '基金债券持仓明细', '字段名': ['基金代码', '基金简称', '持仓日期', '债券类型', '债券名称', '持债数量', '持债市值', '持债市值占基金资产净值比', '第N大重仓股', '所在证券市场', '[所属国家(地区)]', '报告类型']}\n" \
             "{'表名': '基金可转债持仓明细', '字段名': ['基金代码', '基金简称', '持仓日期', '对应股票代码', '债券名称', '数量', '市值', '市值占基金资产净值比', '第N大重仓股', '所在证券市场', '[所属国家(地区)]', '报告类型']}\n" \
             "{'表名': '基金日行情表', '字段名': ['基金代码', '交易日期', '单位净值', '复权单位净值', '累计单位净值', '资产净值']}\n" \
             "{'表名': 'A股票日行情表', '字段名': ['股票代码', '交易日', '[昨收盘(元)]', '[今开盘(元)]', '[最高价(元)]', '[最低价(元)]', '[收盘价(元)]', '[成交量(股)]', '[成交金额(元)]']}\n" \
             "{'表名': '港股票日行情表', '字段名': ['股票代码', '交易日', '[昨收盘(元)]', '[今开盘(元)]', '[最高价(元)]', '[最低价(元)]', '[收盘价(元)]', '[成交量(股)]', '[成交金额(元)]']}\n" \
             "{'表名': 'A股公司行业划分表', '字段名': ['股票代码', '交易日期', '行业划分标准', '一级行业名称', '二级行业名称']}\n" \
             "{'表名': '基金规模变动表', '字段名': ['基金代码', '基金简称', '公告日期', '截止日期', '报告期期初基金总份额', '报告期基金总申购份额', '报告期基金总赎回份额', '报告期期末基金总份额', '定期报告所属年度', '报告类型']}\n" \
             "{'表名': '基金份额持有人结构', '字段名': ['基金代码', '基金简称', '公告日期', '截止日期', '机构投资者持有的基金份额', '机构投资者持有的基金份额占总份额比例', '个人投资者持有的基金份额', '个人投资者持有的基金份额占总份额比例', '定期报告所属年度', '报告类型']}\n\n" \
             "请根据以下用户输入编写sql。\n用户输入: {user_question}"

BS_CHAT_SQLRESULT_TEMPLATE="""问题：“{user_question}”。
答案：“{sql_result}”。

将问题的内容和答案的内容融合的文字内容输出。注意不要输出“问题：”或“答案：”。
"""

class BSPromptGenerator(PromptGenerator):
    def __init__(self,
                 plan_template=BS_PLAN_DEFAULT_PROMPT,
                 task_template=BS_TASK_DEFAULT_PROMPT,
                 task_instruction_template=BS_TASK_INSTRUCTION_TEMPLATE,
                 user_template=BS_USER_QUESTION_TEMPLATE,
                 current_task_template=BS_CURRENT_TASK_TEMPLATE,
                 sep='\n\n',
                 prompt_max_length=10000):
        super().__init__(plan_template, task_template, task_instruction_template, user_template, current_task_template, sep,
                         prompt_max_length)


    def generate(self, task_no=None):
        # init plan
        if task_no is None:
            prompt_list = [self.system_prompt,
                           self.user_prompt]
        # execute tasks
        else:
            # no task result
            if not self.task_result_prompt:
                prompt_list = [self.system_prompt,
                               self.user_prompt,
                               self.current_task_prompt]
            else:
                prompt_list = [self.system_prompt,
                               self.task_result_prompt,
                               self.user_prompt,
                               self.current_task_prompt]
        return self.sep.join(prompt_list)
    
    def update_task_prompt(self, current_task):
        self.current_task_prompt = self.current_task_template.replace("{current_task}", current_task)

class BSChainPromptGenerator(PromptGenerator):
    def __init__(self, 
                 chain_template=BS_CHAIN_PROMPT,
                 task_instruction_template=BS_TASK_INSTRUCTION_TEMPLATE,
                 user_template=BS_USER_QUESTION_TEMPLATE,
                 sep='\n\n'):
        self.chain_template = chain_template
        self.task_instruction_template = task_instruction_template
        self.user_template = user_template
        self.sep = sep
    
    def init_prompt(self, tool_list):
        self.system_prompt = self.chain_template + self.task_instruction_template.replace("{tool_list}",self.get_tool_str(tool_list))
        
        
    
    def generate(self, user_question):
        self.user_prompt = self.user_template.replace("{user_question}",user_question)
        return self.sep.join([self.system_prompt, self.user_prompt])

