from tools import Tool
from apis.dataset_api import sqldb
from apis.model_api import llm
from prompt.bs_prompt import BS_SQL_GENERATOR_TEMPLATE, SCHEME_STRUCTURE_DICT, BS_SQL_GENERATOR_TEMPLATE_1
from utils.post_process_sql_result import post_process_answer
import re

class SqlGeneratorChain(Tool):
    description = 'sql语句生成'
    name = 'SqlGenerator'
    parameters: list = [{
        'name': 'table_list',
        'description': '所需表',
        'required': True
    }]

    def _local_call(self, *args, **kwargs):
        if kwargs.get("table_list") is None:
            return {"error": "参数错误，请检查!table_list: None"}
        user_question = args[0]
        table_structure_introduction = ""
        table_list = re.split(",|，|\ ",kwargs['table_list'])
        for table in table_list:
            table = table.strip(' ')
            if table == "":
                continue
            if SCHEME_STRUCTURE_DICT.get(table) is None:
                return {"error": f"Error: 无{table}的表结构信息，请检查!"}
            else:
                table_structure_introduction += f"表: {table} \n{SCHEME_STRUCTURE_DICT[table]}"
        user_input = BS_SQL_GENERATOR_TEMPLATE.replace("{table_structure_introduction}",table_structure_introduction).replace('{user_question}', user_question)
        result = llm.generate(user_input)
        # user_input = BS_SQL_GENERATOR_TEMPLATE_1.replace('{user_question}', user_question)
        # result = llm.sql_generate(user_input)
        return {"result": result}
