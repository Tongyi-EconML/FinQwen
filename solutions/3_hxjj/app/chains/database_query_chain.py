from tools import Tool
from apis.dataset_api import sqldb
from apis.model_api import llm
from prompt.bs_prompt import BS_CHAT_SQLRESULT_TEMPLATE
from utils.post_process_sql_result import post_process_answer

class DatabaseQueryChain(Tool):
    description = '数据库查询'
    name = 'DatabaseQuery'
    parameters: list = [{
        'name': 'sql_sentence',
        'description': 'sql语句',
        'required': True
    }]

    def _local_call(self, *args, **kwargs):
        if kwargs.get("sql_sentence") is None:
            return {"error": "参数错误，请检查!sql_sentence: None"}
        sql_result = sqldb.select_data(kwargs['sql_sentence'])
        # sql_result = " ".join([" ".join([str(rr) for rr in res]) for res in sql_result])
        sql_result = post_process_answer(args[0], str(sql_result))
        user_input = BS_CHAT_SQLRESULT_TEMPLATE.replace('{user_question}', args[0]).replace('{sql_result}',sql_result)
        result = llm.original_generate(user_input)
        return {"result": result, "immediate_result": sql_result}
