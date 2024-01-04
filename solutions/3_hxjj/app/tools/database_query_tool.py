from .tool import Tool
from apis.dataset_api import sqldb

class DatabaseQueryTool(Tool):
    description = '数据库查询'
    name = 'DatabaseQuery'
    parameters: list = [{
        'name': 'sql_sentence',
        'description': 'sql语句',
        'required': True
    }]

    def _local_call(self, *args, **kwargs):
        if kwargs.get("sql_sentence") is None:
            return {"result": "Error: 参数错误，请检查!sql_sentence: None"}
        result = sqldb.select_data(kwargs['sql_sentence'])
        result = " ".join([" ".join([str(rr) for rr in res]) for res in result])
        return {"result": result}
