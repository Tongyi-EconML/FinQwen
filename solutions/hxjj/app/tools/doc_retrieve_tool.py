from .tool import Tool
from apis.retrieve_api import doc_retrieve

class DocumentRetrieveTool(Tool):
    description = '招股书文档检索'
    name = 'DocumentRetrieve'
    parameters: list = [{
        'name': 'company',
        'description': '公司',
        'required': False
    },
    {
        'name': 'retrieve_info',
        'description': '检索信息',
        'required': True             
    },]
        

    def _local_call(self, *args, **kwargs):
        if kwargs.get("retrieve_info") is None:
            return {"result": "Error: 参数错误，请检查!当前参数retrieve_info: None"}
        result = doc_retrieve.get_single_extra_knowledge(kwargs.get("company",""),kwargs["retrieve_info"])
        return {"result": result}
    
    
