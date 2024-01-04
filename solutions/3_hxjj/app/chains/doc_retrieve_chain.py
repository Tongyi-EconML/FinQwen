from tools import Tool
from apis.retrieve_api import doc_retrieve
from apis.model_api import llm
from prompt.bs_prompt import BS_CHAT_KNOWLEDGE_TEMPLATE

class DocumentRetrieveChain(Tool):
    description = '招股书文档检索'
    name = 'DocumentRetrieve'
    parameters: list = [{
        'name': 'company',
        'description': '公司',
        'required': False
    }]
        

    def _local_call(self, *args, **kwargs):
        extra_knowledge = doc_retrieve.get_single_extra_knowledge(kwargs.get("company",""),args[0])
        user_input = BS_CHAT_KNOWLEDGE_TEMPLATE.replace('{user_question}', args[0]).replace('{extra_knowledge}',extra_knowledge)
        result = llm.original_generate(user_input)
        return {"result": result,"immediate_result": extra_knowledge}
    
    
