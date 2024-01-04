from .tool import Tool
from apis.model_api import llm
from prompt.bs_prompt import BS_CHAT_KNOWLEDGE_TEMPLATE, BS_CHAT_SQLRESULT_TEMPLATE

class LlmChatTool(Tool):
    description = '大模型聊天'
    name = 'LlmChat'
    parameters: list = [{
        'name': 'question',
        'description': '问题',
        'required': True
    },{
        'name': 'question_type',
        'description': '问题类型',
        'required': True
    },{
        'name': 'task_result',
        'description': '任务结果',
        'required': True
    }]

    def _local_call(self, *args, **kwargs):
        if kwargs.get("question") is None or kwargs.get("question_type") is None or kwargs.get("task_result") is None:
            return {"result": f'Error: 参数错误，请检查! question: {kwargs.get("question")}, question_type: {kwargs.get("question_type")}, query_data: {kwargs.get("task_result")},'}
        if kwargs["question_type"] == '文档检索':
            user_input = BS_CHAT_KNOWLEDGE_TEMPLATE.replace('{user_question}', kwargs['question']).replace('{extra_knowledge}',kwargs['task_result'])
        elif kwargs["question_type"] == '数据库查询':
            user_input = BS_CHAT_SQLRESULT_TEMPLATE.replace('{user_question}', kwargs['question']).replace('{sql_result}',kwargs['task_result'])
        else:
            return {'result': f'Error: 问题类型错误，请检查。question_type: {kwargs.get("question_type")}'}
        
        result = llm.original_generate(user_input)
        return {"result": result}