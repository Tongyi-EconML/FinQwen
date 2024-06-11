import json
import requests
from requests.adapters import HTTPAdapter
import warnings
warnings.filterwarnings('ignore')
from requests.adapters import HTTPAdapter
from zhipuai import ZhipuAI
zhipu_api_key = "" #您的GLM4大模型API_key
client = ZhipuAI(api_key=zhipu_api_key)
import dashscope
from http import HTTPStatus
your_api_key = ''  #您的通义千问大模型API_key
dashscope.api_key= your_api_key
def call_qwen(message):
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                message[0]]
    response = dashscope.Generation.call(
            dashscope.Generation.Models.qwen_plus,
            messages=messages,
            result_format='message',  # 将返回结果格式设置为 message
        )
    if response.status_code == HTTPStatus.OK:
        return response.output.choices[0]['message']['content']
    else:
        return '当前调用API失败'
def call_glm(message):
    try:
        response = client.chat.completions.create(
        model="glm-4",  # 填写需要调用的模型名称
        # temperature=0.01,
        messages=message,)
        return response.choices[0].message.content
    except: #如果调用GLM失败则转为Qwen
        messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                message[0]]

        response = dashscope.Generation.call(
            dashscope.Generation.Models.qwen_turbo,
            messages=messages,
            result_format='message',  # 将返回结果格式设置为 message
        )
        if response.status_code == HTTPStatus.OK:
            return response.output.choices[0]['message']['content']
        else:
            return '当前调用API失败'


def call_embedding(prompt,url = "http://127.0.0.1:5555/embedding"):
    
    headers = {"Content-Type":"application/json"}
    data = json.dumps({"prompt":prompt})
    s = requests.Session()
    s.mount('http://',HTTPAdapter(max_retries = 3))
    try:
        res = s.post(url,data=data,headers = headers,timeout = 600)
        # print(res)
        if res.status_code ==200:
            return res.json()['response']
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(e)
        return None

def call_rerank(prompt,url = "http://127.0.0.1:9999/rerank"):
    headers = {"Content-Type":"application/json"}
    data = json.dumps({"prompt":prompt})
    s = requests.Session()
    s.mount('http://',HTTPAdapter(max_retries = 3))
    try:
        res = s.post(url,data=data,headers = headers,timeout = 600)
        # print(res)
        if res.status_code ==200:
            return res.json()['response']
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(e)
        return None
def call_llm(prompt,url = "http://127.0.0.1:6666/chat"):
    headers = {"Content-Type":"application/json"}
    data = json.dumps({"prompt":prompt})
    s = requests.Session()
    s.mount('http://',HTTPAdapter(max_retries = 3))
    try:
        res = s.post(url,data=data,headers = headers,timeout = 600)
        # print(res)
        if res.status_code ==200:
            return res.json()['response']
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(e)
        return None
def call_llm_ner_lora(prompt,url = "http://127.0.0.1:7777/NER"):
    headers = {"Content-Type":"application/json"}
    data = json.dumps({"prompt":prompt})
    s = requests.Session()
    s.mount('http://',HTTPAdapter(max_retries = 3))
    try:
        res = s.post(url,data=data,headers = headers,timeout = 600)
        # print(res)
        if res.status_code ==200:
            return res.json()['response']
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(e)
        return None

def call_llm_sql_lora(prompt,url = "http://127.0.0.1:8888/SQL"):
    headers = {"Content-Type":"application/json"}
    data = json.dumps({"prompt":prompt})
    s = requests.Session()
    s.mount('http://',HTTPAdapter(max_retries = 3))
    try:
        res = s.post(url,data=data,headers = headers,timeout = 600)
        # print(res)
        if res.status_code ==200:
            return res.json()['response']
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(e)
        return None
