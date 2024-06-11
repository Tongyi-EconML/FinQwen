'''
API for Qwen-Finance-14B-Chat
用于问题回答的API
'''
import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Config.set_config import Config
from fastapi import FastAPI,Request
import argparse
import uvicorn,json,datetime
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope import GenerationConfig


app = FastAPI()

def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


@app.post("/chat")
async def create_item(request:Request):
    global model,tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    content = json_post_list.get('prompt')
    prompt = [{"role":"user","content":content}]
    response,history = model.chat(tokenizer,content,history = None)
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
            "response":response,
            "status":200,
            "time":time
    }
    log = "["+time+"]"+'",prompt:"'+content+'",response:"'+repr(response)+'"'
    print(log)
    torch_gc()
    return answer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='API argparser')
    parser.add_argument('--gpus',type = str,default='0,1,2,3') #14B-Chat固定两块显卡即可推理
    parser.add_argument('--port',type=str,default = '6666')
    ######################### 在这里修改你的模型###############################
    model_dir = Config.Qwen_Fin_14B_Chat
   
    #########################################################################
    args = parser.parse_args()
    gpus = args.gpus
    port = args.port
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map='auto', trust_remote_code=True, use_flash_attn=True,bf16=True).eval()
    # model = AutoModelForCausalLM.from_pretrained(model_dir, device_map='auto', trust_remote_code=True,use_cache_kernel=True,use_flash_attn=False,bf16=True).eval()
    model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True,temperature = 0.00001,seed=1234) ## 需要测试设置seed对输出的影响
    uvicorn.run(app,host = '0.0.0.0',port = int(port),workers=1)