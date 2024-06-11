from peft import PeftModel  # dynamic import to avoid dependency on peft
from tqdm import tqdm 
import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Config.set_config import Config
import torch 
from peft import  PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,)
from modelscope import GenerationConfig
from fastapi import FastAPI,Request
import argparse
import uvicorn,json,datetime


def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
app = FastAPI()
@app.post("/NER")
async def create_item(request:Request):
    global model,tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    content = json_post_list.get('prompt')
    prompt = [{"role":"user","content":content}]
    response,history = model.chat(tokenizer,content,history = None,system = '你是一个SQL生成器')
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
    parser.add_argument('--port',type=str,default = '7777')
    args = parser.parse_args()
    model_base = Config.Qwen_7B_Chat
    tokenizer = AutoTokenizer.from_pretrained(model_base,trust_remote_code=True,)
    model = AutoModelForCausalLM.from_pretrained(model_base,device_map="auto", trust_remote_code=True)
    model.generation_config = GenerationConfig.from_pretrained(model_base, trust_remote_code=True,temperature = 0.01,top_k=3)
    model = PeftModel.from_pretrained(model,Config.NER_lora_path)
    print("Loaded PEFT model. Merging...")
    model.merge_and_unload()
    print("Merge complete.")
    model = model.eval()
    port = args.port
    uvicorn.run(app,host = '0.0.0.0',port = int(port),workers=1)
    





    