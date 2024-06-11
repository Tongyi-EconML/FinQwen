from FlagEmbedding import FlagReranker
import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Config.set_config import Config
from fastapi import FastAPI,Request
import argparse
import uvicorn,json,datetime
app_rerank = FastAPI()
import warnings
warnings.filterwarnings('ignore')
import torch

def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


@app_rerank.post("/rerank")
async def create_item(request:Request):
    global rerank_model
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    content = json_post_list.get('prompt')
    prompt = [{"role":"user","content":content}]
    response = rerank_model.compute_score(content)
    # res = response.tolist()
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
            "response":response,
            "status":200,
            "time":time
    }
    log ="["+time+"]"+'",prompt:"'+str(content)+'",response:"'+repr(response)+'"'
    print(log)
    torch_gc()
    return answer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='API argparser')
    parser.add_argument('--gpus',type = str,default='0,1,2,3') #14B-Chat固定两块显卡即可推理
    parser.add_argument('--port',type=str,default = '9999')
    ######################### 在这里修改你的模型###############################
    model_dir = Config.rerank_model_path
    model_dir = model_dir.replace('/','\\')
    print(model_dir)
    # model_dir = "/home/featurize/models/TongyiFinance/Tongyi-Finance-14B-Chat"#Featurize路径
    # model_dir = "/root/autodl-tmp/home/featurize/Qwen_7B-Chat"
    # model_dir ="/root/autodl-tmp/home/featurize/Qwen-14B-Chat/Qwen/Qwen-14B-Chat"
    #########################################################################
    args = parser.parse_args()
    gpus = args.gpus
    port = args.port
    rerank_model = FlagReranker(model_dir, use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
    uvicorn.run(app_rerank,host = '0.0.0.0',port = int(port),workers=1)