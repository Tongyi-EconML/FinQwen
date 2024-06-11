from FlagEmbedding import BGEM3FlagModel #这里使用bge-m3模型
import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Config.set_config import Config
from fastapi import FastAPI,Request
import argparse
import uvicorn,json,datetime
app_encode = FastAPI()
import warnings
warnings.filterwarnings('ignore')
import torch
def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
@app_encode.post("/embedding")
async def create_item(request:Request):
    global embedding_model
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    content = json_post_list.get('prompt')
    response = embedding_model.encode(content,batch_size=12, max_length=8192)['dense_vecs']
    res = response.tolist()
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
            "response":res,
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
    parser.add_argument('--port',type=str,default = '5555')
    ######################### 在这里修改你的模型###############################
    model_dir = Config.embedding_model_path
    #########################################################################
    args = parser.parse_args()
    gpus = args.gpus
    port = args.port
    embedding_model= BGEM3FlagModel(Config.embedding_model_path, 
                      use_fp16=True)
    uvicorn.run(app_encode,host = '0.0.0.0',port = int(port),workers=1)