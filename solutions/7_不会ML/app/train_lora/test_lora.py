
from modelscope import AutoTokenizer,AutoModelForCausalLM
from peft import  get_peft_model,PeftModel
from get_sql_ans import Sqldb

def merge_model(model_name_path,lora_path):
    
    model = AutoModelForCausalLM.from_pretrained(model_name_path,device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_path, trust_remote_code=True,)
    model = PeftModel.from_pretrained(model,lora_path)
    print("Loaded PEFT model. Merging...")
    model.merge_and_unload()
    print("Merge complete.")
    return model,tokenizer

def test_sql_lora(model_name,lora_path):
    model,tokenizer = merge_model(model_name,lora_path)
    sqldb = Sqldb('博金杯比赛数据.db')
    while True:
        print("用户输入SQL生成诉求：", end="")
        prompt = input()
        response, history = model.chat(tokenizer, prompt, history=None,system='你是一个SQL生成器')
        print("当前LORA模型生成SQL语句为：",response)
        response = response.replace("B股票日行情表",'A股票日行情表')   
        response=  response.replace("创业板日行情表",'A股票日行情表')   
        if " 股票日行情表" in  response:
            response =  response.replace(" 股票日行情表",' A股票日行情表')  
        if " 港股日行情表" in response:
            response =  response.replace(" 港股日行情表",' 港股票日行情表')  
            response =  response.replace("”",'').replace("“",'') 
        sql_ans = sqldb.select_data(response)   
        print('当前SQL语句查询结果：',sql_ans)

def test_ner_lora(model_name,lora_path):
    model,tokenizer = merge_model(model_name,lora_path)
    while True:
        print("当前用户输入问题：", end="")
        prompt = input()
        response, history = model.chat(tokenizer, prompt, history=None,system='你是一个NER智能体') 
        print('当前用户问题识别结果：',response)

if __name__=='__main__':
    # test_sql_lora('qwen/Qwen-7B-Chat','./model_save/sql_lora/checkpoint-150')
    test_ner_lora('qwen/Qwen-7B-Chat','./model_save/ner_lora/checkpoint-500')
        


