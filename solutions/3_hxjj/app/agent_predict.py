from agent.bs_agent import BSAgentExecutor
from apis.model_api import llm
from config import logger
from utils.file_processor import read_jsonl
import json
import random
import torch
import os
import numpy as np
import argparse
def seed_it(seed):
    random.seed(seed) #可以注释掉
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) #这个懂吧
    torch.backends.cudnn.deterministic = True #确定性固定
    torch.backends.cudnn.benchmark = True #False会确定性地选择算法，会降低性能
    torch.backends.cudnn.enabled = True  #增加运行效率，默认就是True
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--question_file_path', type=str,default='data/question.json', nargs='?',help='question file which need predict')
    parser.add_argument('--answer_file_path', type=str,default='data/answer.jsonl', nargs='?',help='answer file which model generate')
    parser.add_argument('--random_seed', type=int, default=1234,nargs='?')
    parser.add_argument('--whole_output', type=bool, default=False, nargs='?')

    args = parser.parse_args()
    # set random seed
    seed_it(args.random_seed)
    print("加载Agent...")
    bs_agent = BSAgentExecutor(llm)
    contents = read_jsonl(args.question_file_path)
    print("开始预测...")
    for i,content in enumerate(contents):
        answer = bs_agent.run(content['question'])
        if args.whole_output:
            content['full_result'] = answer
        if len(answer) > 0 and isinstance(answer[-1],str):
            content['answer'] = answer[-1]
        else:
            content['answer'] = content['question']
        with open(args.answer_file_path,'a+',encoding='utf-8') as f:
            json.dump(content,f,ensure_ascii=False)
            f.write('\n')
        if i % 50 == 49:
            logger.info(f"已预测{i}条")
            print(f"已预测{i}条")
    print("预测完成！")

if __name__=='__main__':
    main()