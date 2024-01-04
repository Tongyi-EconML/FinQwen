from langchain import SQLDatabase
import torch
import sql_metadata
import threading
import os
import json
import re
import sqlite3
import jsonlines
import pandas as pd
from modelscope import AutoModelForCausalLM, AutoTokenizer, snapshot_download
from modelscope import GenerationConfig
import difflib
from setting import *

print("sqlite:///" + os.path.join(work_dir,"app/bojin.db"))
db = SQLDatabase.from_uri("sqlite:///" + os.path.join(work_dir,"app/bojin.db"))
conn = sqlite3.connect(os.path.join(work_dir,"app/bojin.db"))
cursor = conn.cursor()

def read_jsonl(path):
    content = []
    with jsonlines.open(path, "r") as json_file:
        for obj in json_file.iter(type=dict, skip_invalid=True):
            content.append(obj)
    return content


def write_jsonl(path, content):
    with jsonlines.open(path, "w") as json_file:
        json_file.write_all(content)


def error_handle_prompt(e, last_response, question, sample, num):
    if num == 1:
        return '''现在请你充当数据库问题的专家角色，将问题翻译为标准的SQL语句。请只使用以下表格的表格名和列名生成SQL语句，SQL表格信息是：{}。
        请直接生成能在以下数据库中执行成功的SQL代码，不要有其他解释，可以参考以下例子：
        {}。
        问题：`{}`，SQL语句:'''.format(db.table_info, sample, question)

    prompt_error = '''你是一个擅长将人类提问的问题转成SQL语句的AI，根据下面的错误提示和人类提问的问题修改一个SQL语句，错误是：`{}`，人类提问的问题是:`{}`。
    需要修改的SQL语句是:`{}`。
    请直接生成能在以下数据库中执行成功的SQL代码, 请修改：'''.format(e, question, last_response)

    try:
        sql = sql_metadata.Parser(last_response)
        table_info = db.get_table_info(sql.tables)
    except:
        return prompt_error

    if 'syntax error' in e:
        prompt_error = '''你的回答有语法错误，错误是：`{}`。
    注意检查SQL语法,请你修改后再回答。
    请直接生成能在以下数据库中执行成功的SQL代码, 现在问题是:`{}`, 请编写SQL语句：'''.format(e, question)

    if 'no such column' in e:
        prompt_error = '''回答有错误，错误是：`{}`。
                注意检查SQL语法,只能用下面的表名和列名生成SQL语句："{}。
                请直接生成能在以下数据库中执行成功的SQL代码, 现在问题是:`{}`，
                请编写SQL语句：'''.format(e, table_info, question)
    return prompt_error


def get_answer(question: dict, model, tokenizer, sample_template):
    def query_db(query, done_event):
        try:
            conn = sqlite3.connect(os.path.join(work_dir,"app/bojin.db"))
            cursor = conn.cursor()
            cursor.execute(query)

            columns = [col[0] for col in cursor.description]

            # 获取查询结果
            results = cursor.fetchall()

            # 处理结果
            formatted_results = []
            for row in results:
                row_data = {}
                for idx, value in enumerate(row):
                    row_data[columns[idx]] = value
                formatted_results.append(row_data)

            result_container["results"] = formatted_results
            # print('succ',result_container)
        except sqlite3.OperationalError as e:

            result_container["error"] = f"SQLite error: {e}"
        finally:
            done_event.set()
            conn.close()

    def run_thread(sql):

        done_event = threading.Event()

        # 启动查询线程
        thread = threading.Thread(target=query_db, args=(sql, done_event))
        thread.start()

        thread.join(120)
        if thread.is_alive():

            raise ValueError('SQL execute timeout')
        else:
            if "error" in result_container:
                results = result_container.get("error", [])
                raise ValueError(results)
            else:
                results = result_container.get("results", [])
            return results

    s = difflib.get_close_matches(question['question'], sample_template.keys(), n=2, cutoff=0.10)
    sample_info = '\n'.join([f'{i}. 问题:`{v}`, SQL语句:`{sample_template[v]}`。' for i, v in enumerate(s)])
    if sample_info == '':
        prompt = '''现在请你充当数据库问题的专家角色，将问题翻译为标准的SQL语句。请只使用以下表格的表格名和列名生成SQL语句，SQL表格信息是：{}。
    请直接生成能在以下数据库中执行成功的SQL代码，不要有其他解释，问题：`{}`，SQL语句:'''
        response, history = model.chat(tokenizer, prompt.format(db.table_info, question['question']), history=None,
                                       temperature=0.0, do_sample=False)
        question['choose'] = 'table'
    else:
        prompt = '''现在请你充当数据库问题的专家角色，从给定的2个问题转SQL例子中选择最相似的一个，然后模仿SQl语句修改。请直接生成能在以下数据库中执行成功的SQL代码，不要有其他解释。
        给定的问题转SQL例子：
        {}。
        问题：`{}`，请给出SQL语句:'''
        response, history = model.chat(tokenizer, prompt.format(sample_info, question['question']), history=None,
                                       temperature=0.0, do_sample=False)
        question['choose'] = 'example'
    question['model_return'] = response
    response = response.replace('半年度报告', '年报')

    sql_error = None
    i = 0
    while True:
        if i > 3:
            question['sql_error'] = sql_error
            break
        try:
            if sql_error:
                prompt_error = error_handle_prompt(sql_error, response, question['question'], sample_info, i)
                if i == 1:
                    response, history = model.chat(tokenizer, prompt_error, history=None)
                else:
                    response, history = model.chat(tokenizer, prompt_error, history=history)
                question['model_return'] = response

            sql = re.findall(r'.*?(SELECT .*?)(?:`|$|。|;)', response, re.DOTALL)
            result_container = {}
            result = run_thread(sql[0])
            question['sql_return'] = result
            prompt = '''请你把问题和答案组成一个完整的回答，回答要简洁但完整，
                        例如："景顺长城中短债债券C基金在20210331的季报里，前三大持仓占比的债券名称是什么?"，需要回答："景顺长城中短债债券C在20210331的季报中，前三大持仓占比的债券名称分别是21国开01、20农发清发01、20国信03。"。
                        现在问题是<{}>,答案是<{}>,请回复:'''
            response, history = model.chat(tokenizer, prompt.format(question['question'],
                                                                    json.dumps(result, ensure_ascii=False)),
                                           history=None, temperature=0.0, do_sample=False)
            question['qwen_sql_answer'] = response
            print(question['question'], question['qwen_sql_answer'])
            question['answer'] = question['qwen_sql_answer']
            break
        except Exception as e:
            sql_error = str(e)
            print(sql_error)
            i += 1
    return question


def sql_solution(fpath):

    model_dir = os.path.join(models_dir,'Tongyi-Finance-14B-Chat-Int4')
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model_ori = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda:0", trust_remote_code=True).eval()
    model_ori.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)

    from swift import SwiftModel, Swift
    model = SwiftModel.from_pretrained(model_ori,
                                       'checkpoint-240',
                                       device_map="cuda:0", inference_mode=True)


    train2_data = {}
    train2 = pd.read_excel('train1211_single.xlsx')
    for v in train2.fillna('0').to_dict('records'):
        if str(v['sql']) == '0':
            continue
        train2_data[v['question']] = v['sql']
    print('template len:',len(train2_data))


    ques = read_jsonl(os.path.join(data_dir,qfpath))
    answer = []
    i = 0
    for n in ques:
        if not re.search(r'(?:股票|基金|A股|港股)', n['question']):
            continue
        i += 1
        print(n['id'])
        q = get_answer(n, model, tokenizer, train2_data)
        answer.append(q)

    to_answer = read_jsonl(os.path.join(data_dir,qfpath))
    i = 0
    for q in answer:
        if 'answer' in q:
            to_answer[q['id']]['answer'] = q['answer']
            i += 1
        to_answer[q['id']]['model_return'] = q['model_return']
    write_jsonl(fpath, to_answer)
