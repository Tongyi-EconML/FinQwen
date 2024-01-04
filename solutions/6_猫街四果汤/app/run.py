import jsonlines
import os
from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope import GenerationConfig
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default="/tcdata", help='data and model directory path')
parser.add_argument('--model_name', type=str, default="Tongyi-Finance-14B-Chat", help='llm model choice')


def read_jsonl(path):
    content = []
    with jsonlines.open(path, "r") as json_file:
        for obj in json_file.iter(type=dict, skip_invalid=True):
            content.append(obj)
    return content


def write_jsonl(path, content):
    with jsonlines.open(path, "w") as json_file:
        json_file.write_all(content)


def model_test(model_dir, query="请解释一下资产负债率"):
    """
    Inference test for qwen series chat model.
    :param model_dir: support model `Tongyi-Finance-14B-Chat, Tongyi-Finance-14B-Chat-Int4, `
        `Qwen-7B-Chat, Qwen-7B-Chat-Int4, and Qwen-7B-Chat-Int8`
    :param query: user query or prompt
    :return:
    """
    # Note: The default behavior now has injection attack prevention off.
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    # use cpu only
    # model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cpu", trust_remote_code=True).eval()
    # use auto mode, automatically select precision based on the device.
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, device_map="cuda:0", trust_remote_code=True, local_files_only=True).eval()

    # Specify hyperparameters for generation
    model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)

    start_time = time.time()
    response, history = model.chat(tokenizer, query, history=None)
    end_time = time.time()
    print(f"Generate response:\n{response}", flush=True)
    generate_tokens = tokenizer(response)['input_ids']
    duration = end_time - start_time
    print(f"Inference time: {duration:.2f}s, "
          f"generate token speed {round(len(generate_tokens) / duration, 2)} token/s", flush=True)
    return response


def generate_answer(questions):
    # 选手需替换成各自的解决方案
    from sql_answer_new import sql_solution
    from text_answer import text_solution
    sql_solution('sql.jsonl')
    text_solution('text.jsonl')
    sql = read_jsonl('sql.jsonl')
    text = read_jsonl('text.jsonl')
    #to_answer = read_jsonl('bs_challenge_financial_14b_dataset/question.json')
    for q in questions:
        if 'answer' in sql[q['id']]:
            q['answer'] = sql[q['id']]['answer']
        elif 'answer' in text[q['id']]:
            q['answer'] = text[q['id']]['answer']
        else:
            q['answer'] = '无法回答问题`{}`'.format(q['question'])
    #write_jsonl('submission.jsonl', to_answer)

    #return [{**item, "answer": ""} for item in questions]
    return questions


if __name__ == "__main__":
    #args = parser.parse_args()
    #data_dir = args.data_dir
    #model_name = args.model_name

    # test LLM model
    #model_test(model_dir=os.path.join(data_dir, "models", model_name))

    # generate solution
    from setting import *
    load_questions = read_jsonl(os.path.join(data_dir,qfpath))
    final_answers = generate_answer(load_questions)
    write_jsonl("./submit_result.jsonl", final_answers)
