# coding=utf-8

import json
import re
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import Levenshtein
import jsonlines
import os
import sys
from tqdm import tqdm
# from tokenizer import MsTokenizer, JiebaTokenizer
import jieba
from .embedding_utils import similarity_match


# 错误字典，这里只是示例
error_msg = {
    1: "Bad input file",
    2: "Wrong input file format",
    3: "Duplicate ids in the submit files",
    4: "Not find valid submit files.",
    5: "Unequal size between the submit files and question files.",
    6: "Unaligned id information.",
    7: "None or empty answer in the submit files."
}


def report_error_msg(detail, showMsg, out_p):
    error_dict = dict()
    error_dict['errorDetail'] = detail
    error_dict['errorMsg'] = showMsg
    error_dict['score'] = 0
    error_dict['scoreJson'] = {}
    error_dict['success'] = False
    dump_2_json(error_dict, out_p)


def report_score(score, out_p):
    result = dict()
    result['success'] = True
    result['score'] = score["score"]

    # 这里{}里面的score注意保留，但可以增加其他key，比如这样：
    # result['scoreJson'] = {'score': score, 'aaaa': 0.1}
    result['scoreJson'] = {
        'score': score["score"],
        "data_query": score.get("数据查询", -1),
        "text_comprehension": score.get("文本理解", -1)
    }
    # result['scoreJson'] = score
    dump_2_json(result, out_p)


def read_jsonl(path):
    content = []
    with jsonlines.open(path, "r") as json_file:
        for obj in json_file.iter(type=dict, skip_invalid=True):
            content.append(obj)
    return content


def dump_2_json(info, path):
    with open(path, 'w') as output_json_file:
        json.dump(info, output_json_file, ensure_ascii=False, indent=4)


def tokenize(text, method="qwen"):
    # if method == "qwen":
    #     tkr = MsTokenizer(name="TongyiFinance/Tongyi-Finance-14B")
    # else:
    #     tkr = JiebaTokenizer()
    # return tkr.tokenize(text)
    return jieba.lcut(text)


def calculate_bleu_score(reference_sentence, hypothesis_sentence):
    """
    N-gram precision metric.
    """
    smoothing = SmoothingFunction().method1
    reference_tokens = [tokenize(reference_sentence)]
    hypothesis_tokens = tokenize(hypothesis_sentence)
    bleu_score = sentence_bleu(
        reference_tokens, hypothesis_tokens, smoothing_function=smoothing, auto_reweigh=True)

    return bleu_score


def calculate_t2v_score(reference_sentence, hypothesis_sentence):
    if sim_model is not None:
        t2v_score = sim_model.get_score(reference_sentence, hypothesis_sentence)
    else:
        t2v_score = similarity_match(reference_sentence, hypothesis_sentence, )
    return t2v_score


def calculate_leven_score(reference_sentence, hypothesis_sentence):
    l_score = Levenshtein.distance(reference_sentence, hypothesis_sentence)
    return l_score


def calculate_f1(reference_sentence, hypothesis_sentence):
    """
    Set F1 score.
    """
    reference_tokens = set(tokenize(reference_sentence))
    hypothesis_tokens = set(tokenize(hypothesis_sentence))
    if len(reference_tokens) == 0 or len(hypothesis_tokens) == 0:
        return 0

    commons = hypothesis_tokens & reference_tokens
    # if there are no common tokens then f1 = 0
    if len(commons) == 0:
        return 0

    prec = len(commons) / len(hypothesis_tokens)
    rec = len(commons) / len(reference_tokens)

    return 2 * (prec * rec) / (prec + rec)


def calculate_scores(reference_sentence, hypothesis_sentence):
    scores = dict()

    # n-gram precision metric
    # scores["BLEU"] = calculate_bleu_nopenalty_score(reference_sentence, hypothesis_sentence)
    # embedding, semantic metric
    scores["text2vec"] = calculate_t2v_score(reference_sentence, hypothesis_sentence)
    # precision & recall
    scores["f1_score"] = calculate_f1(reference_sentence, hypothesis_sentence)

    scores["score"] = 0.6 * scores["text2vec"] + 0.4 * scores["f1_score"]
    return scores


def evaluate_answer(reference_data, reference_answer, user_answer):
    """根据新的标准评估用户的答案"""
    score = 0.0
    if user_answer is None or user_answer == "":
        return score

    # 标准化答案和参考数据的日期格式
    user_answer = standardize_extended_date_formats(user_answer)
    reference_answer = standardize_extended_date_formats(reference_answer)
    reference_data = [standardize_extended_date_formats(i) for i in reference_data]

    # if no reference data given, evaluate the results according to the semantic matching only.
    if reference_data is not None and len(reference_data) > 0:
        score_weight = (0.6, 0.4)
        matched_data_count = sum(1.0 for data in reference_data if data in user_answer)
        score += score_weight[0] * (matched_data_count / len(reference_data))
    else:
        score_weight = (0, 1.0)

    # 计算语义相似度得分
    semantic_scores = calculate_scores(reference_answer, user_answer)
    if semantic_scores:
        score += score_weight[1] * semantic_scores["score"]

    return score


def standardize_extended_date_formats(text):
    """标准化扩展的日期格式"""
    # 定义日期格式的正则表达式及其替代格式
    patterns = [
        (r'(\d{4})年(\d{1,2})月(\d{1,2})[日号]', "{0}{1:02}{2:02}"),  # YYYY年MM月DD日
        (r'(\d{4})/(\d{1,2})/(\d{1,2})', "{0}{1:02}{2:02}"),  # YYYY/MM/DD
        (r'(\d{4})-(\d{1,2})-(\d{1,2})', "{0}{1:02}{2:02}"),  # YYYY-MM-DD
        (r'(\d{4})\.(\d{1,2})\.(\d{1,2})', "{0}{1:02}{2:02}"),  # YYYY.MM.DD
        (r'(\d{1,2})月(\d{1,2})[日号][,，](\d{4})年', "{2}{0:02}{1:02}"),  # MM月DD日,YYYY年
        (r'(\d{1,2})[日号](\d{1,2})月[,，](\d{4})年', "{2}{1:02}{0:02}"),  # DD月MM日,YYYY年
        (r'(\d{4})年?一季度', "{0}0331"),
        (r'(\d{4})年?第一季度', "{0}0331"),
        (r'(\d{4})年?Q1', "{0}0331"),
        (r'(\d{4})年?二季度', "{0}0630"),
        (r'(\d{4})年?第二季度', "{0}0630"),
        (r'(\d{4})年?Q2', "{0}0630"),
        (r'(\d{4})年?三季度', "{0}0930"),
        (r'(\d{4})年?第三季度', "{0}0930"),
        (r'(\d{4})年?Q3', "{0}0930"),
        (r'(\d{4})年?四季度', "{0}1231"),
        (r'(\d{4})年?第四季度', "{0}1231"),
        (r'(\d{4})年?Q4', "{0}1231"),
        (r'(\d{6})日期?', "{0}"),
    ]
    # 遍历所有的模式并进行替换
    for pattern, replacement_format in patterns:
        def replacement(match):
            groups = [int(g) for g in match.groups()]
            return replacement_format.format(*groups)

        text = re.sub(pattern, replacement, text)
    return text


def evaluate(correct_data, user_data):
    total_score = 0.0
    data = []
    paired_data = list(zip(correct_data, user_data))
    pbar = tqdm(len(paired_data), desc="Processing")
    for i, (correct_answer, user_answer) in enumerate(paired_data):
        score = evaluate_answer(
            correct_answer["answer_term"],
            correct_answer["answer"],
            user_answer.get("answer", "")
        )
        total_score += score  # 累加得分

        c = {
            'id': correct_answer['id'],
            'type': correct_answer['type'],
            'question': correct_answer['question'],
            'refer_answer': correct_answer['answer'],
            'refer_answer_term': correct_answer['answer_term'],
            'user_answer': user_answer.get('answer', ""),
            'score': score
        }
        data.append(c)
        if (i + 1) % 50 == 0:
            pbar.set_postfix({"score": round(score, 4)})
            pbar.update(50)
    dump_2_json(data, "./evaluate_result_detail.jsonl")

    data_df = pd.DataFrame(data)
    total_score = round(data_df["score"].mean() * 100.0, 2)
    score_dict = np.round(data_df.groupby("type")["score"].mean() * 100.0, 2).to_dict()
    score_dict["score"] = total_score

    print(f"Scores: {score_dict}", flush=True)
    return score_dict


if __name__ == "__main__":
    '''
      online evaluation 
    '''
    in_param_path = sys.argv[1]
    out_path = sys.argv[2]
    try:
        submit_path = sys.argv[3]
    except IndexError:
        submit_path = None

    # read submit and answer file from first parameter
    with open(in_param_path, 'r') as load_f:
        input_params = json.load(load_f)

    # 标准答案路径
    standard_path = input_params["fileData"]["standardFilePath"]
    print("Read standard from %s" % standard_path)

    # 选手提交的结果文件路径
    if submit_path is None:
        submit_path = input_params["fileData"]["userFilePath"]
    print("Read user submit file from %s" % submit_path)

    # 假设这是您已经下载并加载到环境中的模型，huggingface上可以下载该模型
    sim_model_path = "shibing624/text2vec-base-chinese"
    # sim_model_path = None
    if isinstance(sim_model_path, str) and os.path.exists(sim_model_path):
        from text2vec import Similarity
        sim_model = Similarity(max_seq_length=256)
    else:
        sim_model = None

    if not os.path.exists(submit_path):
        report_error_msg(error_msg[4], f"Error message: {error_msg[4]}", out_path)
        sys.exit()

    try:
        standard_labels = read_jsonl(standard_path)
        submit_preds = read_jsonl(submit_path)
    except json.JSONDecodeError as e:
        report_error_msg(e.msg, f"Error message: {error_msg[2]}", out_path)
        sys.exit()

    if len(standard_labels) != len(submit_preds):
        report_error_msg(error_msg[5], f"Error message: {error_msg[5]}", out_path)
        sys.exit()

    submit_preds = sorted(submit_preds, key=lambda s: s['id'])
    label_ids = [s["id"] for s in standard_labels]
    pred_ids = [s["id"] for s in submit_preds]
    if label_ids != pred_ids:
        report_error_msg(error_msg[6], f"Error message: {error_msg[6]}", out_path)
        sys.exit()

    for s in submit_preds:
        ans = s.get("answer", "")
        if ans is None:
            report_error_msg(error_msg[7], f"Error message: {error_msg[7]}", out_path)
            sys.exit()

    try:
        eval_score = evaluate(standard_labels, submit_preds)
        report_score(eval_score, out_path)
    except Exception as e:
        report_error_msg(f"{e}", f"Error message: {e}", out_path)

