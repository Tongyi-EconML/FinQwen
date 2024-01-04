#!/urs/bin/env python
# -*- coding:utf-8 -*-
"""

:Author:  Houyanlong
:Create:  2023/11/22 13:26
Copyright (c) 2018, Lianjia Group All Rights Reserved.
"""
import re
import json
p = "\d+\.\d+"
bd = '[=,.?!@#$%^&*()_+:"<>/\[\]\\`~——，。、《》？；’：“【】、{}|·！￥…（）-]'
weishu_dict = {
    "1":1,
    "2":2,
    "3":3,
    "4":4,
    "5":5,
    "一":1,
    "二":2,
    "三":3,
    "四":4,
    "五":5,
    "两":2
}

def refine_xiaoshu(answer, ji):
    a = answer.strip()
    a_li = a.split(".")
    xiaoshu_part = re.match("\d+", a_li[-1]).group()
    new_xiaoshu_part = ""
    cur = 0
    while cur < weishu_dict[ji]:
        if len(xiaoshu_part) > cur:
            new_xiaoshu_part += xiaoshu_part[cur]
        else:
            new_xiaoshu_part += "0"
        cur += 1
    a_li[-1] = a_li[-1].replace(xiaoshu_part, new_xiaoshu_part)
    return ".".join(a_li)

def post_process_answer(qustion, answer):
    ans = answer.replace("\"", "")
    ans = ans.replace("[", "").replace("]", "").replace("(","").replace(")","")
    q = qustion
    if "费率" in q and len(ans.split(",")) == 1 and '%' not in ans:
        ans += '%'
    if "百分比" in q and len(ans.split(",")) == 1 and '%' not in ans:
        ans += '%'
    if "涨跌幅" in q and '%' not in ans:
        m = re.search(p, ans)
        if m:
            ans = ans.replace(m.group(), m.group()+'%')
    if "取整" in q and ".0" in ans:
        ans = ans.replace('.0', "")
    q_li = re.split(bd, q)
    for sub_q in q_li:
        if "小数" in sub_q:
            if "不超过" in sub_q:
                continue
            else:
                for ji in weishu_dict:
                    if ji+"位" in sub_q:
                        ans_list = ans.split(",")
                        if len(ans_list) == 1:
                            ans_list[-1] = refine_xiaoshu(ans,ji)
                        else:
                            for i,a0 in enumerate(ans_list):
                                if a0.find(".") > 0:
                                    ans_list[i] = refine_xiaoshu(a0,ji)
                                    break
                        ans = ", ".join(ans_list)
                        break
    return ans

if __name__ == '__main__':
    input = 'groundtruth_1124.json'
    output = 'sql_answer_1124.json'
    with open(input) as f:
        data = json.loads(f.read())
        xiaoshu = 0
        for d in data:
            ans = d["a"]
            ans = ans.replace("\"", "")
            ans = ans.replace("[", "").replace("]", "").replace("(","").repalce(")","")
            q = d["q"]
            if "费率" in q and len(ans.split(",")) == 1 and '%' not in ans:
                ans += '%'
            if "百分比" in q and len(ans.split(",")) == 1 and '%' not in ans:
                ans += '%'
            if "涨跌幅" in q and '%' not in ans:
                m = re.search(p, ans)
                if m:
                    ans = ans.replace(m.group(), m.group()+'%')
            if "取整" in q and ".0" in ans:
                ans = ans.replace('.0', "")
            q_li = re.split(bd, q)
            for sub_q in q_li:
                if "小数" in sub_q:
                    if "不超过" in sub_q:
                        xiaoshu += 1
                        print(ans)
                        continue
                    else:
                        for ji in weishu_dict:
                            if ji+"位" in sub_q:
                                xiaoshu += 1
                                ans_list = ans.split(",")
                                if len(ans_list) == 1:
                                    ans_list[-1] = refine_xiaoshu(ans)
                                else:
                                    for i,a0 in enumerate(ans_list):
                                        if a0.find(".") > 0:
                                            ans_list[i] = refine_xiaoshu(a0)
                                            break
                                ans = ", ".join(ans_list)
                                break

            d["a"] = ans
        with open(output, "w") as f1:
            f1.write(json.dumps(data, ensure_ascii=False))