import jsonlines
import pandas as pd
import csv
import json
def read_jsonl(path):
    content = []
    with jsonlines.open(path, "r") as json_file:
        for obj in json_file.iter(type=dict, skip_invalid=True):
            content.append(obj)
    return content


def write_jsonl(path, content):
    with jsonlines.open(path, "w") as json_file:
        json_file.write_all(content)
        
cont = read_jsonl('/tcdata/question_v2.json')        
g = open('/app/intermediate/question_csv.csv', 'w', newline='', encoding = 'utf-8-sig') 
csvwriter = csv.writer(g)
csvwriter.writerow(['问题id','问题'])
for cyc in range(1000):
    temp_question = cont[cyc]['question']
    temp_question = temp_question .replace(' ','')
    csvwriter.writerow([str(cont[cyc]['id']),cont[cyc]['question']])

g.close()
exit()