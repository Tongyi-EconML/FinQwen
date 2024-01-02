import sqlite3
import re
import csv
import pandas as pd
import faulthandler
faulthandler.enable()
term_list_1 = ['基金股票持仓明细','基金债券持仓明细','基金可转债持仓明细']
conn = sqlite3.connect('/tcdata/bs_challenge_financial_14b_dataset/dataset/博金杯比赛数据.db')
cs = conn.cursor()
new_question_file_dir = '/app/intermediate/question_SQL_V6.csv'
new_question_file = pd.read_csv(new_question_file_dir,delimiter = ",",header = 0)
g = open('/app/intermediate/question_SQL_V6_exed.csv', 'w', newline='', encoding = 'utf-8-sig') 
csvwriter = csv.writer(g)
csvwriter.writerow(['问题id','问题','SQL语句','能否成功执行','执行结果','List'])
print('B02_started')
for cyc in range(1000):
    if cyc % 50 == 0:
        print(cyc)
    SQL_list = list()
    SQL_exe_flag = 0
    Use_similar_table_flag = 0
    SQL_exe_result = 'N_A'
    temp_sql = new_question_file[cyc:cyc+1]['SQL语句'][cyc]
    temp_sql = temp_sql.replace("B股票日行情表",'A股票日行情表')   
    temp_sql = temp_sql.replace("创业板日行情表",'A股票日行情表')   
    if " 股票日行情表" in temp_sql:
        temp_sql = temp_sql.replace(" 股票日行情表",' A股票日行情表')  
    if " 港股日行情表" in temp_sql:
        temp_sql = temp_sql.replace(" 港股日行情表",' 港股票日行情表')  
    temp_sql = temp_sql.replace("”",'').replace("“",'')                                    
    
    origin_success_flag = 0
    try:
        cs.execute(temp_sql)
        cols = cs.fetchall()
        SQL_exe_result = str(cols)
        origin_success_flag = 1
    except:
        for item in term_list_1:
            if item in temp_sql:
                Use_similar_table_flag = 1
                original_item = item
                break
        if Use_similar_table_flag == 1:
            for item in temp_sql:
                try:
                    cs.execute(temp_sql.replace(original_item,item))
                    cols = cs.fetchall()
                    SQL_exe_result = str(cols)
                    SQL_exe_flag = 2
                    break
                except:
                    pass

    csvwriter.writerow([str(new_question_file[cyc:(cyc+1)]['问题id'][cyc]),
                    str(new_question_file[cyc:(cyc+1)]['问题'][cyc]),
                    temp_sql,
                    origin_success_flag,
                    SQL_exe_result])
g.close()
exit()
