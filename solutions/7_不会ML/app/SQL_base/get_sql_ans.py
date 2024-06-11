import sqlite3
import sys
import os 
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from typing import List, Tuple
from Config.set_config import Config,logger
# from post_process_sql_result import post_process_answer

class Sqldb:
    def __init__(self,p_db_path: str) -> None:
        self.conn = sqlite3.connect(p_db_path)

    def __del__(self):
        self.conn.close()
    
    def commit(self) -> None:
        self.conn.commit()

    # 创建表
    def create_table(self,p_tb_name: str,p_tb_columns: List[str],p_tb_columns_type: List[str]) -> None:
        # 创建游标
        cur = self.conn.cursor()
        p_tb_column_lines = [column_name + ' ' + column_type for column_name, column_type in zip(p_tb_columns,p_tb_columns_type)]
        sql_str = f"CREATE TABLE {p_tb_name} ({','.join(p_tb_column_lines)})"
        logger.info('sql: ' + sql_str)
        cur.execute(sql_str)

    # 删除表
    def drop_table(self,p_tb_name: str) -> None:
        # 创建游标
        cur = self.conn.cursor()
        sql_str = 'DROP TABLE ' + p_tb_name
        logger.info('sql: ' + sql_str)
        cur.execute(sql_str)

    # 插入数据
    def insert_data(self, p_tb_name: str, p_data: List[Tuple]) -> None:
        # 创建游标
        cur = self.conn.cursor()
        sql_str = f"INSERT INTO {p_tb_name} VALUES({','.join(['?'] * len(p_data[0]))})"
        logger.info('sql: ' + sql_str)
        cur.executemany(sql_str,p_data)


    # 查询数据
    def select_data(self, p_sql_str: str) -> List[Tuple]:
        # 创建游标
        cur = self.conn.cursor()
        logger.info('sql: ' + p_sql_str)
        cur.execute(p_sql_str)
        sql_answer = cur.fetchall()
        if len(sql_answer) > 10:
            raise ValueError("too many query results")
        return sql_answer
    
def main():
    sqldb = Sqldb(Config.db_sqlite_url)
    # import pandas as pd
    # df = list(pd.read_excel(Config.sql_answer_template_path)['SQL答案'].values)
    # error = 0
    # for q in df:
    #     try:
    #         result = sqldb.select_data(q)
    #         print(result)
    #     except:
    #         error+=1
    #     print('***'*15)
    # print('错误个数：',error)
        
        
    
#     # result = sqldb.select_data("select 股票代码 ,round(([收盘价(元)] -[昨收盘(元)])/[昨收盘(元)]*100,2)||'%' from A股票日行情表 where 交易日 = '20190312' and 股票代码 in (select 股票代码 from A股公司行业划分表 where 行业划分标准 = '中信行业分类' and 一级行业名称 = '消费者服务' and 交易日期 = '20190312') order by ([收盘价(元)] -[昨收盘(元)])/[昨收盘(元)] desc limit 1   ")
#     # print(result)
#     result = sqldb.select_data("""
# SELECT ROUND(((EndPrice - StartPrice) / StartPrice) * 100, 2) AS 年化收益率
# FROM (SELECT (SELECT [收盘价(元)] FROM A股票日行情表 WHERE 股票代码 = '603937' AND 交易日 LIKE '2020%' ORDER BY 交易日 DESC LIMIT 1) AS EndPrice, (SELECT [收盘价(元)] FROM A股票日行情表 WHERE 股票代码 = '603937'  AND 交易日 LIKE '2020%' ORDER BY 交易日 LIMIT 1) AS StartPrice) AS Prices;
# """)
    result = sqldb.select_data(
        """
SELECT ROUND(((EndPrice - StartPrice) / StartPrice) * 100, 2) AS 年化收益率
FROM (SELECT (SELECT [收盘价(元)] FROM A股票日行情表 WHERE 股票代码 = '603937' AND 交易日 LIKE '2020%' ORDER BY 交易日 DESC LIMIT 1) AS EndPrice, (SELECT [收盘价(元)] FROM A股票日行情表 WHERE 股票代码 = '603937'  AND 交易日 LIKE '2020%' ORDER BY 交易日 LIMIT 1) AS StartPrice) AS Prices;
        """
    )
    print(result)
    print(result[0][0])
    print(type(result))
# 请帮我计算，在20190312，中信行业分类划分的一级行业为消费者服务行业中，涨跌幅最大股票的股票代码是？涨跌幅是多少？百分数保留两位小数。股票涨跌幅定义为：（收盘价 - 前一日收盘价 / 前一日收盘价）* 100%。
    # print(post_process_answer("请帮我计算，在20210105，中信行业分类划分的一级行业为综合金融行业中，涨跌幅最大股票的股票代码是？涨跌幅是多少？百分数保留两位小数。股票涨跌幅定义为：（收盘价 - 前一日收盘价 / 前一日收盘价）* 100%。", str(result)))
if __name__ == '__main__':
    main()