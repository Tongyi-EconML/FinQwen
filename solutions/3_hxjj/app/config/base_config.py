
import logging
WORK_DIR = ''

logger = logging.getLogger()
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler(WORK_DIR + "log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


#DB_FILE_PATH = '/tcdata/bs_challenge_financial_14b_dataset/dataset/博金杯比赛数据.db'
DB_FILE_PATH = WORK_DIR + 'data/博金杯比赛数据.db'
PDF_TXT_PATH = WORK_DIR + 'data/pdf_txt_new/'