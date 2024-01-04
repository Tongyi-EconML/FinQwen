import glob
import pdfplumber
import re
from collections import defaultdict
import json
from file_processor import read_jsonl
from config import logger



class PDFProcessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.pdf = pdfplumber.open(filepath)
        self.all_text = defaultdict(dict)
        self.allrow = 0
        self.last_num = 0
        self.title = ""

    def check_lines(self, page, top, buttom):
        try:
            lines = page.extract_words()
        except Exception as e:
            logger.error(f'页码: {page.page_number}, 抽取文本异常，异常信息: {e}')
            return ''
        # empty util
        check_re = '(?:。|；|单位：元|单位：万元|币种：人民币)$'
        page_top_re = '(招股意向书(?:全文)?(?:（修订版）|（修订稿）|（更正后）)?)'

        text = ''
        last_top = 0
        last_check = 0
        if top == '' and buttom == '':
            if len(lines) == 0:
                logger.info(f'{page.page_number}页无数据, 请检查！')
                return ''
        for l in range(len(lines)):
            each_line = lines[l]
            if top == '' and buttom == '':
                if abs(last_top - each_line['top']) <= 2:
                    text = text + each_line['text']
                elif last_check > 0 and (page.height * 0.9 - each_line['top']) > 0 and not re.search(check_re, text):
                    if '\n' not in text and re.search(page_top_re,text):
                        text = text + '\n' + each_line['text'] 
                    else:
                        text = text + each_line['text']
                else:
                    if text == '':
                        text = each_line['text']
                    else:
                        text = text + '\n' + each_line['text']
            elif top == '':
                if each_line['top'] > buttom:
                    if abs(last_top - each_line['top']) <= 2:
                        text = text + each_line['text']
                    elif last_check > 0 and (page.height * 0.85 - each_line['top']) > 0 and not re.search(check_re, text):
                        if '\n' not in text and re.search(page_top_re,text):
                            text = text + '\n' + each_line['text'] 
                        else:
                            text = text + each_line['text']
                    else:
                        if text == '':
                            text = each_line['text']
                        else:
                            text = text + '\n' + each_line['text']
            else:
                if each_line['top'] < top and each_line['top'] > buttom:
                    if abs(last_top - each_line['top']) <= 2:
                        text = text + each_line['text']
                    elif last_check > 0 and (page.height * 0.85 - each_line['top']) > 0 and not re.search(check_re, text):
                        if '\n' not in text and re.search(page_top_re,text):
                            text = text + '\n' + each_line['text'] 
                        else:
                            text = text + each_line['text']
                    else:
                        if text == '':
                            text = each_line['text']
                        else:
                            text = text + '\n' + each_line['text']                    
            last_top = each_line['top']
            last_check = each_line['x1'] - page.width * 0.83

        return text

    def drop_empty_cols(self, data):
        # 删除所有列为空数据的列
        transposed_data = list(map(list, zip(*data)))
        filtered_data = [col for col in transposed_data if not all(cell == '' for cell in col)]
        result = list(map(list, zip(*filtered_data)))
        return result

    @staticmethod
    def keep_visible_lines(obj):
        """
        If the object is a ``rect`` type, keep it only if the lines are visible.

        A visible line is the one having ``non_stroking_color`` not null.
        """
        if obj['object_type'] == 'rect':
            if obj['non_stroking_color'] is None:
                return False
            if obj['width'] < 1 and obj['height'] < 1:
                return False
            return obj['width'] >= 1 and obj['height'] >= 1 and obj['non_stroking_color'] is not None
        # if obj['object_type'] == 'char':
        #     return obj['stroking_color'] is not None and obj['non_stroking_color'] is not None
        return True

    def extract_text_and_tables(self, page):
        buttom = 0
        # page = page.filter(self.keep_visible_lines)
        try:
            tables = page.find_tables()
        except:
            tables = []
        if len(tables) >= 1:
            count = len(tables)
            for table in tables:
                if table.bbox[3] < buttom:
                    pass
                else:
                    count -= 1
                    top = table.bbox[1]
                    text = self.check_lines(page, top, buttom)
                    text_list = text.split('\n')
                    for _t in range(len(text_list)):
                        self.all_text[self.allrow] = {'page': page.page_number, 'allrow': self.allrow,
                                                      'type': 'text', 'inside': text_list[_t]}
                        self.allrow += 1

                    buttom = table.bbox[3]
                    new_table = table.extract()
                    r_count = 0
                    for r in range(len(new_table)):
                        row = new_table[r]
                        if row[0] is None:
                            r_count += 1
                            for c in range(len(row)):
                                if row[c] is not None and row[c] not in ['', ' ']:
                                    if new_table[r - r_count][c] is None:
                                        new_table[r - r_count][c] = row[c]
                                    else:
                                        new_table[r - r_count][c] += row[c]
                                    new_table[r][c] = None
                        else:
                            r_count = 0

                    end_table = []
                    for row in new_table:
                        if row[0] != None:
                            cell_list = []
                            cell_check = False
                            for cell in row:
                                if cell != None:
                                    cell = cell.replace('\n', '')
                                else:
                                    cell = ''
                                if cell != '':
                                    cell_check = True
                                cell_list.append(cell)
                            if cell_check == True:
                                end_table.append(cell_list)
                    end_table = self.drop_empty_cols(end_table)

                    for row in end_table:
                        self.all_text[self.allrow] = {'page': page.page_number, 'allrow': self.allrow,
                                                      'type': 'excel', 'inside': str(row)}
                        # self.all_text[self.allrow] = {'page': page.page_number, 'allrow': self.allrow, 'type': 'excel',
                        #                               'inside': ' '.join(row)}
                        self.allrow += 1

                    if count == 0:
                        text = self.check_lines(page, '', buttom)
                        text_list = text.split('\n')
                        for _t in range(len(text_list)):
                            self.all_text[self.allrow] = {'page': page.page_number, 'allrow': self.allrow,
                                                          'type': 'text', 'inside': text_list[_t]}
                            self.allrow += 1

        else:
            text = self.check_lines(page, '', '')
            text_list = text.split('\n')
            for _t in range(len(text_list)):
                self.all_text[self.allrow] = {'page': page.page_number, 'allrow': self.allrow,
                                              'type': 'text', 'inside': text_list[_t]}
                self.allrow += 1

        first_re = '[^计](招股意向书(?:全文)?(?:（修订版）|（修订稿）|（更正后）)?)'
        end_re = '^(?:\d|\\|\/|第|共|页|-|_| ){1,}'
        if self.last_num == 0:
            try:
                first_text = str(self.all_text[1].get('inside',""))
                if re.search(first_re, first_text) and not '[' in first_text:
                    self.all_text[1]['type'] = '页眉'
                if first_text.endswith('有限公司'):
                    self.title = first_text
                end_text = str(self.all_text[len(self.all_text) - 1].get('inside',""))
                if len(end_text) <= 15 and re.search(end_re, end_text) and not '[' in end_text:
                    self.all_text[len(self.all_text) - 1]['type'] = '页脚'
            except Exception as e:
                logger.error(f"首页页眉页脚出错, 错误信息: {e}")
        else:
            try:
                if self.all_text.get(self.last_num + 1) is not None:
                    first_text = str(self.all_text[self.last_num + 1].get('inside',""))
                    if re.search(first_re, first_text) and '[' not in first_text:
                        self.all_text[self.last_num + 1]['type'] = '页眉'
                end_text = str(self.all_text[len(self.all_text) - 1].get('inside',""))
                if len(end_text) <= 15 and re.search(end_re, end_text) and '[' not in end_text:
                    self.all_text[len(self.all_text) - 1]['type'] = '页脚'
            except Exception as e:
                logger.error(f"页码: {page.page_number}, 页眉页脚处理出错, 错误信息: {e}")

        self.last_num = len(self.all_text) - 1


    def process_pdf(self):
        for i in range(len(self.pdf.pages)):
            self.extract_text_and_tables(self.pdf.pages[i])


    def save_all_text(self, path):
        with open(path, 'w', encoding='utf-8') as file:
            for key in self.all_text.keys():
                file.write(json.dumps(self.all_text[key], ensure_ascii=False) + '\n')


def process_all_pdfs_in_folder(folder_path,output_folder_path,need_title=False):
    file_paths = glob.glob(f'{folder_path}/*')
    file_paths = sorted(file_paths, reverse=True)

    for file_path in file_paths:
        logger.info(f'文档目录: {file_path}')
        try:
            processor = PDFProcessor(file_path)
            processor.process_pdf()
            if need_title:
                save_path = output_folder_path + (processor.title +'.txt' if processor.title != "" else file_path.split('/')[-1].replace('.PDF', '.txt'))
            else:
                save_path = output_folder_path + file_path.split('/')[-1].replace('.PDF', '.txt')
            processor.save_all_text(save_path)
            # new_pdf_file = output_folder_path + (processor.title+'.PDF' if processor.title != "" else file_path.split('/')[-1])
            # os.system(f'cp {file_path} {new_pdf_file}')
        except Exception as e:
            logger.error(f'需要重新检查！错误信息: {e}')

def extract_single_pure_text(file_path, output_dir):
    content = read_jsonl(file_path)
    full_text = ""
    for line in content:
        text = line.get("inside", "")
        text_type = line.get("type", -2)

        if text_type in ("页眉", "页脚") or text == "":
            continue
        elif text_type == "excel":
            full_text += "\t".join(text) + '\n'
        else:
            full_text += text + "\n"
    file_name = file_path.split('/')[-1]
    with open(output_dir + file_name,'w',encoding='utf-8') as f:
        f.write(full_text)

def extract_all_pure_text(folder_path, output_folder_path):
    file_paths = glob.glob(f'{folder_path}/*')
    for file_path in file_paths:
        extract_single_pure_text(file_path, output_folder_path)
    


if __name__ == '__main__':
    # print(1)
    pdf_folder_path = r'./data/pdf/'
    output_folder_path = r'./data/pdf_txt_new/' 
    process_all_pdfs_in_folder(pdf_folder_path, output_folder_path)

    # text_folder_path = r'data/pdf_txt/'
    # output_folder_path = r'data/pdf_pure_txt/'
    # extract_all_pure_text(text_folder_path, output_folder_path)

    # pdf_file = r'/Users/yuzhewen/Desktop/bs-challenge-financial/data/pdf/深圳信立泰药业股份有限公司.PDF'
    # processor = PDFProcessor(pdf_file)
    # processor.process_pdf()
    # processor.save_all_text('./data/pdf_txt/深圳信立泰药业股份有限公司.txt')

