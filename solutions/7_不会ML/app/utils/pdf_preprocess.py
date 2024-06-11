import pdfplumber
import os
from tqdm import tqdm
import re 
from ocr import get_ocr
ocr = get_ocr()
from PIL import Image, ImageEnhance, ImageFilter
from pdf2image import convert_from_path
import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Config.set_config import Config
import pandas as pd

def check_headers(pdf_path, header_height=50): # 检查PDF页眉内容是否重复
    headers = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[3:10]:
            # 提取每页顶部的区域
            top = page.crop((0, 0, page.width, header_height))
            text = top.extract_text()
            if text:
                headers.append(text.strip())

    # 检查是否有重复的页眉内容
    if len(set(headers)) < len(headers):
        return True  # 存在页眉
    else:
        return False  # 不存在页眉
        
def header_process(pdf_path,header_text,id):
    pattern = r'.*(有限公司|股份)'
    if '招股意向书' in header_text:
        end_index = header_text.find('招股意向书')
        header_text = header_text[:end_index]
    if '首次公开' in header_text:
        end_index = header_text.find('首次公开')
        header_text = header_text[:end_index]
    if '重大事项' in header_text:
        end_index = header_text.find('重大事项')
        header_text = header_text[:end_index]
    if  not  header_text : # 这个时候代表页眉有图片
        tmp_im  = convert_from_path(pdf_path, dpi=500, first_page=4, last_page=4)
        header_area = (0, 0, tmp_im[0].width, 700)
        # 使用PIL的crop函数裁剪出页眉部分的图像
        header_image = tmp_im[0].crop(header_area)
        result, _ = ocr(header_image)
        ocr_text = ''
        if result:
            ocr_result = [line[1] for line in result]
            ocr_text += "\n".join(ocr_result)
        matches = re.search(pattern, ocr_text)
        if matches:
            header_text = matches.group()
        else:
            header_text = ''
    return header_text
    
def process_pdf(pdf_path,id):
    with pdfplumber.open(pdf_path) as pdf:
        all_text = ''  # 用于存储提取的文本内容
        header_text = ''  # 用于存储页眉内容
        for i,page in enumerate(pdf.pages):
            try:
                text = page.extract_text()
                if text:
                    lines = text.split('\n')
                    # 对于第二页，提取页眉内容
                    if i == 3:  # 页码从4开始，第二页的索引为1
                        header_text = lines[0]  # 假设页眉在第一行
                    # 从第一页开始去除页眉和页脚
                    text_without_header_footer = '\n'.join(lines[1:-1])  # 去除第一行和最后一行
                    all_text += text_without_header_footer + '\n\n'
                # 检查是否存在表格，并仅处理存在表格的页面
                if page.extract_tables():
                    markdown_content = ""
                    for table in page.extract_tables():
                        markdown_table = ''  # 存储当前表格的Markdown表示
                        for i, row in enumerate(table):
                            # 移除空列，这里假设空列完全为空，根据实际情况调整
                            row = [cell for cell in row if cell is not None and cell != '']
                            # 转换每个单元格内容为字符串，并用竖线分隔
                            processed_row = [str(cell).strip() if cell is not None else "" for cell in row]
                            markdown_row = '| ' + ' | '.join(processed_row) + ' |\n'
                            markdown_table += markdown_row
                            # 对于表头下的第一行，添加分隔线
                            if i == 0:
                                separators = [':---' if cell.isdigit() else '---' for cell in row]
                                markdown_table += '| ' + ' | '.join(separators) + ' |\n'
                        all_text += markdown_table + '\n'
            except Exception as e:
                # 进行ocr处理
                tmp_img =  convert_from_path(pdf_path, dpi=500, first_page=i+1, last_page=i+1) #获取第一页内容
                result, _ = ocr(tmp_img[0])
                ocr_text = ''
                if result:
                    ocr_result = [line[1] for line in result]
                    for res in ocr_result:
                        ocr_text+=res+'\n'
                all_text +=ocr_text+'\n'
                print("利用OCR处理第{}页文件".format(i))  
    pattern = r'.*(有限公司|股份)'
    print(header_text)
    header_text = header_process(pdf_path,header_text,id)
    print(header_text)
    if not header_text or not check_headers(pdf_path): # 如果没有页眉或者页眉识别失败的情况下
        first_text = pdf.pages[0].extract_text() # 获取首页的文本内容
        if first_text: # 如果第一页可以提取文字
            first_text =first_text.replace(' ','') 
            matches = re.search(pattern, first_text)
            if matches:
                header_text = matches.group()
        
            else: # 如果第一页匹配不到公司的名称，就去第四页页眉上提取公司的名称
                header_text= header_process(pdf_path,header_text,id)
        else:
            tmp_img =  convert_from_path(pdf_path, dpi=500, first_page=1, last_page=1) #获取第一页内容
            result, _ = ocr(tmp_img[0])
            ocr_text = ''
            if result:
                ocr_result = [line[1] for line in result]
                for res in ocr_result:
                    ocr_text+=res+'\n'
            matches = re.search(pattern, ocr_text)
            if matches:
                header_text = matches.group()
            else:
                header_text = f'查找失败_{id}'
                    
    header_text = header_text.replace(' ','')
    output_filename = header_text + '.txt' if header_text else 'output.txt'
    output_path = os.path.join(Config.txt_path,output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(all_text)
    print(f'内容已保存到文件: {output_path}')
    return output_filename
    
def main(): 
    main_path = Config.pdf_path
    pdf2txt_name = []
    pdf_list = [f for f in os.listdir(main_path) if f.endswith('.PDF') and not f.startswith('.')] # 避免出现 .为首的隐藏文件
    # txt_list = [f for f in os.listdir(Config.txt_path) if f.endswith('.txt') and not f.startswith('.')]
    # pdf_list = pdf_list[5:]
    for pdf in tqdm(pdf_list):
        # file_name = pdf.replace('pdf','txt')
        # cur_file_name = os.path.join(Config.txt_path,file_name)
        # if cur_file_name in txt_list:
        #     continue
        pdf_file_path = os.path.join(main_path,pdf)
        output_filename = process_pdf(pdf_file_path,pdf)
        pdf2txt_name.append(output_filename)
    df = pd.DataFrame()
    df['txt_name'] = pdf2txt_name
    df.to_csv(Config.company_name_res,index=False)
if __name__=='__main__':
    main()
