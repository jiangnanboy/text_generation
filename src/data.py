import json
import os
import csv
import logging

def read_json(json_path, csv_path):
    '''
    将json数据解析，并写入csv文件中
    :param json_path:
    :param csv_path:
    :return:
    '''
    with open(json_path, 'r', encoding='utf8') as f_read, open(csv_path, 'w', encoding="utf8", newline='') as f_write:
        logging.info('data processing...')
        csv_write = csv.writer(f_write)
        # csv_write.writerow(['title', 'keywords', 'text'])
        for line in f_read:
            line_context = []
            set_ner = set()
            data = json.loads(line)
            title = data.get("title")
            content = data.get("content")
            content_ner = data.get("content_ner")
            if title is None or content is None or content_ner is None:
                continue
            if len(title) == 0 or len(content) == 0 or len(content_ner) == 0:
                continue
            for content_json in content_ner:
                set_ner.add(content_json["value"].strip())
            if len(set_ner) > 0:
                line_context.append(title)
                line_context.append(' '.join(set_ner))
                line_context.append(content)
                csv_write.writerow(line_context)
        logging.info("data processed!")

if __name__ == '__main__':
    path = os.path.abspath(os.path.join(os.getcwd(), ".."))
    csv_data_path = os.path.join(path, "data/news.csv")
    raw_data_path = os.path.join(path, "data/news.txt")
    read_json(raw_data_path, csv_data_path)