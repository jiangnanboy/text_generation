import pandas as pd
import os

import sys
sys.path.append('/home/user/project/text_generation/')


def read_data(data_path, columns):
    '''
    pandas读取data
    :param data_path:
    :param columns:
    :return:
    '''
    train_data = pd.read_csv(data_path, header=None, names=columns)
    return train_data

def split_data(pd, split_ratio):
    '''
    pandas划分数据
    :param pd:
    :return:
    '''
    train_set = pd.sample(frac=split_ratio, replace=False)
    val_set = pd[~pd.index.isin(train_set.index)]

    return train_set, val_set

if __name__ == '__main__':
    path = os.path.abspath(os.path.join(os.getcwd(), ".."))
    csv_data_path = os.path.join(path, "data/news.csv")
    columns = [
        'title',
        'keywords',
        'content'
    ]
    pd_data = read_data(csv_data_path, columns)

    print(pd_data.head(10))
