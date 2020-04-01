#coding=utf-8
import pandas as pd
from whoosh.index import create_in,open_dir
from whoosh.fields import *
from whoosh.query import *
from whoosh.qparser import QueryParser
import numpy as np
import re
import jieba.analyse
from jieba.analyse.analyzer import ChineseAnalyzer
import os

queries = [
    '銀行',
    '信用卡',
    '匯率',
    '台積電',
    '台灣',
    '日本'
]
News_UpperBound = 1001
category_num = 40000
def read_data():
    # Read the file
    print('reading data...')
    df = pd.read_excel("hw1_text.xlsx")
    title_arr = np.array(list(df.loc[:, '標題']))
    content_arr = np.array(list(df.loc[:, '內容']))
    news_list = list(map(lambda x, y: x + y, title_arr, content_arr))
    news_list = news_list[0:News_UpperBound]
    # preprocess(get rid of punctuation and english/number characters)
    for i in range(len(news_list)):
        news_list[i] = ''.join(re.findall(u'([\u4e00-\u9fff])', news_list[i]))
    print('done')
    return news_list
def main():
    jieba.enable_parallel(4)
    jieba.load_userdict('./auxiliary_data/dict.txt.big')
    analyzer = ChineseAnalyzer()
    schema = Schema(title = TEXT(stored = True),
                content = TEXT(stored = True, analyzer = analyzer))
    if not os.path.exists('./indexdir/'):
        os.mkdir('./indexdir/')
        ix = create_in('./indexdir/', schema)
        news = read_data()
        writer = ix.writer()
        print('Add documents...')
        for i, x in enumerate(news):
            if i % 1000 == 0:
                print('\t%d documents have been added.' % i)
            writer.add_document(title = '%06d'%(i+1), content = x)
        writer.commit()
    else:
        print('Directly open previous indexed directory...')
        ix = open_dir('./indexdir')

    article = []
    parser = QueryParser('content', schema = ix.schema)
    with ix.searcher() as searcher:
        for keyword in queries:
            article.append([])
            print("result of ", keyword)
            q = keyword
            q = parser.parse(keyword)
            results = searcher.search(q, limit = category_num)
            for i in range(category_num):
                if i == len(results):
                    break
                article[-1].append(int(results[i]['title']))
    article = np.array(article)
    np.save('./category', article)
    print(article)
if __name__ == '__main__':
    main()