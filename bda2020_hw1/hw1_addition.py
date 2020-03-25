import pandas as pd
import numpy as np
import re
import jieba
from multiprocessing import Pool

# News_UpperBound = 10

def read_data():
    #Read the file
    print('reading data...')
    df = pd.read_csv("./data/hw1_text_all.csv")
    title_arr = np.array(list(df.loc[:,'標題']))
    content_arr = np.array(list(df.loc[:,'內容']))
    news_list = list(map(lambda x,y:x+y,title_arr,content_arr))

    # news_list = news_list[0:News_UpperBound]
    #preprocess(get rid of punctuation and english/number characters)
    for i in range(len(news_list)):
        news_list[i] = ''.join(re.findall(u'([\u4e00-\u9fff])',news_list[i]))
    print('done')
    return news_list

def cut_news(data):
    P = Pool(processes=4) 
    data = P.map(tokenize, data)
    P.close()
    P.join()
    return data

def jieba_cut(news_list):
    jieba.load_userdict('./dict.txt.big')

def tokenize(sentence):
    tokens = jieba.lcut(sentence)
    return tokens

def main():
    news = read_data()
    news = cut_news(news)
    print(news)

if __name__ == "__main__":
    main()