import pandas as pd
import numpy as np
import os
import re
import jieba
from multiprocessing import Pool
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


category_names = ['銀行', '信用卡', '匯率', '台積電', '台灣', '日本']
# News_UpperBound = 10


def read_data(indexes=None):
    # Read the file
    print('reading data...')
    df = pd.read_excel("./data/hw1_text.xlsx")
    if indexes is not None:
        df = df[df['編號'].isin(indexes)]
    print('length: {}'.format(len(df)))
    title_arr = np.array(list(df.loc[:, '標題']))
    content_arr = np.array(list(df.loc[:, '內容']))
    news_list = list(map(lambda x, y: x + y, title_arr, content_arr))

    # news_list = news_list[0:News_UpperBound]
    # preprocess(get rid of punctuation and english/number characters)
    for i in range(len(news_list)):
        news_list[i] = ''.join(re.findall(u'([\u4e00-\u9fff])', news_list[i]))
    print('done')
    return news_list


def cut_news(data):
    P = Pool(processes=4)
    jieba.load_userdict('./dict.txt.big')
    data = P.map(tokenize, data)
    P.close()
    P.join()
    return data


def tokenize(sentence):
    tokens = jieba.lcut(sentence)
    return tokens


def calculate(news, save_name):
    print('==> Counting...')
    vec = CountVectorizer(max_features=7000)
    X = vec.fit_transform(news)
    tf = X.toarray().sum(axis=0)
    df = (X.toarray() > 0).sum(axis=0)
    print('==> Calculating tf-idf...')
    tfidf = TfidfTransformer()
    Y = tfidf.fit_transform(X)
    Y = Y.toarray().sum(axis=0)

    df = pd.DataFrame(list(zip(vec.get_feature_names(), tf, df, Y)), columns=['token', 'tf', 'df', 'tf-idf'])
    df = df.sort_values(by=['tf-idf'], ascending=False)
    out_path = 'result/bonus/{}.csv'.format(save_name)
    print('==> Saving to {} ...'.format(out_path))
    df.to_csv(out_path, index=False)


def main():
    # Full list
    if os.path.exists('temp.npy'):
        print('Loading News...')
        news = np.load('temp.npy', allow_pickle=True)
    else:
        news = read_data()
        news = cut_news(news)
        news = np.array([' '.join(article) for article in news])
        np.save('temp', news)

    calculate(news, save_name='full')

    # Per category
    categories = np.load('category.npy', allow_pickle=True)
    for i, index_list in enumerate(categories):
        if os.path.exists('{}.npy'.format(category_names[i])):
            print('Loading News...')
            news = np.load('{}.npy'.format(category_names[i]), allow_pickle=True)
        else:
            news = read_data(indexes=index_list)
            news = cut_news(news)
            news = np.array([' '.join(article) for article in news])
            np.save(category_names[i], news)

        calculate(news, save_name=category_names[i])


if __name__ == "__main__":
    main()
