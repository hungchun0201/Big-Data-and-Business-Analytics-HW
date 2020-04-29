import pandas as pd
import numpy as np
import os
import re
import jieba
from multiprocessing import Pool
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# News_UpperBound = 10
categories = ['news', 'bbs', 'forum']
target_word = '大立光'


def read_csv(data_path):
    # Read the file
    print('Reading data from {}...'.format(data_path))

    df = pd.read_csv(data_path, engine='python')
    df = df.dropna()
    print('==> length: {}'.format(len(df)))

    df['real_content'] = df['title'] + df['content']

    df = df[df['real_content'].str.contains(target_word)]
    print('==> length after filtering: {}'.format(len(df)))

    # preprocess(get rid of punctuation and english/number characters)
    for i, row in df.iterrows():
        df.at[i, 'real_content'] = ''.join(re.findall(r'([\u4e00-\u9fff])', row['real_content']))
    return df


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

    out_path = 'save/{}.csv'.format(save_name)
    print('==> Saving to {} ...'.format(out_path))
    df.to_csv(out_path, index=False)


def main():
    main_df = pd.DataFrame(columns=['post_time', 'real_content'])
    for c in categories:
        df = read_csv(data_path="./dataset/{}.csv".format(c))
        main_df = main_df.append(df[['post_time', 'real_content']], ignore_index=False)
    print('Total df length: {}'.format(len(main_df)))

    os.makedirs('save', exist_ok=True)
    save_path = 'save/main_article.csv'
    if os.path.exists(save_path):
        print('Loading cut data from {}...'.format(save_path))
        data = np.array(pd.read_csv(save_path)['real_content'])
    else:
        data = cut_news(np.array(main_df['real_content']))
        data = np.array([' '.join(article) for article in data])
        # saving
        print('Saving to {} ...\n'.format(save_path))
        main_df['real_content'] = data
        main_df.to_csv(save_path, index=False)

    calculate(data, save_name='main_tfidf')


if __name__ == "__main__":
    main()
