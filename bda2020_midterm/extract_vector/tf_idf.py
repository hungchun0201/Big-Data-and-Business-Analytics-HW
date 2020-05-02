import pandas as pd
import numpy as np
import os
import re
import jieba
from multiprocessing import Pool
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from tqdm import tqdm

# News_UpperBound = 10
categories = ['news', 'bbs', 'forum']
# target_word = '大立光'
target_word = ''


def read_csv(data_path):
    # Read the file
    print('Reading data from {}...'.format(data_path))

    df = pd.read_csv(data_path, engine='python')
    df = df.dropna()
    print('==> length: {}'.format(len(df)))

    df['real_content'] = df['title'] + df['content']

    if len(target_word) > 0:
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


def calculate_baseline(news_list):
    Quarantine_Threshold = 500
    Delete_Threshold = 0.05

    # For 2gram to 6 gram, each of them has a dict.
    news_dict = {k: v for k, v in [(str(i) + "gram_dict", dict()) for i in range(2, 7)]}

    # compute tf and df
    for ind, news in tqdm(enumerate(news_list), total=news_list.shape[0]):
        appeared_list = []  # store the appeare word to count the df
        for n_gram in range(2, 7):
            n_gram_dict = str(n_gram) + 'gram_dict'
            for i in range(len(news) - n_gram + 1):
                letter = news[i:i + n_gram]
                if letter not in news_dict[n_gram_dict]:  # first appear
                    news_dict[n_gram_dict][letter] = [1, 1]  # [tf,df]
                    appeared_list.append(letter)
                else:
                    if letter not in appeared_list:  # has appeared but not in current document
                        news_dict[n_gram_dict][letter][0] += 1
                        news_dict[n_gram_dict][letter][1] += 1
                        appeared_list.append(letter)
                    else:  # has appeared in current document more than one
                        news_dict[n_gram_dict][letter][0] += 1
        if (ind + 1) % Quarantine_Threshold == 0:
            # print('processed news number :', ind)
            for i in range(2, 7):  # Delete the word that is not common
                for name in list(news_dict[str(i) + 'gram_dict'].keys()):
                    if news_dict[str(i) + 'gram_dict'][name][0] < Delete_Threshold * Quarantine_Threshold:
                        del news_dict[str(i) + 'gram_dict'][name]
            for i in range(2, 6):  # Delete the redundant words
                master, slave = (i + 1, i)
                for name_master, tf_df_master in list(news_dict[str(master) + 'gram_dict'].items()):
                    has_detected = False
                    for name_slave, tf_df_slave in list(news_dict[str(slave) + 'gram_dict'].items()):
                        if name_slave in name_master and tf_df_master[1] == tf_df_slave[1]:
                            has_detected = True
                            del news_dict[str(slave) + 'gram_dict'][name_slave]
                        elif has_detected:
                            break
    for i in range(2, 7):  # Delete the word that is not common
        for name in list(news_dict[str(i) + 'gram_dict'].keys()):
            if news_dict[str(i) + 'gram_dict'][name][0] < Delete_Threshold * Quarantine_Threshold:
                del news_dict[str(i) + 'gram_dict'][name]
    # concatenate all dicts to one dict
    gram_dict_2, gram_dict_3, gram_dict_4, gram_dict_5, gram_dict_6 = [a for a in news_dict.values()]
    news_dict = {**gram_dict_2, **gram_dict_3, **gram_dict_4, **gram_dict_5, **gram_dict_6}

    # sort the result by tf
    sorted_result = list(
        map(lambda x: [x[0], x[1][0], x[1][1]], sorted(news_dict.items(), key=lambda x: x[1][0], reverse=True)))
    pd.DataFrame(sorted_result).to_csv('./save/all.csv', index=False, header=False)


def main():
    main_df = pd.DataFrame(columns=['post_time', 'real_content'])
    for c in categories:
        df = read_csv(data_path="./dataset/{}.csv".format(c))
        main_df = main_df.append(df[['post_time', 'real_content']], ignore_index=False)
    print('Total df length: {}'.format(len(main_df)))

    calculate_baseline(main_df['real_content'])

    # os.makedirs('save', exist_ok=True)
    # save_path = 'save/full_article.csv'
    # if os.path.exists(save_path):
    #     print('Loading cut data from {}...'.format(save_path))
    #     data = np.array(pd.read_csv(save_path)['real_content'])
    # else:
    #     data = cut_news(np.array(main_df['real_content']))
    #     data = np.array([' '.join(article) for article in data])
    #     # saving
    #     print('Saving to {} ...\n'.format(save_path))
    #     main_df['real_content'] = data
    #     main_df.to_csv(save_path, index=False)
    #
    # calculate(data, save_name='full_tfidf')


if __name__ == "__main__":
    main()
