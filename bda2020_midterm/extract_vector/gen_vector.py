import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

limit = 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', '-b', action='store_true')
    args = parser.parse_args()

    # tfidf_df = pd.read_csv('save/main_tfidf.csv')
    # keywords = np.array(tfidf_df['token'][:limit])
    if args.baseline:
        print("Running baseline...")
        main_df = pd.read_csv('save/main_article_baseline.csv')
        pos_keywords = np.array(pd.read_csv('save/main_pos_tfidf_baseline.csv')['token'])
        neg_keywords = np.array(pd.read_csv('save/main_neg_tfidf_baseline.csv')['token'])
    else:
        print("Running improved...")
        main_df = pd.read_csv('save/main_article.csv')
        pos_keywords = np.array(pd.read_csv('save/main_pos_tfidf.csv')['token'])
        neg_keywords = np.array(pd.read_csv('save/main_neg_tfidf.csv')['token'])

    pos_keywords = np.setdiff1d(pos_keywords, neg_keywords, assume_unique=True)
    neg_keywords = np.setdiff1d(neg_keywords, pos_keywords, assume_unique=True)
    keywords = np.concatenate((pos_keywords[:limit // 2], neg_keywords[:limit // 2]))
    with open("save/keywords.txt", 'w') as f_save:
        f_save.write('\n'.join(map(str, keywords)))

    vec = CountVectorizer(vocabulary=keywords)

    real_content = np.array(main_df['real_content'])
    count_matrix = vec.transform(real_content).toarray()
    # vec_df = pd.DataFrame(columns=['post_time', 'vector'])
    vec_df = pd.DataFrame()
    vec_df['post_time'] = main_df['post_time']
    vec_df['vector'] = count_matrix.tolist()

    vec_df.to_csv('save/main_vector.csv', index=False)


if __name__ == '__main__':
    main()
