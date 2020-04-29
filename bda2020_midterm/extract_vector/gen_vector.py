import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

limit = 1000


def main():
    main_df = pd.read_csv('save/main_article.csv')
    tfidf_df = pd.read_csv('save/main_tfidf.csv')

    keywords = np.array(tfidf_df['token'][:limit])
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
