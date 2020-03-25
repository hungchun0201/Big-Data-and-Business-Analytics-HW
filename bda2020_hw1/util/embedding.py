# functions for converting documents and sentences into embedding vectors
# or bag of word vectors
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# ===================================
def sklearn_tfidf(news, queries):
    '''
    Use sklearn.feature_extraction.text.TfidfVectorizer to convert
    documents or sentences into tfidf bag of word vectors.
    # Arguments:
        news(list of str): precut news
        queries(list of str): precut queries
    # Returns:
        matrix(scipy.sparse.csr.csr_matrix)
        feature_names(list of str)
    '''
    print('compute tfidf...')
    vectorizer = TfidfVectorizer()
    vectorizer.fit(news + queries)
    matrix = vectorizer.transform(news)
    feature_names = vectorizer.get_feature_names()
    return matrix, feature_names
