# functions for cutting sentences into tokens

__all__ = ['jieba_cut']

# ====================================

import os
from multiprocessing import Pool

import jieba

# ====================================

def _single_jieba_cut(sentence):
    seg_list = list(jieba.cut(sentence))
    sentence = ' '.join([token for token in seg_list if not token.isspace()])

    return sentence

def jieba_cut(sentences, aux_data_folder = './auxiliary_data/', n_jobs = 4):
    '''
    Use jeiba to cut sentences into tokens, remove all whitespaces,
    and use single whitespace to join them into new sentences.
    This function use multicore to accelerate.

    # Arguments:
        sentences(list of str): 
        aux_data_folder(str): The auxiliary data folder which contains "dict.txt.big"
        n_jobs(int): The number of core used to accelerate.

    # Returns:
        joined_tokens(list of str):
            A list contains the cut sentences.
            A cut sentence is a string composed of tokens
            joined by single whitespace.

    # Example:
    	>>> import utility
        >>> queries = utility.io.load_queries()
        >>> queries_cut = utility.cut_sentences.jieba_cut(queries)
        >>> len(queries)
        20
        >>> len(queries_cut)
        20
        >>> queries[0]
        '通姦在刑法上應該除罪化'
        >>> queries_cut[0]
        '通姦 在 刑法 上 應該 除罪 化'
    '''
    print('Cut sentences...')

    jieba.load_userdict(os.path.join(aux_data_folder, 'dict.txt.big'))
    with Pool(processes = n_jobs) as pool:
        joined_tokens = pool.map(_single_jieba_cut, sentences)

    return joined_tokens