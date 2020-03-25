# functions for input/output

__all__ = ['load_news',
           'load_queries',
           'load_training_data',
           'retrieval',
           'output_result']

# ===================================

import os
import csv
import json
import heapq
from multiprocessing import Pool

import numpy as np
import pandas as pd

# ===================================

def load_news(news_folder = './news_data_1/', aux_data_folder = './auxiliary_data/'):
    '''
    Load the contents of the news.

    # Arguments:
        news_folder(str): The folder which contains the downloaded news data.
        aux_data_folder(str): The auxiliary data folder.

    # Returns:
        news_contents(list of str): a list contains the content of all news in order

    # Example:
        >>> import utility
        >>> news_contents = utility.io.load_news()
        >>> len(news_contents)
        100000
        >>> news_contents[0]
        '新北市第二選區議員候選人陳明義，爭取連任失利，成了落選頭，他認為選前清潔工事件影片遭人變造、剪接，影響選
        務人員公平性，將部分屬於他的選票認定為無效票或其他候選人的票，上月向新北地院聲請全面驗票，並對吊車尾當選
        的陳文治提出當選無效之訴。\n新北地院表示，日前確實收到陳明義的相關聲請，最快在今日就會分案，並擇期開庭，
        傳喚陳明義進一步說明，再認定是否有驗票必要。\n新北市第二選區議員候選人陳明義，去年九合一大選前，因競選旗
        幟與布幕遭到清潔隊員拆除，他得知後怒嗆「讓你們掃個夠」，遭人錄下並上傳網路，引發社會輿論撻伐，更有上百名
        清潔工包圍陳明義競選總部。\n據了解，陳明義認為，該影片遭人剪接、變造，隨媒體強力放送，不僅重創他的形象，
        更影響選務人員的公平性，造成不少投給他的有效選票，遭選務人員認定為無效票，或是他人選票。\n陳明義指出，選
        務人員的不公，已構成《公職人員選舉罷免法》中的「當選票數不實，足認有影響選舉結果之虞」，向新北地院聲請對
        第二選區包括新莊、五股、林口與泰山共387個投開票所全面驗票。\n陳明義表示，上月委託2位律師研究，針對陳文治
        、張晉婷其中1人提出當選無效告訴，然因張已被檢方提出當選無效，才轉而對陳提出告訴；對於驗票與否，陳明義說，
        目前仍無法確認，全數細節仍待8日自上海返台後與律師討論。\n對此，市議員陳文治則說，「真金不怕火煉」，若要重
        新驗票就每個票匭逐一驗票，的確，票數贏的少，僅有15票之差，但也不會影響陳明義落選事實，何況，提出當選無效
        告訴、驗票與否，也是他的權利。\n()'
    '''
    print('Load news...')

    nc_df = pd.read_csv(os.path.join(news_folder, 'NC_1.csv'))
    news_urls = list(nc_df['News_URL'])
    with open(os.path.join(aux_data_folder, 'url2content.json'), 'r') as fin:
        url2content_dict = json.load(fin)
    news_contents = [url2content_dict[url] for url in news_urls]

    return news_contents

def load_queries(news_folder = './news_data_1/'):
    '''
    Load the queries.

    # Arguments:
        news_folder(str): The folder which contains the downloaded news data.

    # Returns:
        queries(list of str): a list contains all the queries in order

    # Example:
        >>> import utility
        >>> queries = utility.io.load_queries()
        >>> len(queries)
        20
        >>> queries[0]
        '通姦在刑法上應該除罪化'
    '''
    print('Load queries...')

    qs_df = pd.read_csv(os.path.join(news_folder, 'QS_1.csv'))
    queries = list(qs_df['Query'])

    return queries

def load_training_data(news_folder = './news_data_1/'):
    '''
    Load the training data.

    # Arguments:
        news_folder(str): The folder which contains the downloaded news data.

    # Returns:
        query(list of str): a list contains the training queries in order
        news_index(list of int): a list contains the "list index" that the 
                                 training news_index correspond to in order.
                                 For example, "news_000001" correspond to 0,
                                 "news_000002" correspond to 1, etc.
        relevance(list of int): a list contains the training relevance in order

    # Example:
        >>> import utility
        >>> query, news_index, relevance = utility.io.load_training_data()
        >>> len(query)
        4742
        >>> len(news_index)
        4742
        >>> len(relevance)
        4742
        >>> query[0]
        '支持陳前總統保外就醫'
        >>> news_index[0]
        64208
        >>> relevance[0]
        2
    '''
    print('Load training data...')

    td_df = pd.read_csv(os.path.join(news_folder, 'TD.csv'))
    query = list(td_df['Query'])
    news_index = [(int(x[-6:]) - 1) for x in td_df['News_Index']]
    relevance = list(td_df['Relevance'])

    return query, news_index, relevance

def retrieval(score_matrix, rank = 300):
    '''
    Input a score_matrix, retrieve those indices with highest scores.

    # Arguments:
        score_matrix(list of 1d score_array or 2d score_matrix): 
            The score matrix with shape = (number of queries, number of news).
        rank: How many ranks to be retrieved.

    # Returns:
        result(2d array of int): The retrieval result.

    # Example:
        >>> import utility
        >>> # assume there are 5 news, 2 queries in total
        >>> # assume the score_matrix looks like the following
        >>> score_matrix
        [[0.34, 0.54, 0.61, 0.32, 0.12],
         [0.18, 0.26, 1.45, 2.34, 1.11]]
        >>> # assume rank = 3
        >>> result = utility.io.retrieval(score_matrix, rank = 3)
        >>> result
        [[2, 1, 0],
         [3, 2, 4]]
    '''
    result = list()
    for score_array in score_matrix:
        pairs = [(-score, i) for i, score in enumerate(score_array)]
        heapq.heapify(pairs)
        single_result = [heapq.heappop(pairs)[1] for _ in range(rank)]
        result.append(single_result)
    result = np.array(result)

    return result

def output_result(result, output_path):
    '''
    Output the retrieval result into the specified file.

    # Arguments:
        result(2d array of int): The retrieval result.
                                 (has the same format as the output of utility.retrieval)
        output_path(str): The complete path to the output csv file.

    # Returns:
        None

    # Example:
        >>> # A result with number of queries = 3 and rank = 5
        >>> result
        array([[3, 4, 7, 3, 2],
               [0, 1, 2, 5, 6],
               [2, 3, 5, 7, 9]])
        >>> # The number in result represents the index of the news "in list",
        >>> # so 0 represents "news_000001", 1 represents "news_000002", etc.
        >>>
        >>> output_result(result)
        >>> # The above will write the following matrix into the specified csv file.
        [['news_000004', 'news_000005', 'news_000008', 'news_000004', 'news_000003'],
         ['news_000001', 'news_000002', 'news_000003', 'news_000006', 'news_000007'],
         ['news_000003', 'news_000004', 'news_000006', 'news_000008', 'news_000010']]
    '''
    print('Output result...')

    result = list(map(lambda single_result: ['news_%06d' % (x + 1) for x in single_result], result))
    result = np.array(result)
    nb_queries, rank = result.shape
    with open(output_path, 'w', newline = '') as fout:
        output = csv.writer(fout)
        output.writerow(['Query_Index'] + ['Rank_%03d' % (i + 1) for i in range(rank)])
        query_idx = np.array(['q_%02d' % (i + 1) for i in range(nb_queries)])[:, np.newaxis]
        output.writerows(np.concatenate((query_idx, result), axis = 1))