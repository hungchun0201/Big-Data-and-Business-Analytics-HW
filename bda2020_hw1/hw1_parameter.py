import pandas as pd
import numpy as np
import re

#Read the file
total_doc = 90507
df = pd.read_csv("./result/bonus/full.csv")
tf_all = np.array(list(df.iloc[:,1])) 
df_all = np.array(list(df.iloc[:,2])) 
news_dict = {k:v for v, k in enumerate(list(df.iloc[:,0]))}

catagories = ["信用卡", "匯率", "台灣", "台積電", "日本", "銀行"]
data_lst = []

for i in range(6):
    data_lst.append(pd.read_csv("./result/bonus/" + catagories[i] + ".csv"))

token_lst = []
tf_lst = []
df_lst = []

for i, csv in enumerate(data_lst):
    token_lst.append(list(np.array(list(csv.iloc[:, 0]))))
    tf_lst.append(list(np.array(list(csv.iloc[:, 1]))))
    df_lst.append(list(np.array(list(csv.iloc[:, 2]))))

#Calculate metric:

def Cal_MI(df_full, df, doc):
    MI = np.log10(df/np.dot(df_full, doc))
    return MI

def Cal_chi2(tf_full, df_full, tf, df, doc):
    tf_ex = np.dot(doc, tf_full/total_doc)
    df_ex = np.dot(doc, df_full/total_doc)
    tf_chi2 = np.power(tf - tf_ex, 2)/tf_ex 
    df_chi2 = np.power(df - df_ex, 2)/df_ex 
    if (tf < tf_ex):
        tf_chi2 = -tf_chi2
    if (df < df_ex):
        df_chi2 = -df_chi2

    return tf_chi2, df_chi2

def Cal_lift( df_full, df, doc): 
    lift = (df/doc) / (df_full / total_doc)
    return lift

MI_lst = []
tf_chi_lst = []
df_chi_lst = []
lift_lst = []

#Caculate 
for i in range(6):
    for ind, token in enumerate(token_lst[i]):
        if(news_dict.get(token) != None):
            dic = news_dict[token]
            tf_full = tf_all[dic]
            df_full = df_all[dic]
            tf = tf_lst[i][ind]
            df = df_lst[i][ind]
            doc = len(token_lst[i])
            MI_lst.append(Cal_MI(df_full, df, doc))
            tf_chi, df_chi = Cal_chi2(tf_full, df_full, tf, df, doc)
            tf_chi_lst.append(tf_chi)
            df_chi_lst.append(df_chi)
            lift_lst.append(Cal_lift(df_full, df, doc) )    
        else:
            MI_lst.append(None)
            tf_chi_lst.append(None)
            df_chi_lst.append(None)
            lift_lst.append(None)    
    
    df_mi = pd.DataFrame(list(zip(token_lst[i], MI_lst)), columns = ['token', 'MI'])
    df_tf_chi = pd.DataFrame(list(zip(token_lst[i], tf_chi_lst)), columns = ['token', 'tf_chi_square'])
    df_df_chi = pd.DataFrame(list(zip(token_lst[i], df_chi_lst)), columns = ['token', 'df_chi_square'])
    df_lift = pd.DataFrame(list(zip(token_lst[i], lift_lst)), columns = ['token', 'lift'])

    #Sorting
    df_mi = df_mi.sort_values(by = ['MI'], ascending=False)
    df_tf_chi = df_tf_chi.sort_values(by = ['tf_chi_square'], ascending=False)
    df_df_chi = df_df_chi.sort_values(by = ['df_chi_square'], ascending=False)
    df_lift = df_lift.sort_values(by = ['lift'], ascending=False)

    #Output
    output_fpath1 = '{}_MI.csv'.format(catagories[i])
    output_fpath2 = '{}_tf_chi.csv'.format(catagories[i])
    output_fpath3 = '{}_df_chi.csv'.format(catagories[i])
    output_fpath4 = '{}_lift.csv'.format(catagories[i])

    df_mi.to_csv(output_fpath1, index = False)
    df_tf_chi.to_csv(output_fpath2, index = False)
    df_df_chi.to_csv(output_fpath3, index = False)
    df_lift.to_csv(output_fpath4, index = False)

    #Clear lst
    MI_lst.clear()
    tf_chi_lst.clear()
    df_chi_lst.clear()
    lift_lst.clear()