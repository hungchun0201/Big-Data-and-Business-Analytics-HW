import pandas as pd
import numpy as np
import os
import re

categories = np.load("./category.npy", allow_pickle=True)
category_names = ['銀行', '信用卡', '匯率', '台積電', '台灣', '日本']

Quarantine_Threshold = 500
Delete_Threshold = 0.05
# News_UpperBound = 100

#Read the file
df = pd.read_csv("./data/all.csv")
title_arr = np.array(list(df.loc[:,'標題']))
content_arr = np.array(list(df.loc[:,'內容']))
news_list = list(map(lambda x,y:x+y,title_arr,content_arr))

# news_list = news_list[0:News_UpperBound]
# For 2gram to 6grame, each of them has a dict.
news_dict = {k:v for k,v in [(str(i)+"gram_dict",dict()) for i in range(2,7)]}

#preprocess(get rid of punctuation and english/number characters)
for i in range(len(news_list)):
    news_list[i] = ''.join(re.findall(u'([\u4e00-\u9fff])',news_list[i]))
print(list(map(lambda x:len(x),categories)))
for category_id,category_arr in enumerate(categories):
    news_dict = {k:v for k,v in [(str(i)+"gram_dict",dict()) for i in range(2,7)]}

    for ind,val in enumerate(category_arr):
        news = news_list[val-1] 
        appeared_list = [] #store the appeare word to count the df
        for n_gram in range(2,7):
            n_gram_dict = str(n_gram)+'gram_dict'
            for i in range(len(news)-n_gram+1):
                letter = news[i:i+n_gram]
                if(letter not in news_dict[n_gram_dict]):#first appear
                    news_dict[n_gram_dict][letter] = [1,1] #[tf,df]
                    appeared_list.append(letter)
                else:
                    if(letter not in appeared_list):#has appeared but not in current document
                        news_dict[n_gram_dict][letter][0]+=1
                        news_dict[n_gram_dict][letter][1]+=1
                        appeared_list.append(letter)
                    else:#has appeared in current document more than one
                        news_dict[n_gram_dict][letter][0]+=1
        if((ind+1)%Quarantine_Threshold==0):
            print('processed news number :',ind)
            for i in range(2,7):#Delete the word that is not common
                for name in list(news_dict[str(i)+'gram_dict'].keys()):
                    if(news_dict[str(i)+'gram_dict'][name][0]<Delete_Threshold*Quarantine_Threshold):
                        del news_dict[str(i)+'gram_dict'][name]
            for i in range(2,6):#Delete the redundant words
                master,slave = (i+1,i)
                for name_master,tf_df_master in list(news_dict[str(master)+'gram_dict'].items()):
                    has_detected = False
                    for name_slave,tf_df_slave in list(news_dict[str(slave)+'gram_dict'].items()):
                        if(name_slave in name_master and tf_df_master[1]==tf_df_slave[1]):
                            has_detected = True
                            del news_dict[str(slave)+'gram_dict'][name_slave]
                        elif(has_detected == True):
                            break
    


    #concatenate all dicts to one dict
    gram_dict_2,gram_dict_3,gram_dict_4,gram_dict_5,gram_dict_6 = [a for a in news_dict.values()]
    news_dict = {**gram_dict_2,**gram_dict_3,**gram_dict_4,**gram_dict_5,**gram_dict_6}

    #sort the result by tf

    sorted_result = list(map(lambda x:[x[0],x[1][0],x[1][1]],sorted(news_dict.items(), key=lambda x: x[1][0],reverse=True)))
    pd.DataFrame(sorted_result).to_csv('./result/{category}.csv'.format(category=category_names[category_id]),index=False,header=False)
    # print(sorted_result[0:10])




