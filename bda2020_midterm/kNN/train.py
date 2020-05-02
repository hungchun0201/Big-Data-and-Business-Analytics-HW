import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime,timedelta




def read_data():


    df = pd.read_csv('../extract_vector/save/main_vector.csv')
    updown_data = np.load('../result.npy')
    f = lambda x:'/'.join([i[1:] if i[0]=='0' else i for i in x.split('/') ])
    updown_dict = {f(k):int(v) for k,v in updown_data}
    train_labels = []
    train_data = []
    def find_nearest_date(date):
        # print('from '+date,end="")
        while(date not in updown_dict.keys()):
            date = f((datetime.strptime(date, "%Y/%m/%d")+timedelta(days=1)).strftime("%Y/%m/%d"))
            if((datetime.strptime(date, "%Y/%m/%d")-datetime(2018,12,28)).total_seconds()>0):
                date = '2018/12/28'
        # print(' to '+date)
        return date
    for index, row in df.iterrows():
        
        available_date = find_nearest_date(row['post_time'].split()[0])
        train_labels.append(updown_dict[available_date])
        # print(available_date ,updown_dict[available_date])
        # train_labels.append()
        train_data.append(row['vector'])
    return train_labels,train_data



if __name__ == '__main__':
    read_data()

