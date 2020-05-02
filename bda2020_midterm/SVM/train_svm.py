import pandas as pd
import numpy as np

from datetime import datetime,timedelta
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def read_data():
    #Load data
    df_data = pd.read_csv('./extract_vector/save/main_vector.csv')
    date_lst = list(i[0:i.find(' ')] for i in df_data.iloc[1:,0])
    vector_lst = list(i[1:-1].split(", ") for i in df_data.iloc[1:, 1])


    #Load label
    check = np.load('./result.npy')

    #Tag label on every data
    label_lst = []

    check_date_lst = []
    for i in range(len(check)):
        oper = check[i][0].split('/')
        if oper[1][0] == '0':
            oper[1] = oper[1][1:]
        if oper[2][0] == '0':
            oper[2] = oper[2][1:]
        check[i][0] = '/'.join(oper)
        check_date_lst.append(check[i][0])

    f = lambda x:'/'.join([i[1:] if i[0]=='0' else i for i in x.split('/') ])
    def find_nearest_date(date):
        # print('from '+date,end="")
        while(date not in check_date_lst):
            date = f((datetime.strptime(date, "%Y/%m/%d")+timedelta(days=1)).strftime("%Y/%m/%d"))
            if((datetime.strptime(date, "%Y/%m/%d")-datetime(2018,12,28)).total_seconds()>0):
                date = '2018/12/28'
            # print(' to '+date)
        return date
        


    for idx1, i in enumerate(date_lst):
        for idx2, date_match in enumerate(check):
            if date_match[0] == i:
                label_lst.append(date_match[1])
        #print(len(label_lst), idx1)
        if len(label_lst) != idx1+1:
            for find in range(len(check)):
                if check[find][0] == find_nearest_date(i):
                    label_lst.append(check[find][1])
                    break
    print(label_lst) 

    return vector_lst, label_lst

def train(trainX, trainY):

    train_X, valid_X, train_Y, valid_Y = train_test_split(trainX, trainY, test_size=0.2)
    print(np.shape(np.array(train_X)))
    print(np.shape(np.array(train_Y)))

    model = SVC(gamma='auto')
    model.fit(train_X, train_Y)
    test_Y = model.predict(valid_X)
    print(test_Y)
    print(valid_Y)

    all_label = list(zip(test_Y, valid_Y))
    np.save('SVM_result.npy', all_label)

    matrix = confusion_matrix(valid_Y, test_Y)
    print(matrix)

    

if __name__ == '__main__':
    trainX, trainY = read_data()
    train(trainX, trainY)


