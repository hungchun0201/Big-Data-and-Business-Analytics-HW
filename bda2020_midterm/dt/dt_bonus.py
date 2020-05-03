import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime, timedelta
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn import tree
# from sklearn.cross_validation import train_test_split
start_ym = (2016, 1)
# start_ym = (2017, 5)
stop_ym = (2018, 9)
# stop_ym = (2016, 2)
against = 2

def f(x): return '/'.join([i[1:] if i[0]
                               == '0' else i for i in x.split('/')])

def load_data():
    df = pd.read_csv('../extract_vector/save/main_vector.csv')
    updown_data = np.load('../result.npy')
    updown_dict = {f(k): int(v) for k, v in updown_data}
    intense_date = [f(i.replace('-','/')) for i in np.load('../impdate.npy')]
    return df, updown_dict, intense_date
def process_data(df, updown_dict, intense_date, start_ym):
    train_labels = []
    train_data = []
    test_labels = []
    test_data = []

    def find_nearest_date(date):
        # print('from '+date,end="")
        while(date not in updown_dict.keys()):
            date = f((datetime.strptime(date, "%Y/%m/%d") +
                      timedelta(days=1)).strftime("%Y/%m/%d"))
            if((datetime.strptime(date, "%Y/%m/%d")-datetime(2018, 12, 28)).total_seconds() > 0):
                date = '2018/12/28'
        # print(' to '+date)
        return date
    def get_available_ym(start_ym):
        ym = []
        for i in range(4):
            if start_ym[1] + i <= 12:
                ym.append((start_ym[0], start_ym[1] + i))
            else:
                ym.append((start_ym[0] + 1, start_ym[1] + i - 12))
        return ym
    ym = get_available_ym(start_ym)
    date_idx = {}
    print(ym)
    for _, row in df.iterrows():
        available_date = find_nearest_date(row['post_time'].split()[0])
        year, month, _ = available_date.split('/')
        year = int(year)
        month = int(month)
        # print(available_date)
        if updown_dict[available_date]!=0:
            # print("available")
            if (year, month) in ym[0:3]:
                # print("\tstore data ", end = "")
                # print("train")
                train_labels.append(updown_dict[available_date])
                train_data.append([int(i) for i in row['vector'][1:-1].split(', ')])
                if(available_date in intense_date):
                    train_labels.append(updown_dict[available_date])
                    train_data.append([int(i) for i in row['vector'][1:-1].split(', ')])
            elif (year, month) == ym[-1]:
                # print("store data ", end = "")
                # print("test")
                if available_date not in date_idx:
                    date_idx[available_date] = len(test_data)
                test_labels.append(updown_dict[available_date])
                test_data.append([int(i) for i in row['vector'][1:-1].split(', ')])
                if(available_date in intense_date):
                    test_labels.append(updown_dict[available_date])
                    test_data.append([int(i) for i in row['vector'][1:-1].split(', ')])
        # elif updown_dict[available_date] == 0:
            # print("=====zero updown")
        elif (year, month) > ym[-1]:
            # print("=====late, break")
            break
        # else:
            # print((year, month), "?", ym[-1])
            # print("=====early or ")
    return train_labels, train_data, test_labels, test_data, date_idx


def train(train_label, train_data, test_label, test_data, random_state, current_ym):
    # print(np.shape(np.array(train_data)))

    train_data = np.array(train_data)
    pca = PCA(n_components=30, random_state=random_state)
    pca.fit(train_data)
    train_data_pca = pca.transform(train_data)
    # print("original shape:   ", train_data.shape)
    # print("transformed shape:", train_data_pca.shape)

    train_data = train_data_pca

    test_data = np.array(test_data)
    pca = PCA(n_components=30, random_state=random_state)
    # print(test_data)
    pca.fit(test_data)
    test_data_pca = pca.transform(test_data)

    test_data = test_data_pca


    dtc = tree.DecisionTreeClassifier()
    dtc.fit(train_data, train_label)
    predict_label = dtc.predict(test_data)
    # print(predict_label)
    # print(test_label)

    all_label = list(zip(predict_label,test_label))
    np.save('all_label.npy',all_label)
    the_matrix = confusion_matrix(test_label,dtc.predict(test_data))

    # print(the_matrix)
    acc = np.trace(the_matrix)/np.sum(the_matrix)
    print('{} accuracy: '.format(current_ym), acc)
    np.save('result_confusion_matrix_{}.npy'.format(current_ym),the_matrix)
    return predict_label

def increase_ym(ym):
    if ym[1] == 12:
        return (ym[0] + 1, 1)
    else:
        return (ym[0], ym[1] + 1)
def cal_result(predict_label, date_idx):
    idx = []
    for date in date_idx:
        idx.append(date_idx[date])
    idx.append(len(predict_label))
    predict = []
    for i, date in enumerate(date_idx):
        score = sum(predict_label[idx[i]:idx[i + 1]])
        # print(date, score)
        if score > against:
            predict.append((date, 1))
        elif score < -against:
            predict.append((date, -1))
        else:
            predict.append((date, 0))
    # print(predict)
    return predict
    
if __name__ == '__main__':
    df, updown_dict, intense_date = load_data()
    current_ym = start_ym
    total_predict = []
    while current_ym <= stop_ym:
        # print(current_ym)
        train_label, train_data, test_label, test_data, date_idx = process_data(df, updown_dict, intense_date, current_ym)
        # print(date_idx)
        if len(test_data) == 0:
            current_ym = increase_ym(current_ym)
            continue
        predict_label = train(train_label, train_data, test_label, test_data, 83, current_ym)
        total_predict.extend(cal_result(predict_label, date_idx))
        current_ym = increase_ym(current_ym)
    total_predict = np.array(total_predict)
    np.save('total_predict.npy', total_predict)
    