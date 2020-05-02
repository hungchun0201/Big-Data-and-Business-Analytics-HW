import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime, timedelta
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

def read_data():
    df = pd.read_csv('../extract_vector/save/main_vector.csv')
    updown_data = np.load('../result.npy')
    def f(x): return '/'.join([i[1:] if i[0]
                               == '0' else i for i in x.split('/')])
    updown_dict = {f(k): int(v) for k, v in updown_data}
    train_labels = []
    train_data = []
    intense_date = [f(i.replace('-','/')) for i in np.load('../impdate.npy')]
    

    def find_nearest_date(date):
        # print('from '+date,end="")
        while(date not in updown_dict.keys()):
            date = f((datetime.strptime(date, "%Y/%m/%d") +
                      timedelta(days=1)).strftime("%Y/%m/%d"))
            if((datetime.strptime(date, "%Y/%m/%d")-datetime(2018, 12, 28)).total_seconds() > 0):
                date = '2018/12/28'
        # print(' to '+date)
        return date
    for _, row in df.iterrows():
        available_date = find_nearest_date(row['post_time'].split()[0])
        if(updown_dict[available_date]!=0):
            train_labels.append(updown_dict[available_date])
            train_data.append([int(i) for i in row['vector'][1:-1].split(', ')])
            if(available_date in intense_date):
                train_labels.append(updown_dict[available_date])
                train_data.append([int(i) for i in row['vector'][1:-1].split(', ')])

    return train_labels, train_data


def train():
    train_label, train_data = read_data()
    print(np.shape(np.array(train_data)))

    train_data = np.array(train_data)
    pca = PCA(n_components=100)
    pca.fit(train_data)
    train_data_pca = pca.transform(train_data)
    print("original shape:   ", train_data.shape)
    print("transformed shape:", train_data_pca.shape)

    train_data = train_data_pca

    train_data, test_data, train_label, test_label = train_test_split(
        train_data, train_label, test_size=0.2)

    knn = KNeighborsClassifier()
    knn.fit(train_data,train_label)
    predict_label = knn.predict(test_data)
    print(predict_label)
    print(test_label)

    all_label = list(zip(predict_label,test_label))
    np.save('all_label.npy',all_label)
    the_matrix = confusion_matrix(test_label,knn.predict(test_data))

    print(the_matrix)
    print('accuracy: ',np.trace(the_matrix)/np.sum(the_matrix))
    np.save('result_confusion_matrix.npy',the_matrix)


if __name__ == '__main__':
    train()
