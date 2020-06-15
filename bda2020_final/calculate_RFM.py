import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import math
import random
from tqdm import tqdm
from raw_dataset import *
from mpl_toolkits import mplot3d

"""
Usage: 
```
from calculate_RFM import RFM_model
RFM_model().get_user_team(MemberID)
```
* The filename is default as RFM_team.csv
* If RFM_team.csv does not exist, it will calculate a new one, and may cause you lots of time.
* raw_dataset.py is needed
"""

class RFM_model:
    def __init__(self):
        self.team_df = pd.DataFrame()

    def read_RFM_model(self, countNew=False, dir_path='./'):
        filepath = os.path.join(dir_path, 'RFM_team.csv')
        print('Initializing...')
        if (os.path.isfile(filepath) and countNew==False):
            print("Using existing model")
            self.team_df = pd.read_csv(filepath,dtype={'MemberID':'str','team':'int','recency':'int','frequency':'int','monetary':'int'})
        else:
            print("Construct the RFM_model now ......")
            self.construct_RFM_model()

    def construct_RFM_model(self,filename='RFM_team.csv'):
        data = RawDataset().get_order_info(['TradesDateTime', 'TotalPrice'])
        # remove repeat member id
        Member_list = list(set(list(data['MemberID'])))
        # define the df
        columns = ['MemberID', 'recency', 'frequency', 'monetary', 'team']
        self.team_df = pd.DataFrame(
            0, columns=columns, index=np.arange(len(Member_list)))
        self.team_df.loc[:, 'MemberID'] = Member_list
        self.team_df = self.team_df.set_index('MemberID')
        print(self.team_df)
        for num, row in tqdm(data.iterrows(), total=len(data)):
            recency = (to_datetime(MAX_DATE) -
                       to_datetime(row['TradesDateTime'].split(' ')[0])).days
            old_recency = self.team_df.loc[row['MemberID']]['recency']
            if(old_recency == 0):
                self.team_df.loc[row['MemberID'], 'recency'] = recency
            else:
                self.team_df.loc[row['MemberID']
                                 ]['recency'] = recency if old_recency >= recency else old_recency
            self.team_df.loc[row['MemberID']]['frequency'] += 1
            if row['TotalPrice']>0:
                self.team_df.loc[row['MemberID']]['monetary'] += row['TotalPrice']
            # if(num>100):
            #     break
        

        self.team_df.to_csv(filename)
        self.team_df.reset_index(inplace=True)
        self.calculate_team(filename)

    def calculate_team(self,filename='RFM_team.csv'):
        '''
        Devide the customers.
        team 1 = low R, high frequency, high monetary => Best Customer
        team 2 = low R, high frequency, low monetary
        team 3 = low R, low frequency, high monetary
        team 4 = low R, low frequency, high monetary
        team 5 = high R, high frequency, high monetary
        team 6 = high R, high frequency, low monetary
        team 7 = high R, low frequency, high monetary
        team 8 = high R, low frequency, low monetary => Worst Customer
        '''
        boundaries = [719, 2, 2379]
        if self.team_df.empty:
            self.read_RFM_model()
        self.team_df.set_index('MemberID',inplace=True)
        
        for index, row in tqdm(self.team_df.iterrows(), total=len(self.team_df)):
            if row['recency']<=boundaries[0]:
                if(row['frequency']>boundaries[1]):
                    if(row['monetary']>boundaries[2]):
                        self.team_df.loc[index,'team'] = 1
                    else:
                        self.team_df.loc[index,'team'] = 2
                else:
                    if(row['monetary']>boundaries[2]):
                        self.team_df.loc[index,'team'] = 3
                    else:
                        self.team_df.loc[index,'team'] = 4
            else:
                if(row['frequency']>boundaries[1]):
                    if(row['monetary']>boundaries[2]):
                        self.team_df.loc[index,'team'] = 5
                    else:
                        self.team_df.loc[index,'team'] = 6
                else:
                    if(row['monetary']>boundaries[2]):
                        self.team_df.loc[index,'team'] = 7
                    else:
                        self.team_df.loc[index,'team'] = 8
        self.team_df.reset_index(inplace=True)
        self.team_df.to_csv(filename,index_label=False)
        

    def get_user_team(self, user_id):
        if self.team_df.empty:
            self.read_RFM_model()
        return self.team_df.loc[user_id]['team']

    def show_results(self):
        def log_tick_formatter(val, pos=None):
            return "{:.2e}".format(10**val)

        if self.team_df.empty:
            self.read_RFM_model()
        
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        # ax = fig.add_subplot(111, projection='3d_custom')
        # plt.ylim(1,10)
        temp = []
        for i in range(1000):
            val = random.choice(range(len(self.team_df)))
            if(self.team_df.loc[val]['monetary']==0 or self.team_df.loc[val]['recency']==0):
                continue
            temp.append(val)
        temp = self.team_df.loc[temp]
        z_points = list(map(lambda x:math.log10(x),list(temp['monetary'])))
        x_points = list(map(lambda x:math.log10(x),list(temp['recency'])))
       
        y_points = list(map(lambda x:math.log10(x),list(temp['frequency'])))
        ax.scatter3D(x_points, y_points, z_points, c=z_points, cmap='hsv')
        # ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
        # ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
        # ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
        ax.set_xlabel('recency(log)')
        ax.set_ylabel('frequency(log)')
        ax.set_zlabel('monetary(log)')


        plt.show()



if __name__ == '__main__':
    RFM_model().show_results()
