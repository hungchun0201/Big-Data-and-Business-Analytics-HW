import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from raw_dataset import *

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

    def construct_RFM_model(self):
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
        for _, row in tqdm(data.iterrows(), total=len(data)):
            recency = (to_datetime(MAX_DATE) -
                       to_datetime(row['TradesDateTime'].split(' ')[0])).days
            old_recency = self.team_df.loc[row['MemberID']]['recency']
            if(old_recency == 0):
                self.team_df.loc[row['MemberID'], 'recency'] = recency
            else:
                self.team_df.loc[row['MemberID']
                                 ]['recency'] = recency if old_recency >= recency else old_recency
            self.team_df.loc[row['MemberID']]['frequency'] += 1

            self.team_df.loc[row['MemberID']]['monetary'] += row['TotalPrice']

        self.team_df.to_csv('RFM_team.csv')
        self.calculate_team()

    def calculate_team(self):
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
        boundaries = [718, 4, 9862]
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
        self.team_df.to_csv('RFM_team.csv',index_label=False)
        

    def get_user_team(self, user_id):
        if self.team_df.empty:
            self.read_RFM_model()
        return self.team_df.loc[user_id]['team']



if __name__ == '__main__':
    print(RFM_model().get_user_team('C1n1i3LhlKdws%2BchutttaaIKLQBlMjkT7tChnoB4oEU%3D'))
