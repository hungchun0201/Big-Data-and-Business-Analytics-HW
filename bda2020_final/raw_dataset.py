import os
import datetime
import pandas as pd
from tqdm import tqdm

MIN_DATE = '2018-06-26'
MAX_DATE = '2020-04-30'


def to_datetime(date_str):
    return datetime.datetime.strptime(date_str, '%Y-%m-%d')


def to_date_str(date_time):
    return date_time.strftime('%Y-%m-%d')


class RawDataset:
    def __init__(self, dir_path='./91ForNTUDataSet'):
        self.member_csv_path = os.path.join(dir_path, 'MemberData.csv')
        self.order_csv_path = os.path.join(dir_path, 'OrderData.csv')
        self.order_slave_csv_path = os.path.join(dir_path, 'OrderSlaveData.csv')

        self.behavior_dir_path = os.path.join(dir_path, '123_new')

    def get_member_info(self, keys):
        """
        obtain the info according to the keys
        :param keys: items to observe from MemberData.csv (ex. ['RegisterDateTime', 'Birthday'])
        :return: df with the 1st col being MemberID
        """
        member_df = pd.read_csv(self.member_csv_path)
        return member_df[['MemberID', *keys]]

    def get_order_info(self, keys, is_slave=False):
        """
        :param keys: items to observe from OrderData(Slave).csv (ex. ['TradesDateTime', 'Status'])
        :param is_slave: bool: use slave data or not
        :return: df with the 1st col being MemberID
        """
        if is_slave:
            csv_path = self.order_csv_path
        else:
            csv_path = self.order_slave_csv_path
        order_df = pd.read_csv(csv_path)
        return order_df[['MemberID', *keys]]

    def get_members_behavior_info(self, behavior_name, keys, member_IDs, start_date=MIN_DATE, end_date=MAX_DATE):
        """
        :param behavior_name: which behavior to observe (filename)
        :param keys: items to observe(ex. ['did', 'bh'])
        :param member_IDs: a list of member IDs
        :param start_date: ex. '2019-06-06'
        :param end_date: ex. '2020-04-03'
        :return: df of target_uid, containing uid, keys, and date information
        """
        start_date = max(to_datetime(start_date), to_datetime(MIN_DATE))
        end_date = min(to_datetime(end_date), to_datetime(MAX_DATE))

        pbar = tqdm(desc='Processing behavior data', total=(end_date - start_date).days, ncols=80, leave=False)

        cur_date = start_date
        bh_df = pd.DataFrame(columns=['uid', *keys, 'date'])
        while cur_date <= end_date:
            behavior_csv_path = os.path.join(self.behavior_dir_path,
                                             behavior_name + '_' + to_date_str(cur_date) + '.csv')

            bh_date_df = pd.read_csv(behavior_csv_path)[['uid', *keys]]
            bh_date_df = bh_date_df[bh_date_df['uid'].isin(member_IDs)]
            bh_date_df['date'] = to_date_str(cur_date)

            bh_df = bh_df.append(bh_date_df)
            cur_date += datetime.timedelta(days=1)

            pbar.update()

        return bh_df


if __name__ == '__main__':
    dataset = RawDataset()

    # getting the member data w/ specific keys
    member_info = dataset.get_member_info(keys=['Gender', 'MemberCardLevel'])

    # filter the member data w/ conditions
    selected_member_info = member_info[(member_info['Gender'] == 'Female') & (member_info['MemberCardLevel'] > 30)]

    # obtain the behavior data using the selected member IDs
    behavior = dataset.get_members_behavior_info(member_IDs=selected_member_info['MemberID'],
                                                 behavior_name='activityPageView', keys=['dc', 'ul', 'geoid']
                                                 )
    behavior.to_csv('test.csv', index=False)
