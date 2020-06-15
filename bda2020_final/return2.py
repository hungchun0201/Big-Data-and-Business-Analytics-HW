import os
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

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

    def get_order_info(self, keys, conditions, is_slave=False):
        """
        :param keys: items to observe from OrderData(Slave).csv (ex. ['TradesDateTime', 'Status'])
        :param is_slave: bool: use slave data or not
        :return: df with the 1st col being MemberID
        """
        if is_slave:
            csv_path = self.order_slave_csv_path
        else:
            csv_path = self.order_csv_path
        print('\treading data')
        order_df = pd.read_csv(csv_path, dtype={'ChannelDetail': 'str', 'PaymentType': 'str', 'ShippingType': 'str'})
        # for i in order_df:
        #     print(i)
        print('\tparsing conditions')
        for c in conditions:
            order_df = order_df[order_df[c[0]] == c[1]]
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
def get_time(tg, behavior_purchase):
    want = behavior_purchase[behavior_purchase['ti'] == tg]
    if want.empty:
        return 0, 0

    return list(want['uid'])[0], list(want['ht'])[0]
def check(mem_id, time, df, time_space):
    df_selected = df[df['uid'] == mem_id]
    t = time - time_space
    df_selected = df_selected[df_selected['ht'] >= t]
    df_selected = df_selected[df_selected['ht'] <= time]
    if df_selected.empty:
        return False
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--team', type=int)
    parser.add_argument('-r', '--read')
    parser.add_argument('-ts', '--time_space', type=int, default=43200)
    args = parser.parse_args()

    if not args.team:
        print('argument missing')
        exit()
    current_team = int(args.team)
    print('current reading team: ', current_team)
    print(args.read)
    print('current time space: ', args.time_space)
    
    print('reading RFM team')
    dataset = RawDataset()
    RFM_team = pd.read_csv('RFM_team.csv', dtype={'team': 'int'})
    if args.read == 'False':
        print('collecting member data')
        member_info = dataset.get_member_info(keys=['Gender', 'MemberCardLevel', 'IsEnableEmail', 'IsEnablePushNotification', 'IsEnableShortMessage', 'RegisterSourceTypeDef'])
        ###########
        current_member = list(RFM_team[RFM_team['team'] == current_team]['MemberID'])
        selected_member_info = member_info[member_info['MemberID'].isin(current_member)]
        selected_member_info.to_csv('member.csv', index=False)
        ###########

        print('collecting order data')
        order_info = dataset.get_order_info(keys=['TradesGroupCode',
                                                  'TotalDiscount',
                                                  'TotalPromotionDiscount',
                                                  'TotalCouponDiscount',
                                                  'TotalLoyaltyDiscount',
                                                  'ChannelType',
                                                  'ChannelDetail',
                                                  'PaymentType',
                                                  'ShippingType',
                                                  'Qty'
                                                  ],
                                            conditions=[('Status', 'Return')],
                                            is_slave=False
                                            )
        selected_member = list(selected_member_info['MemberID'])
        order_info = order_info[order_info['MemberID'].isin(selected_member)]
        # print(order_info)
        order_info.to_csv('order_cancel.csv', index=False)

        print('collecting behavior - purchase...')
        behavior_purchase = dataset.get_members_behavior_info(member_IDs=selected_member_info['MemberID'],
                                                              behavior_name='purchase', keys=['ht', 'ti']
                                                              )
        behavior_purchase.to_csv('behavor_purchase.csv', index=False)
        print()
        print('collecting behavior - addToCart...')
        behavior_addToCart = dataset.get_members_behavior_info(member_IDs=selected_member_info['MemberID'],
                                                              behavior_name='addToCart', keys=['ht']
                                                              )
        behavior_addToCart.to_csv('behavor_addToCart.csv', index=False)
        print()
        print('collecting behavior - trafficSource...')
        behavior_trafficSource = dataset.get_members_behavior_info(member_IDs=selected_member_info['MemberID'],
                                                              behavior_name='trafficSource', keys=['ht']
                                                              )
        behavior_trafficSource.to_csv('behavor_trafficSource.csv', index=False)

    else:
        print('reading stored data...')
        selected_member_info = pd.read_csv('member.csv')
        order_info = pd.read_csv('order_cancel.csv')
        behavior_purchase = pd.read_csv('behavor_purchase.csv')
        behavior_addToCart = pd.read_csv('behavor_addToCart.csv')
        behavior_trafficSource = pd.read_csv('behavor_trafficSource.csv')

    time_space = args.time_space
    ##### i #####
    print('team: Channel Type')
    channel_type = ['Pos', 'OfficialECom']

    for t in channel_type:
        print('calculating', t)
        TG = list(order_info[order_info['ChannelType'] == t]['TradesGroupCode'])
        group_cnt = [0, 0, 0]
        try:
            with tqdm(enumerate(TG), total = len(TG), ascii=True) as x:
                for i, tg in x:
                    mem_id, time = get_time(tg, behavior_purchase)
                    if time == 0:
                        continue
                    is_addToCart = check(mem_id, time, behavior_addToCart, time_space)
                    is_trafficSource = check(mem_id, time, behavior_trafficSource, time_space)
                    if is_addToCart:
                        # print('g1++', end='\r')
                        group_cnt[0] = group_cnt[0] + 1
                    elif is_trafficSource:
                        # print('g2++', end='\r')
                        group_cnt[1] = group_cnt[1] + 1
                    else:
                        # print('g3++', end='\r')
                        group_cnt[2] = group_cnt[2] + 1
                print()
                print(group_cnt)
        except KeyboardInterrupt:
            x.close()
            raise
        x.close()

    ##### ii ######
    print('team: Channel Detail')
    channel_type = ['DesktopOfficialWeb', 'MobileWeb', 'iOSApp', 'AndroidApp']

    for t in channel_type:
        print('calculating', t)
        TG = list(order_info[order_info['ChannelDetail'] == t]['TradesGroupCode'])
        group_cnt = [0, 0, 0]
        try:
            with tqdm(enumerate(TG), total = len(TG), ascii=True) as x:
                for i, tg in x:
                    mem_id, time = get_time(tg, behavior_purchase)
                    if time == 0:
                        continue
                    is_addToCart = check(mem_id, time, behavior_addToCart, time_space)
                    is_trafficSource = check(mem_id, time, behavior_trafficSource, time_space)
                    if is_addToCart:
                        # print('g1++', end='\r')
                        group_cnt[0] = group_cnt[0] + 1
                    elif is_trafficSource:
                        # print('g2++', end='\r')
                        group_cnt[1] = group_cnt[1] + 1
                    else:
                        # print('g3++', end='\r')
                        group_cnt[2] = group_cnt[2] + 1
                print()
                print(group_cnt)
        except KeyboardInterrupt:
            x.close()
            raise
        x.close()

    ##### iii ######
    print('team: Payment Type')
    channel_type = ['Cash', 'Family', 'SevenEleven', 'CreditCardOnce', 'CreditCardInstallment']

    for t in channel_type:
        print('calculating', t)
        TG = list(order_info[order_info['PaymentType'] == t]['TradesGroupCode'])
        group_cnt = [0, 0, 0]
        try:
            with tqdm(enumerate(TG), total = len(TG), ascii=True) as x:
                for i, tg in x:
                    mem_id, time = get_time(tg, behavior_purchase)
                    if time == 0:
                        continue
                    is_addToCart = check(mem_id, time, behavior_addToCart, time_space)
                    is_trafficSource = check(mem_id, time, behavior_trafficSource, time_space)
                    if is_addToCart:
                        # print('g1++', end='\r')
                        group_cnt[0] = group_cnt[0] + 1
                    elif is_trafficSource:
                        # print('g2++', end='\r')
                        group_cnt[1] = group_cnt[1] + 1
                    else:
                        # print('g3++', end='\r')
                        group_cnt[2] = group_cnt[2] + 1
                print()
                print(group_cnt)
        except KeyboardInterrupt:
            x.close()
            raise
        x.close()

    ##### iv ######
    print('team: Shipping Type')
    channel_type = ['Store', 'Family', 'FamilyPickup', 'SevenEleven', 'SevenElevenPickup', 'Home']

    for t in channel_type:
        print('calculating', t)
        TG = list(order_info[order_info['ShippingType'] == t]['TradesGroupCode'])
        group_cnt = [0, 0, 0]
        try:
            with tqdm(enumerate(TG), total = len(TG), ascii=True) as x:
                for i, tg in x:
                    mem_id, time = get_time(tg, behavior_purchase)
                    if time == 0:
                        continue
                    is_addToCart = check(mem_id, time, behavior_addToCart, time_space)
                    is_trafficSource = check(mem_id, time, behavior_trafficSource, time_space)
                    if is_addToCart:
                        # print('g1++', end='\r')
                        group_cnt[0] = group_cnt[0] + 1
                    elif is_trafficSource:
                        # print('g2++', end='\r')
                        group_cnt[1] = group_cnt[1] + 1
                    else:
                        # print('g3++', end='\r')
                        group_cnt[2] = group_cnt[2] + 1
                print()
                print(group_cnt)
        except KeyboardInterrupt:
            x.close()
            raise
        x.close()

    
