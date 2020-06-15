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

    #For Failed/Canceled/Returned Orders
    def get_fcr_order_info(self, keys, is_slave=False):
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
        order_df = order_df[((order_df['Status']=='Fail') | (order_df['Status']=='Cancel') | (order_df['Status']=='Return'))]
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

def get_ratio(order):
    #####Return Ratio########
    # print('Return ratio:', target_comeback.shape[0] / target_fcr.shape[0])


    #####Comeback Ratio########
    canceled_order = order[((order['Status']=='Fail') | (order['Status']=='Cancel') | (order['Status']=='Return'))]
    '''
    Looking for each members latest Fail/Cancel/Return orders
    '''
    canceled_member= list(canceled_order['MemberID'])
    # list of members canceled or returned before (not repeated)
    new_cm = []
    for cm in canceled_member:
        if cm not in new_cm:
            new_cm.append(cm)
    '''
    Looking for each members latest comeback orders
    '''
    finished_order  = order[(order['Status']=='Finish') & (order['MemberID'].isin(new_cm))]
    finished_member= list(finished_order['MemberID'])
    new_fm = []
    for fm in finished_member:
        if fm not in new_fm:
            new_fm.append(fm)

    # print('Comeback ratio:', len(new_fm), '/', len(new_cm))
    if len(new_cm) == 0:
        return(len(new_fm), len(new_cm), "Nan")
    else:
        return(len(new_fm), len(new_cm), float(len(new_fm))/float(len(new_cm)))





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
    # parser.add_argument('-ts', '--time_space', type=int, default=43200)
    args = parser.parse_args()

    if not args.team:
        print('argument missing')
        exit()
    current_team = int(args.team)
    print('current reading team: ', current_team)
    print(args.read)
    # print('current time space: ', args.time_space)
    
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
                                                  'ChannelType',
                                                  'ChannelDetail',
                                                  'PaymentType',
                                                  'ShippingType',
                                                  'Qty',
                                                  'Status'
                                                  ],
                                            is_slave=False
                                            )
        selected_member = list(selected_member_info['MemberID'])
        order_info = order_info[order_info['MemberID'].isin(selected_member)]
        # print(order_info)
        order_info.to_csv('order_info.csv', index=False)


    else:
        print('reading stored data...')
        selected_member_info = pd.read_csv('member.csv')
        order_info = pd.read_csv('order_info.csv')


    
    
    
    ###### 3 #####
    print('Calculating comeback ratio for each feature...')
    print('Ratio Name: (Comeback, Total_Failed_Cancel_Return)')
    # ChannelType
    print('ChannelType')
    OfficialECom = order_info[order_info['ChannelType'] == 'OfficialECom']
    Pos = order_info[order_info['ChannelType'] == 'Pos']
    Mall = order_info[order_info['ChannelType'] == 'Mall']
    LocationWizard = order_info[order_info['ChannelType'] == 'LocationWizard']

    print('\tOfficialECom ratio: ', get_ratio(OfficialECom))
    print('\tPos ratio: ', get_ratio(Pos))
    print('\tMall ratio: ', get_ratio(Mall))
    print('\tLocationWizard ratio: ', get_ratio(LocationWizard))

    # ChannelDetail
    print('ChannelDetail')
    DesktopOfficialWeb = order_info[order_info['ChannelDetail'] == 'DesktopOfficialWeb']
    MobileWeb = order_info[order_info['ChannelDetail'] == 'MobileWeb']
    iOSApp = order_info[order_info['ChannelDetail'] == 'iOSApp']
    AndroidApp = order_info[order_info['ChannelDetail'] == 'AndroidApp']

    print('\tDesktopOfficialWeb ratio: ', get_ratio(DesktopOfficialWeb))
    print('\tMobileWeb ratio: ', get_ratio(MobileWeb))
    print('\tiOSApp ratio: ', get_ratio(iOSApp))
    print('\tAndroidApp ratio: ', get_ratio(AndroidApp))

    # PaymentType
    print('PaymentType')
    # print(order_info['PaymentType'])
    Cash = order_info[order_info['PaymentType'] == 'Cash']
    ATM = order_info[order_info['PaymentType'] == 'ATM']
    Family = order_info[order_info['PaymentType'] == 'Family']
    SevenEleven = order_info[order_info['PaymentType'] == 'SevenEleven']
    CreditCardOnce = order_info[order_info['PaymentType'] == 'CreditCardOnce']
    CreditCardInstallment = order_info[order_info['PaymentType'] == 'CreditCardInstallment']
    LinePay = order_info[order_info['PaymentType'] == 'LinePay']
    JKOPay = order_info[order_info['PaymentType'] == 'JKOPay']

    print('\tCash ratio: ', get_ratio(Cash))
    print('\tATM ratio: ', get_ratio(ATM))
    print('\tFamily ratio: ', get_ratio(Family))
    print('\tSevenEleven ratio: ', get_ratio(SevenEleven))
    print('\tCreditCardOnce ratio: ', get_ratio(CreditCardOnce))
    print('\tCreditCardInstallment ratio: ', get_ratio(CreditCardInstallment))
    print('\tLinePay ratio: ', get_ratio(LinePay))
    print('\tJKOPay ratio: ', get_ratio(JKOPay))

    # ShippingType 
    print('ShippingType')
    Store = order_info[order_info['ShippingType'] == 'Store']
    LocationPickup = order_info[order_info['ShippingType'] == 'LocationPickup']
    Family = order_info[order_info['ShippingType'] == 'Family']
    FamilyPickup = order_info[order_info['ShippingType'] == 'FamilyPickup']
    SevenEleven = order_info[order_info['ShippingType'] == 'SevenEleven']
    SevenElevenPickup = order_info[order_info['ShippingType'] == 'SevenElevenPickup']
    Home = order_info[order_info['ShippingType'] == 'Home']

    print('\tStore ratio: ', get_ratio(Store))
    print('\tLocationPickup: ', get_ratio(LocationPickup))
    print('\tFamily ratio: ', get_ratio(Family))
    print('\tFamilyPickup ratio: ', get_ratio(FamilyPickup))
    print('\tSevenEleven ratio: ', get_ratio(SevenEleven))
    print('\tSevenElevenPickup ratio: ', get_ratio(SevenElevenPickup))
    print('\tHome ratio: ', get_ratio(Home))

    # Qty
    # print('Qty')
    # Qty = np.array(order_info['Qty'])
    # print('\tQty mean: ', Qty.mean())

    # print(TG)
