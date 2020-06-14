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
        order_info = dataset.get_fcr_order_info(keys=['TradesGroupCode',
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
                                            is_slave=False
                                            )
        selected_member = list(selected_member_info['MemberID'])
        order_info = order_info[order_info['MemberID'].isin(selected_member)]
        # print(order_info)
        order_info.to_csv('order_fail_cancel_return.csv', index=False)

        print('collecting behavior - purchase...')
        exit
        behavior_purchase = dataset.get_members_behavior_info(member_IDs=selected_member_info['MemberID'],
                                                              behavior_name='purchase', keys=['ht', 'ti']
                                                              )
        behavior_purchase.to_csv('behavior_purchase_comeback.csv', index=False)
        print()
        print('collecting behavior - userRegisteration...')
        behavior_userRegisteration = dataset.get_members_behavior_info(member_IDs=selected_member_info['MemberID'],
                                                              behavior_name='userRegisteration', keys=['ht']
                                                              )
        behavior_userRegisteration.to_csv('behavior_userRegisteration_comeback.csv', index=False)

    else:
        print('reading stored data...')
        selected_member_info = pd.read_csv('member.csv')
        order_info = pd.read_csv('order_fail_cancel_return.csv')
        behavior_purchase = pd.read_csv('behavior_purchase_comeback.csv')
        behavior_userRegisteration = pd.read_csv('behavior_userRegisteration_comeback.csv')

    ##### 1 ######
    print('calculating comeback rate')


    ######### i ###########
    order = dataset.get_order_info(keys=['ShippingType', 'TradesDateTime', 'Status', 'TradesGroupCode'], is_slave = True)
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
    fd_dict = {}
    for fd in finished_order:
        ID = fd[0]
        time = fd[3]
        code = fd[5]
        if cm_dict[ID][0] < time:
            if ID in fd_dict:
                if fd_dict[ID][0] > time:
                    fd_dict[ID] = [time, code]
            else:
                fd_dict[ID] = [time, code]
    # latestcanceled_order = order[(order['Status']=='Finish') & (order['MemberID'].isin(new_cm)) & (order['ShippingType'] == 'SevenEleven')]
    # comeback_order       = order[(order['Status']=='Finish') & (order['MemberID'].isin(new_cm)) & (order['ShippingType'] == 'SevenEleven')]
    print("i : Comeback member sum vs Fail/Cancel/Return sum")
    print(len(new_cm), len(new_fm))

    
    '''
    ii
    '''
    time_space = args.time_space
    
    TG = list(order_info['TradesGroupCode'])
    group_cnt = [0, 0]
    for tg in TG:
        mem_id, time = get_time(tg, behavior_purchase)
        if time == 0:
            continue
        is_userRegisteration = check(mem_id, time, behavior_userRegisteration, time_space)
        if is_userRegisteration:
            print('g1++', end='\r')
            group_cnt[0] = group_cnt[0] + 1
        else:
            print('g2++', end='\r')
            group_cnt[1] = group_cnt[1] + 1
    print("ii : Register right before Purchase vs Others")
    print(group_cnt)

    ##### 2 ######
    target_member = list(order_info['MemberID'])
    target_member = selected_member_info[selected_member_info['MemberID'].isin(target_member)]
    level = np.array(target_member['MemberCardLevel'])
    print('mean level:', level.mean())

    IsEnableEmail = target_member[target_member['IsEnableEmail'] == True]
    IsEnablePushNotification = target_member[target_member['IsEnablePushNotification'] == True]
    IsEnableShortMessage = target_member[target_member['IsEnableShortMessage'] == True]

    IsEnableEmail_cnt = IsEnableEmail.shape[0]
    IsEnablePushNotification_cnt = IsEnablePushNotification.shape[0]
    IsEnableShortMessage_cnt = IsEnableShortMessage.shape[0]

    print('enable:')
    print('\tIsEnableEmail_cnt: ', IsEnableEmail_cnt)
    print('\tIsEnablePushNotification_cnt: ', IsEnablePushNotification_cnt)
    print('\tIsEnableShortMessage_cnt: ', IsEnableShortMessage_cnt)

    Store          = target_member[target_member['RegisterSourceTypeDef'] == 'Store']
    LocationWizard = target_member[target_member['RegisterSourceTypeDef'] == 'LocationWizard']
    iOSApp         = target_member[target_member['RegisterSourceTypeDef'] == 'iOSApp']
    AndroidApp     = target_member[target_member['RegisterSourceTypeDef'] == 'AndroidApp']
    Web            = target_member[target_member['RegisterSourceTypeDef'] == 'Web']
    NaN            = target_member[target_member['RegisterSourceTypeDef'] == 'NaN']

    Store_cnt          = Store.shape[0]
    LocationWizard_cnt = LocationWizard.shape[0]
    iOSApp_cnt         = iOSApp.shape[0]
    AndroidApp_cnt     = AndroidApp.shape[0]
    Web_cnt            = Web.shape[0]
    NaN_cnt            = target_member.shape[0] - Store_cnt - LocationWizard_cnt - iOSApp_cnt - AndroidApp_cnt - Web_cnt

    print('register type:')
    print('\tStore_cnt:', Store_cnt)
    print('\tLocationWizard_cnt:', LocationWizard_cnt)
    print('\tiOSApp_cnt:', iOSApp_cnt)
    print('\tAndroidApp_cnt_cnt:', AndroidApp_cnt)
    print('\tWeb_cnt:', Web_cnt)
    print('\tNaN_cnt:', NaN_cnt)
    
    
    ###### 3 #####
    if_discount = np.array(order_info[order_info['TotalDiscount'] != 0])
    discount = np.array(order_info['TotalDiscount'])
    use_ratio = if_discount.shape[0] / discount.shape[0]
    print('use discount ratio: ', use_ratio)
    print('use mean: ', discount.mean())

    TotalPromotionDiscount = np.array(order_info['TotalPromotionDiscount'])
    TotalCouponDiscount = np.array(order_info['TotalCouponDiscount'])
    TotalLoyaltyDiscount = np.array(order_info['TotalLoyaltyDiscount'])
    print('TotalPromotionDiscount mean: ', TotalPromotionDiscount.mean())
    print('TotalCouponDiscount mean: ', TotalCouponDiscount.mean())
    print('TotalLoyaltyDiscount mean: ', TotalLoyaltyDiscount.mean())

    # ChannelType
    print('ChannelType')
    OfficialECom = order_info[order_info['ChannelType'] == 'OfficialECom']
    Pos = order_info[order_info['ChannelType'] == 'Pos']
    Mall = order_info[order_info['ChannelType'] == 'Mall']
    LocationWizard = order_info[order_info['ChannelType'] == 'LocationWizard']

    print('\tOfficialECom ratio: ', OfficialECom.shape[0] / order_info.shape[0])
    print('\tPos ratio: ', Pos.shape[0] / order_info.shape[0])
    print('\tMall ratio: ', Mall.shape[0] / order_info.shape[0])
    print('\tLocationWizard ratio: ', LocationWizard.shape[0] / order_info.shape[0])

    # ChannelDetail
    print('ChannelDetail')
    DesktopOfficialWeb = order_info[order_info['ChannelDetail'] == 'DesktopOfficialWeb']
    MobileWeb = order_info[order_info['ChannelDetail'] == 'MobileWeb']
    iOSApp = order_info[order_info['ChannelDetail'] == 'iOSApp']
    AndroidApp = order_info[order_info['ChannelDetail'] == 'AndroidApp']

    print('\tDesktopOfficialWeb ratio: ', DesktopOfficialWeb.shape[0] / order_info.shape[0])
    print('\tMobileWeb ratio: ', MobileWeb.shape[0] / order_info.shape[0])
    print('\tiOSApp ratio: ', iOSApp.shape[0] / order_info.shape[0])
    print('\tAndroidApp ratio: ', AndroidApp.shape[0] / order_info.shape[0])

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

    print('\tCash ratio: ', Cash.shape[0] / order_info.shape[0])
    print('\tATM ratio: ', ATM.shape[0] / order_info.shape[0])
    print('\tFamily ratio: ', Family.shape[0] / order_info.shape[0])
    print('\tSevenEleven ratio: ', SevenEleven.shape[0] / order_info.shape[0])
    print('\tCreditCardOnce ratio: ', CreditCardOnce.shape[0] / order_info.shape[0])
    print('\tCreditCardInstallment ratio: ', CreditCardInstallment.shape[0] / order_info.shape[0])
    print('\tLinePay ratio: ', LinePay.shape[0] / order_info.shape[0])
    print('\tJKOPay ratio: ', JKOPay.shape[0] / order_info.shape[0])

    # ShippingType 
    print('ShippingType')
    Store = order_info[order_info['ShippingType'] == 'Store']
    LocationPickup = order_info[order_info['ShippingType'] == 'LocationPickup']
    Family = order_info[order_info['ShippingType'] == 'Family']
    FamilyPickup = order_info[order_info['ShippingType'] == 'FamilyPickup']
    SevenEleven = order_info[order_info['ShippingType'] == 'SevenEleven']
    SevenElevenPickup = order_info[order_info['ShippingType'] == 'SevenElevenPickup']
    Home = order_info[order_info['ShippingType'] == 'Home']

    print('\tStore ratio: ', Store.shape[0] / order_info.shape[0])
    print('\tLocationPickup: ', LocationPickup.shape[0] / order_info.shape[0])
    print('\tFamily ratio: ', Family.shape[0] / order_info.shape[0])
    print('\tFamilyPickup ratio: ', FamilyPickup.shape[0] / order_info.shape[0])
    print('\tSevenEleven ratio: ', SevenEleven.shape[0] / order_info.shape[0])
    print('\tSevenElevenPickup ratio: ', SevenElevenPickup.shape[0] / order_info.shape[0])
    print('\tHome ratio: ', Home.shape[0] / order_info.shape[0])

    # Qty
    print('Qty')
    Qty = np.array(order_info['Qty'])
    print('\tQty mean: ', Qty.mean())

    # print(TG)
