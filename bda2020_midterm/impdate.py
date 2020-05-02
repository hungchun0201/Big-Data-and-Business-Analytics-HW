import pandas as pd
import numpy as np
import statistics
diff = 0
day = 2
# company = ['2330 台積電', '2317 鴻海', '2412 中華電', '6505 台塑化', '2454 聯發科', '1301 台塑', '3008 大立光', '2882 國泰金', '1303 南亞', '1326 台化']
company = '3008 大立光'
def read_xlsx(file_name = 'C:\\Users\\mikem\\Desktop\\BDA\\HW2\\3008.xlsx'):
    print('reading stock data...')
    data = pd.read_excel(file_name, sheet_name = '全部資料', usecols = [0, 1, 8])
    return data
def main():
    # data = read_xlsx('small_stock.xlsx')
    data = read_xlsx()
    company_data = data[data['證券代碼'] == company]['成交筆數(筆)']
    company_date = data[data['證券代碼'] == company]['年月日']
    num = np.array(company_data)
    date = np.array(company_date)
    num = list(num)
    date = list(date)
    date = [np.datetime_as_string(d, unit='D') for d in date]
    diff = statistics.mean(num) + statistics.pstdev(num)
    print(diff)
    num.reverse()
    date.reverse()
    result = []
    for i in range(len(num) - day):
        if (num[i + day]) > diff:
            imp = date[i]
            result.append(imp)
    result = np.array(result)
    # print(result)
    np.save('impdate.npy', result)

        
if __name__ == "__main__":
    main()
    

print(np.load('impdate.npy'))


