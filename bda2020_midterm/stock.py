import pandas as pd
import numpy as np
import statistics
sigma = 0
day = 2
# company = ['2330 台積電', '2317 鴻海', '2412 中華電', '6505 台塑化', '2454 聯發科', '1301 台塑', '3008 大立光', '2882 國泰金', '1303 南亞', '1326 台化']
company = '3008 大立光'
def read_xlsx(sheet_name,file_name = './dataset/stock_data.xlsx'):
    print('reading stock data of {sheet_name}...'.format(sheet_name=sheet_name))
    data = pd.read_excel(file_name, sheet_name = sheet_name, usecols = [0, 1, 5])
    return data
def main():
    # data = read_xlsx('small_stock.xlsx')
    sheets = ['上市2018','上市2017','上市2016']
    result = []
    for sheet_name in sheets:
        data = read_xlsx(sheet_name)
        company_data = data[data['證券代碼'] == company]['收盤價(元)']
        company_date = data[data['證券代碼'] == company]['年月日']
        price = np.array(company_data)
        date = np.array(company_date)
        price = list(price)
        date = list(date)
        date = [np.datetime_as_string(d, unit='D') for d in date]
        date = [d.replace('-','/') for d in date]
        price.reverse()
        date.reverse()
        updown = []
        for i in range(len(price) - day):
            if (price[i + day] - price[i]) / price[i] > sigma:
                updown.append(1)
            elif (price[i + day] - price[i]) / price[i] < -sigma:
                updown.append(-1)
            else:
                updown.append(0)
        for i in range(day):
            updown.append(0)
        
        for i in range(len(updown)):
            result.append((date[i], updown[i]))
        print(result)
        
    result = np.array(result)
    np.save('result_zero_sigma.npy', result)

        
if __name__ == "__main__":
    main()