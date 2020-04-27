import pandas as pd

def read_xlsx(file_name = './dataset/stock_data.xlsx'):
    print('reading stock data...')
    data = pd.read_excel(file_name, sheet_name = '上市2018', usecols = [0, 5])
    print(data['收盤價(元)'])
def main():
    read_xlsx()
if __name__ == "__main__":
    main()