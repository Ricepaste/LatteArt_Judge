import pandas as pd
import numpy as np


def data_load(year=2003, load='winner'):
    '''
    從資料中讀取出每場的勝者
    註:資料中有些名字有空格,有些名字有括號,有些名字有空格、括號、\xa0
    註:從2013起,勝者出現在第4個column,之前是第3個column

    year: 參數為讀取的年份
    load: 參數為'winner'時可用來讀取勝者，為'loser'時可用來讀取敗者

    ### TODO: 
        * 讀取每場的敗者
    '''
    FMT = None
    if load == 'winner':
        FMT = 3
        if year >= 2013:
            FMT = 4
    elif load == 'loser':
        FMT = 6
        if year >= 2013:
            FMT = 7
    # PATH
    path = f"./spider/rank_data/{year}-{year+1}_Record.csv"
    print(path)
    df = np.array(pd.read_csv(path, sep='\t'))

    winner = df[:, FMT]
    for i in range(winner.size):
        while '\xa0' in winner[i]:
            # print(winner[i])
            winner[i] = winner[i][1:]
        if '(' in winner[i]:
            # print(winner[i])
            winner[i] = winner[i][:winner[i].index('(')]\
                + winner[i][winner[i].index(')')+1:]
            # print(winner[i])
        if ' ' in winner[i]:
            # print(winner[i])
            winner[i] = winner[i][:winner[i].index(' ')]\
                + winner[i][winner[i].index(' ')+1:]
            # print(winner[i])

    return np.array(winner)


def main():
    for i in range(2003, 2023):
        x = data_load(i)
        print(x)
    pass


def debug():
    x = data_load(2003, load='loser')
    print(x)


if __name__ == '__main__':
    main()
    debug()
