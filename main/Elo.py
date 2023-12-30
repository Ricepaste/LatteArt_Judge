import os
import pandas as pd
import numpy as np
import sys


def data_load(year):
    '''
    從資料中讀取出每場的勝者
    註:資料中有些名字有空格,有些名字有括號,有些名字有空格、括號、\xa0
    註:從2013起,勝者出現在第4個column,之前是第3個column
    TODO:
    * 讀取每場的敗者
    '''
    FMT = 3
    if year >= 2013:
        FMT = 4
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
    pass


if __name__ == '__main__':
    main()
    # debug()
