import os
import pandas as pd
import numpy as np
import sys


def data_load(year):
    # PATH
    path = f"./spider/rank_data/{year}-{year+1}_Record.csv"
    print(path)
    df = pd.read_csv(path, sep='\t')
    return np.array(df)


def main():
    pass


def debug():
    x = data_load(2003)
    winner = x[:, 3]
    for i in range(winner.size):
        while '\xa0' in winner[i]:
            print(winner[i])
            winner[i] = winner[i][1:]
        if '(' in winner[i]:
            print(winner[i])
            winner[i] = winner[i][:winner[i].index('(')]\
                + winner[i][winner[i].index(')')+1:]
            print(winner[i])
        if ' ' in winner[i]:
            print(winner[i])
            winner[i] = winner[i][:winner[i].index(' ')]\
                + winner[i][winner[i].index(' ')+1:]
            print(winner[i])
    # print(winner.size)
    # print(len(set(winner)))


if __name__ == '__main__':
    # main()
    debug()
