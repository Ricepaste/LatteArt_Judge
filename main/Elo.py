import os
import pandas as pd
import numpy as np
import sys


def data_load(year):
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
