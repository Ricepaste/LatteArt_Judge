import pandas as pd
import numpy as np


def read_csv(filename: str = './form_rank/Football_Ranking.csv') -> pd.DataFrame:
    df = pd.read_csv(filename, delimiter=',', encoding='utf-8')
    return df[1:]


def extract_rank():
    idx = ["1st", "2nd", "3rd", "4th"]
    df = read_csv()
    all_rank = []
    for i in range(2019, 2024):
        year_rank = []
        for j in idx:
            temp = list(df[f"{i} - {i+1} [{j}]"])
            year_rank.append(temp)
        all_rank += list(np.array(year_rank).T)
    all_rank = np.array(all_rank)

    print(all_rank)


def main():
    extract_rank()


if __name__ == '__main__':
    main()
