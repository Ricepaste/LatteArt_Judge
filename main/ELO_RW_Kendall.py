import pandas as pd
import random as rd


def Kendall_tau(elo_array, ap_array):
    '''
    計算兩個排名的相似度
    註:Kendall tau係數的計算方式為:
    tau = (P-Q)/(P+Q)
    P: 有相同排名的對數
    Q: 沒有相同排名的對數
    '''

    # 現在計算係數的方法是比較字串大小(有誤)，非比較隊伍間的排序
    P = 0
    Q = 0
    for i in range(len(elo_array)):
        for j in range(i+1, len(elo_array)):
            if (elo_array[i] > elo_array[j] and ap_array[i] > ap_array[j]) or (elo_array[i] < elo_array[j] and ap_array[i] < ap_array[j]):
                P += 1
            else:
                Q += 1
    tau = (P-Q)/(P+Q)
    return tau


# TODO: 隊名問題: 有些隊伍名稱有空格，有些沒有，所以要先處理一下
for year in range(2003, 2023):
    # 做為比較的兩個排名是用elo_100和random walk的排名，因為用已經繼承的檔案會有多的隊伍出現
    elo_rank = pd.read_csv(
        f'./spider/rank_data/{year}-{year+1}_elo100_K32_shuffleTrue_stepLRFalse_inheritFalse.csv', sep='\t')
    random_walk_rank = pd.read_csv(
        f'./spider/rank_data/{year}-{year+1}_RandomWalk_STEP1000_RWTIMES300.csv', sep='\t')

    # turn the first column in the dataframe into a list
    elo_temp = elo_rank.values.tolist()
    elo_temp_header = elo_rank.columns.values.tolist()
    if ' ' in elo_temp_header[0]:
        elo_temp_header[0] = elo_temp_header[0].replace(' ', '')
    elo_array = [elo_temp_header[0]]

    for i in range(len(elo_temp)):
        ELO_temp = elo_temp[i][0].split(',')[0].rstrip(' ')
        if ' ' in ELO_temp:
            ELO_temp = ELO_temp.replace(' ', '')
        elo_array.append(ELO_temp)

    # print(elo_array)

    random_walk_temp = random_walk_rank.values.tolist()
    random_walk_temp_header = random_walk_rank.columns.values.tolist()
    if ' ' in random_walk_temp_header[0]:
        random_walk_temp_header[0] = random_walk_temp_header[0].replace(
            ' ', '')
    random_walk_array = [random_walk_temp_header[0].split(',')[0].rstrip(' ')]

    for i in range(len(random_walk_temp)):
        RW_temp = random_walk_temp[i][0].split(',')[0].rstrip(' ')
        if ' ' in RW_temp:
            RW_temp = RW_temp.replace(' ', '')
        random_walk_array.append(RW_temp)

    # print(elo_array)

    for i in range(len(elo_array)):
        try:
            random_walk_array[i] = elo_array.index(random_walk_array[i])+1
        except:
            random_walk_array[i] = rd.randint(0, 1000)
            pass

    for i in range(len(elo_array)):
        elo_array[i] = i+1

    print(f"第{year}-{year+1}年度: ", Kendall_tau(elo_array, random_walk_array))
