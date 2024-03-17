import pandas as pd

"""
TCU = TexasChristian
USC = SouthernCalifornia
LSU = LouisianaState
BYU = BrighamYoung
UCF = CentralFlorida
NCState = NorthCarolina State
SanJoseState = SanJose State
Hawai'i = Hawaii
"""

# build a dictionary to store the team name
team_name = {
    'TCU': 'TexasChristian',
    'USC': 'SouthernCalifornia',
    'LSU': 'LouisianaState',
    'BYU': 'BrighamYoung',
    'UCF': 'CentralFlorida',
    'NCState': 'NorthCarolina State',
    'SanJoseState': 'SanJose State',
    "Hawai'i": "Hawaii",
    "OleMiss": "Mississippi",
}


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


CORR = []

for year in range(2003, 2023):
    print(f"{year}-{year+1} 年度")
    try:
        elo_rank = pd.read_csv(
        f'./spider/rank_data/{year}-{year+1}_random_walk_matrix.csv', sep='\t')
        ap_rank = pd.read_csv(
            f'./spider/rank_data/{year}-{year+1}_ap-poll.csv', sep='\t')
        # get the header of the dataframe
        elo_header = elo_rank.columns.values.tolist()[0]
        elo_rank = elo_rank.values.tolist()
        
        for i in range(len(elo_rank)):
            elo_rank[i][0] = elo_rank[i][0].split(',')[0].rstrip('')
            # replace the space
            if ' ' in elo_rank[i][0]:
                elo_rank[i][0] = elo_rank[i][0].replace(' ', '')

        ap_header = ap_rank.columns.values.tolist()[1]
        ap_rank = ap_rank.values.tolist()
        # print(header)
        # print(elo_rank)

        elo_array = [elo_header.split(',')[0].rstrip('')]
        ap_array = [ap_header]

        # elo_team_number 用來查表
        elo_team_number = elo_array.copy()
        # print(elo_team_number)
        for i in range(len(elo_rank)):
            elo_team_number.append(elo_rank[i][0].split(',')[0].rstrip(''))
        # print(elo_team_number)

        # 第一個隊名不會被處理到，所以先處理
        if ap_array[0] in team_name:
            ap_array[0] = team_name[ap_array[0]]

        for i in range(23):
            elo_array.append(elo_rank[i][0])
            if ' ' in ap_rank[i][1]:
                ap_rank[i][1] = ap_rank[i][1].replace(' ', '')
            if ap_rank[i][1] in team_name:
                ap_rank[i][1] = team_name[ap_rank[i][1]]
            ap_array.append(ap_rank[i][1])

        # copy 作為用來比較的陣列
        copy = ap_array.copy()
        for i in range(len(ap_array)):
            ap_array[i] = i+1
        for i in range(len(copy)):
            if copy[i] in elo_team_number:
                elo_array[i] = elo_team_number.index(copy[i])+1
            else:
                elo_array[i] = 1000

        print("已讀取檔案")
    except:
        print("沒讀到檔案")

    print("Elo前24名次隊伍 : ", elo_array)
    print("Ap前24名次隊伍 : ", ap_array)
    print("Kendall tau係數 : ", Kendall_tau(elo_array, ap_array))

    CORR.append(Kendall_tau(elo_array, ap_array))


# print("2003~2023 Kendall's tau 係數 : ", CORR)

for i in range(len(CORR)):
    print(f"{2003+i}-{2004+i} 年度 : {CORR[i]}")
