import random
import pandas as pd
import csv


def deal_team_name(team_name):
    for i in range(len(team_name)):
        if '\xa0' in team_name[i]:
            team_name[i] = team_name[i].replace('\xa0', '')
        if '(' in team_name[i]:
            while '(' in team_name[i]:
                start = team_name[i].find('(')
                end = team_name[i].find(')')
                team_name[i] = team_name[i][:start] + team_name[i][end+1:]
    return team_name



# 例外狀況 有比賽被取消，比分先暫定0:0 (手動加上比分)
# 3	Sep 14, 2013	2:00 PM	Sat	Fresno State		@	Colorado		Game Cancelled
# 1	Sep 5, 2015	7:30 PM	Sat	McNeese State		@	(14) Louisiana State		Cancelled due to weather


for year in range(2003, 2004):

    record = pd.read_csv(
        f'./spider/rank_data/{year}-{year+1}_Record.csv', sep="\t", header=None)

    if year < 2013:
        team_index = 3
        score_index = 4
    else:
        team_index = 4
        score_index = 5

    # team name
    # get the col4 and col7 of the record and combine them into a list
    header = record.columns[team_index]
    win_team_name = record[team_index]
    win_team_name = win_team_name.dropna()
    win_team_name = win_team_name.tolist()
    win_team_name = deal_team_name(win_team_name)

    header = record.columns[team_index+3]
    lose_team_name = record[team_index+3]
    lose_team_name = lose_team_name.dropna()
    lose_team_name = lose_team_name.tolist()
    lose_team_name = deal_team_name(lose_team_name)
    # print(len(set(lose_team_name)))

    # all_team_name = set(win_team_name + lose_team_name)
    # use list to store the team name and delete the duplicate team name
    all_team_name = list(set(win_team_name + lose_team_name))
    # print(all_team_name)
    
    # build a 2D list to store the lose-win matrix
    matrix = [[0.001 for i in range(len(all_team_name))]
              for j in range(len(all_team_name))]
    
    # 計算單支隊伍敗場數
    lose_count = {}
    for i in range(len(lose_team_name)):
        if lose_team_name[i] not in lose_count:
            lose_count[lose_team_name[i]] = 1
        else:
            lose_count[lose_team_name[i]] += 1
            
    # 對於輸過的隊伍，計算走到該隊伍的機率
    for i in range(len(win_team_name)):
        win_index = all_team_name.index(win_team_name[i])
        lose_index = all_team_name.index(lose_team_name[i])
        probability = (1-0.001*(len(all_team_name)-lose_count[lose_team_name[i]]))/lose_count[lose_team_name[i]]
        
        if (matrix[lose_index][win_index] != 0.001):
            matrix[lose_index][win_index] += probability
        else:
            matrix[lose_index][win_index] = probability
    
    # 檢查矩陣列和是否接近1
    for i in range(len(all_team_name)):
        sum = 0
        for j in range(len(all_team_name)):
            sum += matrix[i][j]
        print(all_team_name[i], sum)
        # if (sum < 0.9):
        #     print(all_team_name[i], sum)