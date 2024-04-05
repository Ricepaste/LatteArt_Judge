import pandas as pd
import numpy as np
import csv
import random as rd
import matplotlib.pyplot as plt

ARRAY_FLASH = []
ARRAY_AVERAGE_TAU = []
FLASH = 0.0001
ADD_F = 0.00005
CON = 0.0001

class RandomWalkMatrix:
    def __init__(self, FLASH, CONVERGENCE):
        self.FLASH = FLASH
        self.CONVERGENCE = CONVERGENCE
        self.generate_matrix()
        

    def deal_team_name(self, team_name):
        for i in range(len(team_name)):
            # 去除隊名中的空白和括號中的內容(含括號)
            if '\xa0' in team_name[i]:
                team_name[i] = team_name[i].replace('\xa0', '')
            if '(' in team_name[i]:
                while '(' in team_name[i]:
                    start = team_name[i].find('(')
                    end = team_name[i].find(')')
                    team_name[i] = team_name[i][:start] + team_name[i][end+1:]
        return team_name

    def generate_matrix(self):
        FLASH = self.FLASH
        CONVERGENCE = self.CONVERGENCE
        
        for year in range(2003, 2023):
        # for year in range(2023, 2024):

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
            win_team_name = self.deal_team_name(win_team_name)

            header = record.columns[team_index+3]
            lose_team_name = record[team_index+3]
            lose_team_name = lose_team_name.dropna()
            lose_team_name = lose_team_name.tolist()
            lose_team_name = self.deal_team_name(lose_team_name)
            # print(len(set(lose_team_name)))

            # use list to store the team name and delete the duplicate team name
            all_team_name = sorted(list(set(win_team_name + lose_team_name)))
            # print(all_team_name)

            # build a 2D list to store the lose-win matrix
            matrix = [[0.0 for i in range(len(all_team_name))]
                    for j in range(len(all_team_name))]

            # 計算單支隊伍敗場數
            lose_count = {}
            for i in range(len(lose_team_name)):
                if lose_team_name[i] not in lose_count:
                    lose_count[lose_team_name[i]] = 1
                else:
                    lose_count[lose_team_name[i]] += 1

            # # 對於輸過的隊伍，計算走到該隊伍的機率
            # for i in range(len(win_team_name)):
            #     win_index = all_team_name.index(win_team_name[i])
            #     lose_index = all_team_name.index(lose_team_name[i])
            #     probability = (1-0.001*(len(all_team_name) -
            #                    lose_count[lose_team_name[i]]))/lose_count[lose_team_name[i]]

            #     if (matrix[lose_index][win_index] != 0.001):
            #         matrix[lose_index][win_index] += probability
            #     else:
            #         matrix[lose_index][win_index] = probability
            for i in range(len(win_team_name)):
                matrix[all_team_name.index(lose_team_name[i])
                    ][all_team_name.index(win_team_name[i])] += 1

            # 對矩陣每一列做MinMaxScaler
            for i in range(len(matrix)):
                row_max = max(matrix[i])
                row_min = min(matrix[i])
                for j in range(len(matrix)):
                    try:
                        matrix[i][j] = (matrix[i][j] - row_min) / (row_max - row_min)
                    except:
                        continue

            # 使矩陣每列總和為1
            try:
                for i in range(len(matrix)):
                    row_sum = sum(matrix[i])
                    try:
                        matrix[i] = [x / row_sum for x in matrix[i]]
                    except:
                        continue
            except Exception as e:
                print(matrix[i])
                print(e)
                

            # 每個元素加上閃現機率
            for i in range(len(matrix)):
                for j in range(len(matrix)):
                    if matrix[i][j] == 0 and i != j:
                        matrix[i][j] += FLASH
                        

            # 使矩陣每列總和為1
            for i in range(len(matrix)):
                row_sum = sum(matrix[i])
                try:
                    matrix[i] = [x / row_sum for x in matrix[i]]
                except:
                    continue

            # ----------------------------------------------
            '''
            TAG: print matrix
            '''
            # print("\t", end='')
            # for i in range(len(all_team_name)):
            #     print(all_team_name[i], end='\t')
            # print()

            # for i in range(len(all_team_name)):
            #     print(all_team_name[i], end='\t')
            #     for j in range(len(all_team_name)):
            #         print(f"{matrix[i][j]:.5f}", end='\t')
            #     print()
            # 檢查矩陣列和是否接近1
            # for i in range(len(all_team_name)):
            #     sum = 0
            #     for j in range(len(all_team_name)):
            #         sum += matrix[i][j]
            #     print(all_team_name[i], sum)
                # if (sum < 0.9):
                #     print(all_team_name[i], sum)
            # ----------------------------------------------

            # turn matrix to numpy array
            matrix = np.array(matrix)
            # print(matrix)

            # 給定初始狀態，求穩定態
            state = np.array([1/len(all_team_name) for i in range(len(all_team_name))])
            state = state.dot(matrix)
            while (np.linalg.norm(state - state.dot(matrix)) > CONVERGENCE):
                state = state.dot(matrix)
            # print(state)

            # 印出對應隊伍，前十名
            state = state.tolist()
            state = list(zip(all_team_name, state))
            state.sort(key=lambda x: x[1], reverse=True)

            # 印出state和all_team_name
            # for i in range(len(state)):
            #     print(state[i])

            # save to csv
            with open(f'./spider/rank_data/{year}-{year+1}_random_walk_matrix{year}.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # writer.writerow(['team', 'state'])
                for i in range(len(state)):
                    writer.writerow([state[i][0], state[i][1]])



class KendallTauCalculator:
    def __init__(self):
        pass

    def kendall_tau(self, elo_array, ap_array):
        '''
        Calculate the similarity between two rankings using Kendall's tau coefficient.
        Kendall's tau coefficient is calculated as:
        tau = (P-Q)/(P+Q)
        P: Number of concordant pairs
        Q: Number of discordant pairs
        '''
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

    def calculate_average_tau(self):
        for year in range(2003, 2023):
            # 做為比較的兩個排名是用elo_100和random walk的排名，因為用已經繼承的檔案會有多的隊伍出現
            elo_rank = pd.read_csv(
                f'./spider/rank_data/{year}-{year+1}_elo100.csv', sep='\t')
            # elo_rank = pd.read_csv(
            #     f'./spider/rank_data/{year}-{year+1}_elo100_K32_shuffleTrue_stepLRFalse_inheritFalse.csv', sep='\t')
            random_walk_rank = pd.read_csv(
                f'./spider/rank_data/{year}-{year+1}_random_walk_matrix{year}.csv', sep='\t')

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

            # print(f"第{year}-{year+1}年度: ", Kendall_tau(elo_array, random_walk_array))
            CORR = []
            CORR.append(self.kendall_tau(elo_array, random_walk_array))
        # print("平均值: " , sum(CORR)/len(CORR))
        return (sum(CORR)/len(CORR))



for i in range(1, 10):
    FLASH = FLASH + ADD_F * i
    ARRAY_FLASH.append(FLASH)
    
    RWM = RandomWalkMatrix(FLASH, CON)
    print("FLASH: ", FLASH, "CON: ", CON)
    ELO = KendallTauCalculator()
    average_tau = ELO.calculate_average_tau()
    print("Average Kendall's tau coefficient:", average_tau)
    ARRAY_AVERAGE_TAU.append(average_tau)
    
plt.plot(ARRAY_FLASH, ARRAY_AVERAGE_TAU)
plt.xlabel('FLASH')
plt.ylabel('Average Kendall Tau')
plt.title('FLASH vs Average Kendall Tau Convergence: 0.0001')
plt.show()
# print(ARRAY_FLASH)
# print(ARRAY_AVERAGE_TAU)

