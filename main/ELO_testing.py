import pandas as pd
import random as rd


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
                f'./spider/rank_data/{year}-{year+1}_random_walk_matrix.csv', sep='\t')

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
        return sum(CORR)/len(CORR)

# Example usage:
kendall_calculator = KendallTauCalculator()
average_tau = kendall_calculator.calculate_average_tau()
print("Average Kendall's tau coefficient:", average_tau)
