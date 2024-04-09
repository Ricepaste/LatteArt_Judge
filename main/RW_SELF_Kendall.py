import pandas as pd
import csv
import os

rank1_result = []
rank1_arg = []
header = ['Flash', 'Convergence', 'Initial_state', "Average Kendall's tau"]

class TeamRankings:
    def __init__(self, flash, convergence, initial_state, flash2, convergence2, initial_state2):
        self.flash = flash
        self.convergence = convergence
        self.initial_state = initial_state
        self.flash2 = flash2
        self.convergence2 = convergence2
        self.initial_state2 = initial_state2
        self.CORR = []
        
        self.calculate_correlation()

    def Kendall_tau(self, arr1, arr2):
        P = 0
        Q = 0
        for i in range(len(arr1)):
            for j in range(i+1, len(arr1)):
                if (arr1[i] > arr1[j] and arr2[i] > arr2[j]) or (arr1[i] < arr1[j] and arr2[i] < arr2[j]):
                    P += 1
                else:
                    Q += 1
        tau = (P-Q)/(P+Q)
        return tau

    def calculate_correlation(self):
        
        for year in range(2003, 2023):
            try:
                rank1 = pd.read_csv(
                    f'./spider/rank_data/{year}-{year+1}_random_walk_matrix_Flash{self.flash}_CON{self.convergence}_INIT{self.initial_state}.csv', sep=',', header=None)
                rank2 = pd.read_csv(
                    f'./spider/rank_data/{year}-{year+1}_random_walk_matrix_Flash{self.flash2}_CON{self.convergence2}_INIT{self.initial_state2}.csv', sep=',', header=None)

                rank1_arr = [rank1.values.tolist()[i][0].rstrip()
                            for i in range(len(rank1.values.tolist()))]
                rank2_arr = [rank2.values.tolist()[i][0].rstrip()
                            for i in range(len(rank2.values.tolist()))]

                for i in range(len(rank1_arr)):
                    rank2_arr[i] = rank1_arr.index(rank2_arr[i])+1
                for i in range(len(rank1_arr)):
                    rank1_arr[i] = i+1

                print("File read successfully")
            except:
                print("File not found")

            correlation = self.Kendall_tau(rank1_arr, rank2_arr)
            self.CORR.append(correlation)
            
            print(f"{year}-{year+1} 年度 Kendall tau係數 : ", correlation)
        
        CORR_AVERAGE = sum(self.CORR)/len(self.CORR)
        
        rank1_result.append(CORR_AVERAGE)
        rank1_arg.append([self.flash, self.convergence, self.initial_state])
        
        if len(rank1_result) == 125:
            # Do the average of the correlation
            avg = sum(rank1_result)/len(rank1_result)    
            nearest = min(rank1_result, key=lambda x: abs(x - avg))
            
            # Write the result to a csv file
            data = [rank1_arg[rank1_result.index(nearest)][0], rank1_arg[rank1_result.index(nearest)][1], rank1_arg[rank1_result.index(nearest)][2], nearest]
            write_to_csv_with_header(data, header)

            rank1_result.clear()
            rank1_arg.clear()
        
def write_to_csv_with_header(data, header):
    with open(f'./spider/rank_data/RW_argument_Kendall_mean.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        if os.path.getsize(f'./spider/rank_data/RW_argument_Kendall_mean.csv') == 0:
            writer.writerow(header)

        writer.writerow(data)

# Parameters
FLASH = [0.1, 0.01, 0.001, 0.0001, 0.00001]
CONVERGENCE = [0.1, 0.01, 0.001, 0.0001, 0.00001]
INITIAL_STATE = [0.1, 0.01, 0.001, 0.0001, 0.00001]

for i in range(len(FLASH)):
    for j in range(len(CONVERGENCE)):
        for k in range(len(INITIAL_STATE)):
            for l in range(len(FLASH)):
                for m in range(len(CONVERGENCE)):
                    for n in range(len(INITIAL_STATE)):
                        team_rankings = TeamRankings(FLASH[i], CONVERGENCE[j], INITIAL_STATE[k], FLASH[l], CONVERGENCE[m], INITIAL_STATE[n])