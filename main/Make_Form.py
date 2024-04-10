import pandas as pd

for year in range(2018, 2023):
    data1 = pd.read_csv(f'./spider/rank_data/{year}-{year+1}_random_walk_matrix_Flash0.0001_CON0.1_INIT0.01.csv', header=None)
    data2 = pd.read_csv(f'./spider/rank_data/{year}-{year+1}_elo10_K24_shuffleFalse_stepLRFalse_inheritFalse.csv', sep = '\t', header=None)
    data3 = pd.read_csv(f'./spider/rank_data/{year}-{year+1}_Bradley_Terry.csv', header=None)
    
    # get the first 10 rows of first column of data1, data2, data3
    rank1 = list(data1.iloc[:10, 0].values)
    rank2 = list(data2.iloc[:10, 0].values)
    rank3 = list(data3.iloc[:10, 0].values)
    
    # combine them into a list, same index means same row
    All_rank = []
    for i in range(10):
        All_rank.append([i+1, rank1[i], rank2[i], rank3[i]])
    # print(All_rank)
    
    # save into a csv file
    df = pd.DataFrame(All_rank, columns=['Rank', 'Rank 1', 'rank 2', 'Rank 3'])
    df.to_csv(f'./spider/rank_data/{year}-{year+1}_all_rank_form.csv', index=False)