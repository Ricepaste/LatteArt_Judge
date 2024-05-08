# Random Walk
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import numpy as np

# # Function to generate random walk data
# def random_walk(n_steps):
#     # Initialize arrays to store x and y coordinates
#     x = np.zeros(n_steps)
#     y = np.zeros(n_steps)
    
#     # Generate random steps
#     for i in range(1, n_steps):
#         # Generate random direction (0-3)
#         direction = np.random.randint(0, 4)
#         if direction == 0:
#             x[i] = x[i-1] + 1  # Move right
#         elif direction == 1:
#             x[i] = x[i-1] - 1  # Move left
#         elif direction == 2:
#             y[i] = y[i-1] + 1  # Move up
#         else:
#             y[i] = y[i-1] - 1  # Move down
    
#     return x, y

# # Number of steps in the random walk
# n_steps = 50

# # Generate random walk data
# x, y = random_walk(n_steps)

# # Function to update plot
# def update(frame):
#     plt.cla()
#     if frame < len(x):
#         plt.plot(x[:frame], y[:frame], lw=2)
#         plt.scatter(x[frame-1], y[frame-1], color='red', s=100)  # Mark the current position
#         plt.title('Random Walk')
#         plt.xlabel('X')
#         plt.ylabel('Y')
#         plt.grid(True)

# # Create animation
# fig = plt.figure(figsize=(8, 6))
# ani = animation.FuncAnimation(fig, update, frames=range(1, n_steps), interval=500)

# # Save animation to file using Pillow
# ani.save('random_walk_animation.gif', writer='pillow')

# plt.show()


# Elo Rating System
import matplotlib.pyplot as plt

class Team:
    def __init__(self, name, initial_elo):
        self.name = name
        self.elo_ratings = [initial_elo]

def update_elo(winning_team, losing_team, k_factor=32):
    # 計算預期勝率
    expected_win_winning = 1 / (1 + 10 ** ((losing_team.elo_ratings[-1] - winning_team.elo_ratings[-1]) / 400))
    expected_win_losing = 1 - expected_win_winning
    
    # 根據實際結果更新 Elo 分數
    winning_team_new_elo = winning_team.elo_ratings[-1] + k_factor * (1 - expected_win_winning)
    losing_team_new_elo = losing_team.elo_ratings[-1] + k_factor * (0 - expected_win_losing)
    
    # 將新的 Elo 分數添加到隊伍列表中
    winning_team.elo_ratings.append(winning_team_new_elo)
    losing_team.elo_ratings.append(losing_team_new_elo)

def plot_teams(teams, game_number):
    for team in teams:
        plt.plot(range(len(team.elo_ratings)), team.elo_ratings, marker='o', label=team.name)
        for i, elo in enumerate(team.elo_ratings):
            plt.annotate(round(elo, 2), (i, elo), textcoords="offset points", xytext=(0,10), ha='center')
    plt.title('Elo Ratings Over Time')
    plt.xlabel('Game Number')
    plt.ylabel('Elo Rating')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'Elo_Ratings_Game_{game_number}.png')  # 儲存圖片
    plt.show()

# 創建兩支隊伍
team1 = Team("Team 1", 1500)
team2 = Team("Team 2", 1500)

# 模擬幾場比賽並根據結果更新 Elo 分數
matches = [(team1, team2), (team1, team2), (team2, team1), (team2, team1), (team1, team2)]
for i, match in enumerate(matches):
    # 假設隊伍 1 贏得了比賽
    update_elo(match[0], match[1])
    # 繪製並儲存折線圖
    plot_teams([team1, team2], i+1)



# Bradley-Terry Model
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import minimize

# class BradleyTerryModel:
#     def __init__(self, n_players):
#         self.n_players = n_players
#         self.strengths = np.ones(n_players)  # 初始設定所有參賽者的實力參數為1
#         self.iter_count = 0  # 初始化迭代計數器
    
#     def fit(self, matches, save_paths):
#         def log_likelihood(strengths):
#             ll = 0
#             for match in matches:
#                 i, j, result = match
#                 p_i = strengths[i] / (strengths[i] + strengths[j])
#                 p_i = np.clip(p_i, 1e-15, 1 - 1e-15)  # 將概率限制在0和1之間，避免出現無效的對數值
#                 ll += result * np.log(p_i) + (1 - result) * np.log(1 - p_i)
#             return -ll
        
#         def callback(strengths):
#             self.iter_count += 1  # 每次調用增加迭代計數器
#             self.plot_strengths(strengths, save_paths[self.iter_count - 1])
        
#         # 最大化對數概率，使用L-BFGS-B作為優化方法，並增加最大迭代次數限制
#         result = minimize(log_likelihood, self.strengths, method='L-BFGS-B', callback=callback, options={'maxiter': 1000})
#         self.strengths = result.x
    
#     def plot_strengths(self, strengths, save_path):
#         plt.plot(range(0, self.n_players), strengths, marker='o')
#         for i, strength in enumerate(strengths):
#             plt.annotate(round(strength, 2), (i, strength), textcoords="offset points", xytext=(20,0), ha='center')
#         plt.title('Strength Parameter Changes Over Time')
#         plt.xlabel('Player Index')
#         plt.ylabel('Strength Parameter')
#         plt.xticks(range(0, self.n_players))
#         plt.grid(True)
#         plt.savefig(save_path)  # 將圖片保存為文件
#         plt.close()  # 關閉當前的圖表，以釋放資源

# # 假設有4名參賽者，初始化 Bradley-Terry 模型
# n_players = 4
# bt_model = BradleyTerryModel(n_players)

# # 假設這是對戰紀錄，每場比賽包括兩名參賽者的索引和比賽結果（1表示第一個參賽者獲勝，0表示第二個參賽者獲勝）
# matches = [(0, 1, 1), (1, 2, 0), (2, 0, 1), (3, 1, 0), (2, 3, 1)]

# # 指定圖片保存的路徑和文件名
# save_paths = ["strength_parameter_changes_1.png", "strength_parameter_changes_2.png", "strength_parameter_changes_3.png", "strength_parameter_changes_4.png", "strength_parameter_changes_5.png", "strength_parameter_changes_6.png"]

# # 使用 Bradley-Terry 模型擬合對戰紀錄並保存每次生成的圖片
# bt_model.fit(matches, save_paths)

# print("Number of parameter updates:", bt_model.iter_count)
