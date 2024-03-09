import random
import pandas as pd
import csv

# STEP為隨機遊走的步數 RW_TIMES為隨機遊走的次數
STEP = 1000
RW_TIMES = 300
RANDOM_SEEDS = 50
WRITE = 1
random.seed(RANDOM_SEEDS)


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


class Node:
    def __init__(self, name):
        self.name = name
        self.neighbors = []
        self.probabilities = []

    def add_neighbor(self, neighbor, probability):
        if neighbor not in self.neighbors:
            self.neighbors.append(neighbor)
            self.probabilities.append(probability)
            neighbor.neighbors.append(self)
            neighbor.probabilities.append(probability)
        else:
            index = self.neighbors.index(neighbor)
            self.probabilities[index] += probability
            # index = neighbor.neighbors.index(self)
            # neighbor.probabilities[index] += probability


class RandomWalk:
    def __init__(self, nodes):
        self.nodes = nodes

    def walk(self, steps):

        global pass_time, year
        for _1 in range(RW_TIMES):
            # fixed random seeds
            current_node = random.choice(self.nodes)
            # print("Starting at node:", current_node.name)
            for _ in range(steps):
                # print("Current node:", current_node.name)
                pass_time[current_node.name] += 1
                current_node = self.choose_next_node(current_node)
            # print("Finished at node:", current_node.name)
            pass_time[current_node.name] += 1

        # find 10 teams that have the highest pass time
        pass_time = dict(
            sorted(pass_time.items(), key=lambda item: item[1], reverse=True))
        print(f"第{year}年度Ranking: ")
        for i in range(10):
            print(f"第{i+1}名次 : ", list(pass_time.keys())
                  [i], " 經過次數: ", list(pass_time.values())[i])
        print("=====================================")

        # 將結果寫入csv
        if WRITE:
            with open(f'./spider/rank_data/{year}-{year+1}_RandomWalk_STEP{STEP}_RWTIMES{RW_TIMES}.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                for key, value in pass_time.items():
                    writer.writerow([key, value])

    def choose_next_node(self, current_node):
        # TODO 加上移動前判定這次是否會閃現的功能，機率0.001
        if random.random() <= 0.001:
            next_node = random.choice(self.nodes)
            # print("jump")
        else:
            # print("move")
            next_node = random.choices(
                current_node.neighbors, weights=current_node.probabilities, k=1)[0]
        # for i in current_node.neighbors:
        #     print(i.name)
        # print(current_node.probabilities)
        # print(next_node.name)
        return next_node

# 例外狀況 有比賽被取消，比分先暫定0:0 (手動加上比分)
# 3	Sep 14, 2013	2:00 PM	Sat	Fresno State		@	Colorado		Game Cancelled
# 1	Sep 5, 2015	7:30 PM	Sat	McNeese State		@	(14) Louisiana State		Cancelled due to weather


for year in range(2003, 2023):

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

    # score
    # get the col5 and col8 of the record and combine them into a list
    # header = record.columns[score_index]
    # score_winner = record[score_index]
    # score_winner = score_winner.dropna()
    # score_winner = score_winner.tolist()

    # header = record.columns[score_index+3]
    # score_loser = record[score_index+3]
    # score_loser = score_loser.dropna()
    # score_loser = score_loser.tolist()

    node_array = []

    # 創建節點
    # combine the win and lose team name and remove the duplicate
    all_team_name = set(win_team_name + lose_team_name)
    for team in all_team_name:
        # create new node for each team
        node = Node(team)
        # add the new node to the node_array
        node_array.append(node)

    # 創建一個字典，存放每個隊伍的總失分
    total_lose_times = {}
    pass_time = {}
    for node in node_array:
        total_lose_times[node.name] = 0
        pass_time[node.name] = 0

    # 增加鄰居節點及機率
    # TODO 查找node效率有點低，可能可以改成字典
    for node in node_array:
        # 創建array查找對手
        # 創建一個win_array和lose_array，分別存放贏和輸的隊伍的index
        win_array = []
        lose_array = []
        for i in range(len(win_team_name)):
            if node.name == win_team_name[i]:
                if node.name == 'Oklahoma':
                    print(lose_team_name[i])
                win_array.append(i)
            elif node.name == lose_team_name[i]:
                lose_array.append(i)
                total_lose_times[node.name] += 1

        # 計算總失分
        # for i in range(len(lose_array)):
        #     total_lose_point[node.name] += score_col5[lose_array[i]]
        # for i in range(len(win_array)):
        #     total_lose_point[node.name] += score_col8[win_array[i]]

        # print(total_lose_point)

        for i in range(len(lose_array)):
            # TODO 查詢效率低
            for neighbor_node in node_array:
                if neighbor_node.name == win_team_name[lose_array[i]]:
                    neighbor = neighbor_node
                    try:
                        probability = 1 / total_lose_times[node.name]
                    except:
                        assert False, "total_lose_times[node.name] = 0"
                        # probability = 0
                    node.add_neighbor(neighbor, probability)

# TODO bookmark
    random_walk = RandomWalk(node_array)

    random_walk.walk(STEP)
