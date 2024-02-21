import random
import pandas as pd

record = pd.read_csv('./spider/rank_data/2022-2023_Record.csv', sep = "\t")

def deal_team_name(team_name):
    for i in range(len(team_name)):
        if '\xa0' in team_name[i]:
            team_name[i] = team_name[i].replace('\xa0', '')
        if '(' in team_name[i]:
            while '(' in team_name[i]:
                start = team_name[i].find('(')
                end = team_name[i].find(')')
                team_name[i] = team_name[i][:start] + team_name[i][end+1:]

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

class RandomWalk:
    def __init__(self, nodes):
        self.nodes = nodes

    def walk(self, steps):
        global pass_time
        current_node = random.choice(self.nodes)
        print("Starting at node:", current_node.name)
        for _ in range(steps):
            # print("Current node:", current_node.name)
            pass_time[current_node.name] += 1
            current_node = self.choose_next_node(current_node)
        print("Finished at node:", current_node.name)
        pass_time[current_node.name] += 1
        
        # find 10 teams that have the highest pass time
        pass_time = dict(sorted(pass_time.items(), key=lambda item: item[1], reverse=True))
        for i in range(10):
            print(f"第{i+1}名次 : ",list(pass_time.keys())[i], " 經過次數: ",list(pass_time.values())[i])
            
            

    def choose_next_node(self, current_node):
        next_node = random.choices(current_node.neighbors, weights=current_node.probabilities)[0]
        return next_node


# get the col4 and col7 of the record and combine them into a list
header = record.columns[4]
win_team_name = record.iloc[:, 4]
win_team_name = win_team_name.dropna()
win_team_name = win_team_name.tolist()
win_team_name.insert(0, header)
deal_team_name(win_team_name)

header = record.columns[7]
lose_team_name = record.iloc[:, 7]
lose_team_name = lose_team_name.dropna()
lose_team_name = lose_team_name.tolist()
lose_team_name.insert(0, header)
deal_team_name(lose_team_name)

# get the col5 and col8 of the record and combine them into a list
header = record.columns[5]
score_col5 = record.iloc[:, 5]
score_col5 = score_col5.dropna()
score_col5 = score_col5.tolist()
score_col5.insert(0, int(header))
header = record.columns[8]
score_col8 = record.iloc[:, 8]
score_col8 = score_col8.dropna()
score_col8 = score_col8.tolist()
score_col8.insert(0, int(header))

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
total_lose_point = {}
pass_time = {}
for node in node_array:
    total_lose_point[node.name] = 0
    pass_time[node.name] = 0

# 增加鄰居節點及機率
for node in node_array:
    # 創建array查找對手
    # 創建一個win_array和lose_array，分別存放贏和輸的隊伍的index
    win_array = []
    lose_array = []
    for i in range(len(win_team_name)):
        if node.name == win_team_name[i]:
            win_array.append(i)
        elif node.name == lose_team_name[i]:
            lose_array.append(i)
    
    # 計算總失分
    for i in range(len(lose_array)):
        total_lose_point[node.name] += score_col5[lose_array[i]]
    for i in range(len(win_array)):
        total_lose_point[node.name] += score_col8[win_array[i]]    
    
    # print(total_lose_point)
    
    for i in range(len(win_array)):
        for neighbor_node in node_array:
            if neighbor_node.name == lose_team_name[win_array[i]]:
                neighbor = neighbor_node
                probability = score_col8[win_array[i]] / total_lose_point[node.name]
                node.add_neighbor(neighbor, probability)
            elif neighbor_node.name == node.name:
                continue
            else:
                neighbor = neighbor_node
                probability = 0.001
                node.add_neighbor(neighbor, probability)
    for i in range(len(lose_array)):
        for neighbor_node in node_array:
            if neighbor_node.name == win_team_name[lose_array[i]]:
                neighbor = neighbor_node
                probability = score_col5[lose_array[i]] / total_lose_point[node.name]
                node.add_neighbor(neighbor, probability)
            elif neighbor_node.name == node.name:
                continue
            else:
                neighbor = neighbor_node
                probability = 0.001
                node.add_neighbor(neighbor, probability)

random_walk = RandomWalk(node_array)

random_walk.walk(1000000)