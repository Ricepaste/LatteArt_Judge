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
        current_node = random.choice(self.nodes)
        print("Starting at node:", current_node.name)
        for _ in range(steps):
            print("Current node:", current_node.name)
            current_node = self.choose_next_node(current_node)
        print("Finished at node:", current_node.name)

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

node_array = []

# 創建節點
# combine the win and lose team name and remove the duplicate
all_team_name = set(win_team_name + lose_team_name)
for team in all_team_name:
    # create new node for each team
    node = Node(team)
    # add the new node to the node_array
    node_array.append(node)

# 增加鄰居節點及機率
for node in node_array:
    # random choose the neighbor and add the probability
    # 創建array查找對手
    # 創建一個win_array和lose_array，分別存放贏和輸的隊伍的index
    win_array = []
    lose_array = []
    for i in range(len(win_team_name)):
        if node.name == win_team_name[i]:
            win_array.append(i)
        elif node.name == lose_team_name[i]:
            lose_array.append(i)
    
    
    for i in range(len(win_array)):
        for neighbor_node in node_array:
            if neighbor_node.name == lose_team_name[win_array[i]]:
                neighbor = neighbor_node
                break
        probability = random.random()
        node.add_neighbor(neighbor, probability)
    for i in range(len(lose_array)):
        for neighbor_node in node_array:
            if neighbor_node.name == win_team_name[lose_array[i]]:
                neighbor = neighbor_node
                break
        probability = random.random()
        node.add_neighbor(neighbor, probability)

random_walk = RandomWalk(node_array)

random_walk.walk(20)