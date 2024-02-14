import pandas as pd

node_array = []

class Node:
    def __init__(self, name):
        self.name = name
        # self.win_score = 0
        self.link_to = []
        
    def add_link(self, link):
        self.link_to.append(link)
        
    def get_links(self):
        return self.link_to

win = []
lose = []

def build_node_array(win_team, lose_team):
    for i in range(len(win_team)):
        if win_team[i] not in win:
            win.append(win_team[i])
            node_array.append(Node(win_team[i]))
        if lose_team[i] not in lose:
            lose.append(lose_team[i])
            node_array.append(Node(lose_team[i]))
    for i in range(len(win_team)):
        for j in range(len(node_array)):
            if win_team[i] == node_array[j].name:
                node_array[j].add_link(lose_team[i])
            if lose_team[i] == node_array[j].name:
                node_array[j].add_link(win_team[i])
            
            

def deal_team_name(team_name):
    for i in range(len(team_name)):
        if '\xa0' in team_name[i]:
            team_name[i] = team_name[i].replace('\xa0', '')
        if '(' in team_name[i]:
            while '(' in team_name[i]:
                start = team_name[i].find('(')
                end = team_name[i].find(')')
                team_name[i] = team_name[i][:start] + team_name[i][end+1:]
    build_node_array(team_name)


record = pd.read_csv('./spider/rank_data/2022-2023_Record.csv', sep = "\t")
# get the col4 of the record
# Win side
header = record.columns[4]
win_team_name = record.iloc[:, 4]
win_team_name = win_team_name.dropna()
win_team_name = win_team_name.tolist()
win_team_name.insert(0, header)
deal_team_name(win_team_name)

# get the col7 of the record
# Lose side
header = record.columns[7]
lose_team_name = record.iloc[:, 7]
lose_team_name = lose_team_name.dropna()
lose_team_name = lose_team_name.tolist()
lose_team_name.insert(0, header)
deal_team_name(lose_team_name)

    
# print(len(node_array))
for i in range(len(node_array)):
    print(node_array[i].name)