trans = '''
    F1                 0.155
    F2                 0.147
    F3                 0.146
    F4                 0.147
    F5                 0.147
    F6                 0.148
    F7                 0.146
    F8                 0.146
    F9                 0.153
    F10                0.147
    F11                0.147
    F12                0.147
    F13                0.147
    F14                0.147
    F15                0.146
    F16                0.147
    F17                0.147
    F18                0.146
    F19                0.000
'''
team = ['NotreDame', 'Michigan', 'PennState', 'AppalachianState', 'Clemson', 'TexasA&M', 'Kentucky', 'Florida', 'Alabama', 'Oklahoma',
        'Army', 'Syracuse', 'OhioState', 'LouisianaState', 'Georgia', 'CentralFlorida', 'Cincinnati', 'Washington', 'WashingtonState']

for i in range(len(team)):
    trans = trans.replace(f'    F{i+1} ', team[i])

rank = {}
for line in trans.split('\n'):
    if line == '':
        continue
    line = line.split()
    rank[line[0]] = float(line[1])

num = 1
for key, value in sorted(rank.items(), key=lambda x: x[1], reverse=True):
    print(f'{num}\t{key:25s}{value}')
    num += 1

# print(trans)
