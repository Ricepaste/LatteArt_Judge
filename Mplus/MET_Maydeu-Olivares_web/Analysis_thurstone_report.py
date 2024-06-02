trans = '''
    F1                -0.180
    F2                -0.627
    F3                -0.243
    F4                -0.243
    F5                -0.177
    F6                -0.232
    F7                -0.222
    F8                -0.226
    F9                -0.242
    F10               -0.272
    F11               -1.115
    F12               -0.226
    F13               -0.519
    F14               -0.267
    F15               -0.272
    F16               -0.262
    F17               -0.244
    F18                0.000
'''
team = ['FloridaState', 'LouisianaState', 'Mississippi', 'Tulane', 'Oklahoma', 'SouthernMethodist', 'Texas', 'Alabama',
        'JamesMadison', 'Troy', 'Missouri', 'Washington', 'Oregon', 'OhioState', 'PennState', 'Georgia', 'Michigan', 'Louisville']

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
