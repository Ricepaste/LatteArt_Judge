import Elo
import random
import numpy as np


class TTmodel():
    def __init__(self, teams_amount) -> None:
        self.player_list = []
        self.mu_calculations = {}
        self.mu = None  # shoud be n*1 array
        self.var = 0.5
        self.n = teams_amount
        self.A = np.array([])  # shoud be C_(n,2) * n array

    def addPlayer(self, player):
        self.player_list.append(player)

    def gameOver(self, winner, loser):
        if winner not in self.mu_calculations:
            self.mu_calculations[winner] = [1, 1]
        else:
            self.mu_calculations[winner][0] += 1
            self.mu_calculations[winner][1] += 1

        if loser not in self.mu_calculations:
            self.mu_calculations[loser] = [0, 1]
        else:
            self.mu_calculations[loser][1] += 1

    def fit(self):
        self.mu = []

        for name in self.player_list:
            self.mu.append(
                self.mu_calculations[name][0] / self.mu_calculations[name][1])

        self.mu = np.array(self.mu)
        self.mu = self.mu / ((1 + 2 * self.var)**0.5)
        self.var = self.var / (1 + 2 * self.var)


def TT_calculate(winner, loser, K=32, epochs=1, shuffle=False, stepLR=True, league=None, schoolset=None):
    '''
    計算所有隊伍的elo值
    註:elo值的計算方式為:elo = elo + k * (result - expected_result)

    winner: 參數為每場比賽的勝者(list)
    loser: 參數為每場比賽的敗者(list)
    K: 參數為elo值的變化率
    epochs: 參數為計算elo值的次數
    shuffle: 參數為是否打亂順序
    stepLR: 參數為是否使用learning rate schduler
    league: 參數為上一年的league
    schoolset: 參數為上一年的schoolset
    '''
    all_team = set(winner)
    all_team.update(set(loser))
    n = len(all_team)
    ranking = []
    if league is None:
        TTLeague = TTmodel(n)
    else:
        TTLeague = league
    if schoolset is None:
        school_join = set()
    else:
        school_join = schoolset
        print("Inherit last record...")
        print(school_join)
        print("Processing Elo...")

    for _ in range(epochs):
        # print(f"{_} times")

        # 打亂比賽紀錄訓練順序
        if shuffle:
            # print([(w, l) for w, l in zip(winner, loser)])
            # print()
            start_state = random.getstate()
            random.shuffle(winner)
            random.setstate(start_state)
            random.shuffle(loser)
            # print([(w, l) for w, l in zip(winner, loser)])

        result = zip(winner, loser)
        for w, l in result:
            if not (w in school_join):
                school_join.add(w)
                TTLeague.addPlayer(w)
            if not (l in school_join):
                school_join.add(l)
                TTLeague.addPlayer(l)
            TTLeague.gameOver(winner=w, loser=l)

        for key, value in TTLeague.mu_calculations.items():
            print(key, value)
        # print(w, l)

        # learning rate schduler
        if (stepLR):
            TTLeague.k = int(TTLeague.k * 0.9)

    for key, value in sorted((TTLeague.ratingDict).items(), key=lambda x: x[1], reverse=True):
        # print(f"{key:30s}\t{value:.1f}")
        ranking.append([key, value])

    return ranking, TTLeague, school_join


def main(EPOCHS=100, K=32, SHUFFLE=False, STEPLR=False, INHERIT=False):

    league = None
    school = None
    for year in range(2003, 2023):
        winner = Elo.data_load(year, load='winner')
        loser = Elo.data_load(year, load='loser')

        if (INHERIT):
            rank_data, league, school = TT_calculate(
                winner, loser, K=K, epochs=EPOCHS, shuffle=SHUFFLE,
                stepLR=STEPLR, league=league, schoolset=school)
        else:
            rank_data, _, _ = TT_calculate(
                winner, loser, K=K, epochs=EPOCHS, shuffle=SHUFFLE, stepLR=STEPLR)

        Elo.save_to_csv(
            year, rank_data, f'elo{EPOCHS}_K{K}_shuffle{SHUFFLE}_stepLR{STEPLR}_inherit{INHERIT}')


if __name__ == '__main__':
    main()
