import pandas as pd

class Elo:
    def __init__(self, winner_index, loser_index, k=32):
        self.df = pd.read_csv("./LabelTool/ForTestingImage.csv", sep=",")
        self.winner = self.df.iloc[winner_index][1]
        self.winner_index = winner_index
        self.loser = self.df.iloc[loser_index][1]
        self.loser_index = loser_index
        self.k = k
        
        self.calculate()
        self.update()
    
    def expected(self, winner, loser):
        return 1 / (1 + 10**((loser - winner) / 400))   
    
    def calculate(self):
        expected_result = self.expected(self.winner, self.loser)
        self.winner = self.winner + self.k * (1 - expected_result)
        self.loser = self.loser + self.k * (expected_result - 1)
        return self.winner, self.loser
    
    def update(self):
        self.df.iloc[self.winner_index][1] = self.winner
        self.df.iloc[self.loser_index][1] = self.loser
        self.df.to_csv("./LabelTool/ForTestingImage.csv", index=False)