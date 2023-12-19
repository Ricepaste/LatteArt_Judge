from bs4 import BeautifulSoup as bs
import os
import requests

# Get the team names
url = 'https://www.covers.com/sport/football/ncaaf/teams'
web = requests.get(url)
soup = bs(web.text, 'html.parser')

# 分辨賽區
Division = ["American Athletic", "Atlantic Coast", "Big 12", "Big Ten", "Conference USA", "Independents I-A", "Mid-American", "Mountain West", "Pacific-12", "Southeastern", "Sun Belt"]
folder = os.path.join(os.getcwd(), 'NCAA\\Division')
for i in range(len(Division)):
    if (Division[i] not in os.listdir(folder)):
        os.mkdir(os.path.join(folder, Division[i]))

index = 0
for i in range(len(Division)):
    if (i == 3 or i == 6 or i == 9 or i == 10):
        plus = 2
    else:
        plus = 1
    team_folder = os.path.join(folder, Division[i])
    for j in range(index, index + plus):
        td_tags = soup.find_all('tbody')[j].find_all('tr')[1:]
        
        for td in td_tags:
            if (td.find('a').text not in os.listdir(team_folder)):
                os.mkdir(os.path.join(team_folder, td.find('a').text))
    index += plus
    
