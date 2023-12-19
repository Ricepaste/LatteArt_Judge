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
    # 創建資料夾
    if (i == 3 or i == 6 or i == 9 or i == 10):
        plus = 2
    else:
        plus = 1
        
    team_folder = os.path.join(folder, Division[i])
    
    team_name = []
    team_team_link = []
    for j in range(index, index + plus):
        td_tags = soup.find_all('tbody')[j].find_all('tr')[1:]
        
        for td in td_tags:
            team_name.append(td.find('a').text)
            try:
                if (td.find('a').text not in os.listdir(team_folder)):
                    os.mkdir(os.path.join(team_folder, td.find('a').text))
            except:
                continue
        # print(len(team_name))
        # 抓取隊伍連結

        for tr in soup.find_all('tbody')[j].find_all('tr'):
            link = tr.find('a')
            if link:
                team_team_link.append(link['href'])
        # print(len(team_team_link))
    index += plus
    # ---------------------------------------------        
            

    # for j in range(len(team_team_link)):
    #     team_web = requests.get("http://covers.com"+team_team_link[j])
    #     team_soup = bs(team_web.text, 'html.parser')
        
    
    




    
