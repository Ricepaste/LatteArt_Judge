import requests
from bs4 import BeautifulSoup

url = "https://sportsdata.usatoday.com/football/ncaaf/coaches-poll/2020-2021/2021-01-12"
res = requests.get(url)
# print(res.status_code)

soup = BeautifulSoup(res.text, "html.parser")
soup = soup.find('table')
name = BeautifulSoup(str(soup), "lxml")
rank = BeautifulSoup(str(soup), "lxml")
name = name.find_all('span')
rank = BeautifulSoup(str(rank.find_all('tr')), "lxml")
rank = rank.find_all('td')

# for i in name:
#     print(i.text)

for i in rank:
    print(i.text)
