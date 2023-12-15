from selenium import webdriver
from selenium.webdriver.common.by import By
# driver = webdriver.(executable_path=".\\spider\\msedgedriver.exe")
service = webdriver.EdgeService(executable_path='.\\spider\\msedgedriver.exe')
driver = webdriver.Edge(service=service)

driver.get(
    "https://sportsdata.usatoday.com/football/ncaaf/coaches-poll/2020-2021/2021-01-12")

school_data = []

x = driver.find_elements(By.TAG_NAME, 'table')
for i in x:
    # print(i.text)
    y = i.find_elements(By.TAG_NAME, 'tr')
    for j in y:
        # print(j.text)
        temp = []
        z = j.find_elements(By.TAG_NAME, 'td')
        for k in z:
            # print(k.text)
            temp.append(k.text)
        school_data.append(temp)

for _ in school_data:
    print(_)


driver.close()
