from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from time import sleep

# driver = webdriver.(executable_path=".\\spider\\msedgedriver.exe")
Service = webdriver.EdgeService(executable_path='.\\spider\\msedgedriver.exe')
driver = webdriver.Edge(service=Service)

driver.get(
    "https://sportsdata.usatoday.com/football/ncaaf/coaches-poll/2020-2021/2021-01-12")
sleep(1)
driver.execute_script('window.scrollTo(0, 500)')   # 捲動到 500px 位置
sleep(1)
driver.execute_script('window.scrollTo(0, 2500)')  # 捲動到 2500px 位置
sleep(1)
driver.execute_script('window.scrollTo(0, 988)')     # 捲動到 0px 位置

school_data = []

table_all = driver.find_elements(By.TAG_NAME, 'table')
for table_each in table_all:
    # print(i.text)
    rows = table_each.find_elements(By.TAG_NAME, 'tr')
    for row in rows:
        # print(j.text)
        temp_row = []
        elements = row.find_elements(By.TAG_NAME, 'td')
        for element in elements:
            # print(k.text)
            temp_row.append(element.text)
        school_data.append(temp_row)

button = driver.find_element(
    By.XPATH, '/html/body/div[1]/div/div[2]/div[2]/div[1]/div[4]/div[1]/div/button')

for _ in school_data:
    print(_)

act = ActionChains(driver)
act.click(button)
act.perform()

competition_date = driver.find_element(
    By.XPATH, '/html/body/div[1]/div/div[2]/div[2]/div[1]/div[4]/div[1]/div/ul').find_elements(By.TAG_NAME, 'li')

for date in competition_date:
    print(date.find_element(By.TAG_NAME, 'a').text)

x = input()
driver.close()
