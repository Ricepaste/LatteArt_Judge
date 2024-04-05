from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from time import sleep

# driver = webdriver.(executable_path=".\\spider\\msedgedriver.exe")
Service = webdriver.EdgeService(executable_path='.\\spider\\msedgedriver.exe')
driver = webdriver.Edge(service=Service)

driver.get(
    "https://eco.mtk.nao.ac.jp/cgi-bin/koyomi/eclipsex_s_en.cgi")


# hi = driver.find_elements(By.TAG_NAME, 'body')
# for i in hi:
#     print(i.text)

hi = driver.find_element(
    By.XPATH, '/html/body/form/div[1]/div[2]/fieldset[1]/p')
print(hi.text)

with open('spider/angle90.txt', 'w+') as file:
    file.write(hi.text)
    file.write('\n')

x = input("enter any key to continue...")
driver.close()
