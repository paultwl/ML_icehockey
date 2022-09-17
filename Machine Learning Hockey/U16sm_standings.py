
from time import sleep
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import csv

seasons = ["2011-2012","2012-2013","2013-2014","2014-2015","2015-2016"]

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
for season in seasons:
    URL = f"https://www.eliteprospects.com/league/u16-sm-sarja/{seasons[seasons.index(season)]}"
    driver.get(URL)
    sleep(1)
    raw = driver.find_elements(by=By.XPATH, value="/html/body/section/div/div[1]/div[4]/div[3]/div/div[1]/div/div[3]/table/tbody")
    raw = raw[-1].text.split("\n")
    clean = []
    for dp in raw:
        splitrow = dp.split(" ", maxsplit=4)
        del splitrow[-1]
        
        if splitrow[3] in ['30', '18']:
            splitrow[1] = splitrow[1] + splitrow[2]
        else: splitrow[1] = splitrow[1] + splitrow[2] + splitrow[3]
        
        del splitrow[3]
        del splitrow[2]
        splitrow[0].removesuffix('.') #delete the . dot from the standings
        splitrow.append(season)
        clean.append(splitrow)

    with open("U16sm_standings.csv", "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerows(clean)

driver.quit()
