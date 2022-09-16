import os.path
from time import sleep
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from datetime import datetime
import csv
import pandas as pd


sarja = "u16-sm-sarja"
kausi = "2014-2015"
seasons = ["2011-2012","2012-2013","2013-2014","2014-2015","2015-2016"] #we want to fetch data from 5 seasons
page = 6
MadeItURL = f"https://www.eliteprospects.com/league/{sarja}/stats/{kausi}?has-played-in-league=Liiga&page={page}"
NotItURL = f"https://www.eliteprospects.com/league/{sarja}/stats/{kausi}?never-played-in-league=Liiga&page={page}"


URL = f"https://www.eliteprospects.com/league/{sarja}/stats/{kausi}?page={page}"
print(URL)
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get(URL)
sleep(1) #give the website time to render properly, to ensure all data is fetched
html = driver.page_source #not necessarily needed but lets keep it just in case
#singlepalyerdata = driver.find_elements(by=By.XPATH, value= "/html/body/section/div/div[1]/div[4]/div[4]/div[1]/div/div[4]/table/tbody[1]/tr[1]")
#playerdata = driver.find_elements(by=By.XPATH, value="/html/body/section/div/div[1]/div[4]/div[4]/div[1]/div/div[4]/table/tbody")

playerdata = driver.find_elements(by=By.XPATH, value="/html/body/section/div/div[1]/div[4]/div[4]/div[1]/div/div[4]/table/tbody/tr")
driver.quit #



soup = BeautifulSoup(html, "html.parser")
'''
sorteddata = []
print("pillu!!!\n\n\n")
for dp in playerdata:
    dp = dp.text[dp.text.index(' '):] #remove the number before the name, since it's irrelevant
    dp = dp.split(maxsplit=13) #turn the string into a list of data
#the data is as follows [index, firstname, lastname, position, team, league,(possibly academy), GP, G, A, T, PPG, PIM, +/-]
    if len(dp) == 13:  #these if elses combine the texts signifying teams
        team = dp[3]+dp[4]+dp[5]
        del dp[4]
        del dp[5]
        dp[3] = team
    elif len(dp) == 12:
        team = dp[3]+dp[4]
        del dp[4]
        dp[3] = team
    sorteddata.append(dp)
print("\n\n\npillu!!!\n\n\n")
'''
def datacleaner(pdata: list):
    sorteddata = []
    cleanplayerdata = []
    for dp in pdata:
        dp = dp.text[dp.text.index(' '):] #remove the number before the name, since it's irrelevant
        dp = dp.split() #turn the string into a list of data
    #the data is as follows [firstname, lastname, position, team, league,(possibly academy), GP, G, A, T, PPG, PIM, +/-]
        
        if len(dp) == 13:  #these if elses combine the texts signifying teams
            team = dp[3]+' '+dp[4]+' '+dp[5]
            del dp[5]
            del dp[4]
            dp[3] = team
        elif len(dp) == 12:
            team = dp[3]+' '+dp[4]
            del dp[4]
            dp[3] = team
        sorteddata.append(dp)
        print(len(dp))
        print(dp)

    '''
    for dp in sorteddata: #selecting a team for players who have played in 2 or more teams during the season
        if len(dp) == 11: #this eliminates empty rows and rows containing faulty data i.e. teams only
            if dp[3] == 'totals':
                nextrow = sorteddata[sorteddata.index(dp)+1]
                dp[3] = nextrow[0] + nextrow[1]
            cleanplayerdata.append(dp)
            print(dp)
    '''
datacleaner(playerdata)   








'''
page = soup.contents[0]
#body = page.find_all('table', class_ = "league-stats desktop country-fi")

body = page.find
print(body.contents[1])

print("program finished")

xpath = "/html/body/section/div/div[1]/div[4]/div[4]/div[1]/div/div[4]/table/tbody[1]"
'''