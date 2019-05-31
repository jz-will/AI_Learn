import requests
from bs4 import BeautifulSoup

path = "E:/LEARN/大二上学期/职业规划/【职业生涯规划课】作业20190104/朗途职业规划报告.htm"
with open(path,"r") as file:
    soup = BeautifulSoup(file)
print(soup)