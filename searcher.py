import googlesearch as gs
from bs4 import BeautifulSoup as bs
import requests
import torch

with open('./Data_and_models/stopwords.txt') as stops:
    for i in stops:
        stopwords = eval(i)
   

#model = torch.load('./Data_and_models/model80')


def remove_stops(x, tolist=False):
    if tolist:
        return list(filter(lambda x: x not in stopwords, x))
    else:
        return filter(lambda x: x not in stopwords, x)

#probably won't be used since google already sorts the results fairly well
def scorer(x, target):
    scores = {'p': 1, 'h': 3, 'meta': 8, 'title': 10, }
    target = remove_stops(target)
    r = requests.get(x)
    soup = bs(r.content)
    score = 0
    for i in scores:
        for j in soup.find_all(i):
            for word in target:
                score += j.count(word) * scores[i]
    return {scores: x}


reliable_endings = ['.org', '.int', '.gov', 'PolitiFact', '.us','.edu']


def check_if_reliable(x):
    for i in reliable_endings:
        if i in x:
            return True
    return False


def googler(search, reliable=False, score=False):
    results = list(gs.search(search))
    if not reliable:
        return results
    elif reliable and score:
        good_ones = filter(lambda x: check_if_reliable(x), results)
        rated = {}
        for i in good_ones:
            rated.update(scorer(i, search))
        rated = {j: rated[j] for j in sorted(rated)}
        return rated
    elif score:
        rated = {}
        for i in results:
            rated.update(scorer(i, search))
        rated = {j: rated[j] for j in sorted(rated)}
        return rated
    else:
        return list(filter(lambda x: check_if_reliable(x), results))

def scraper(url):
    r=requests.get(url)
    soup=bs(r.content)
    titles=[str(i) for i in soup.findAll('title')]
    metas=[str(i) for i in soup.findAll('meta')]
    p=[i for i in soup.findAll('p') if len(str(i))>100]
    return (titles,p,metas)

def isurl(x):
    return 'https://' in str(x)

