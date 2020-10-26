import pandas as pd
import torch
import string
import sys as s
import sys
from collections import defaultdict
from torch import nn


def p(thing):
    s.stdout.write(thing)


def isnan(x):
    if type(x) == str:
        return False
    try:
        int(x)
        return False
    except:
        return True


class fbar:
    def __init__(self, length):
        self.length = length - 1
        self.count = 0
        self.slen = 0

    def step(self):
        if self.count == self.length:
            sys.stdout.write('\b' * self.slen)
            print(f'{self.count}/{self.length}')
            return
        sys.stdout.write('\b' * self.slen)
        s = f'{self.count}/{self.length} '
        self.slen = len(s)
        sys.stdout.write(s)
        self.count += 1


newsdf = pd.read_csv('./Data_and_models/corona_fake.csv')
titles = newsdf['title']
text = newsdf['text']
labels = newsdf['label']
letters = string.ascii_letters + ' !0123456789?'


def enum1(x):
    for i in range(len(x)):
        yield (i + 1, x[i])


vocab = {letter: i for i, letter in enum1(letters)}
vocab = defaultdict(lambda: 0, vocab)


# print(vocab[''],vocab['7'])


def word2tensor(word):
    tens = [vocab[i] for i in word]
    return torch.tensor(tens) if len(tens) > 0 else torch.tensor([0])


def stack(data):
    result = []
    for i in range(len(data)):
        try:
            result.append(word2tensor(data[i]))
        except:
            result.append(torch.zeros(1))
    return result


tensorlabel = []
for label in labels:
    if label == 'Fake' or label == 'fake':
        tensorlabel.append(torch.zeros(1))
    elif label == 'TRUE':
        tensorlabel.append(torch.ones(1))
    else:
        # print(label,'n')
        tensorlabel.append(-1)
tensortitles = stack(titles)


def greatest_len(x):
    great = len(x[0])
    for i in x:
        if len(i) > great:
            great = i
        return great


def pad_to(x, length):
    return torch.nn.functional.pad(x, (0, length - len(x)))


def clean(thing, length):
    new = []
    for sentence in thing:
        new.append(pad_to(sentence, length))
    return new



with open('./Data_and_models/stopwords.txt') as stops:
    for i in stops:
        stopwords = eval(i)


def remove_stops(x, tolist=False):
    if tolist:
        return list(filter(lambda x: x not in stopwords, x))
    else:
        return filter(lambda x: x not in stopwords, x)

def within(x, y, s):
    return abs(x - y) < s


def mean_deviation(x):
    mean = sum(x) / len(x)
    means = [abs(i - mean) for i in x]
    return sum(means) / len(means)


def flat(x, x2, opt, olr):
    if mean_deviation(x[-3:]) < 0.002 or mean_deviation(x2[-3:]) < 0.002:
        for i in opt.param_groups:
            i['lr'] += 0.00002
    else:
        if opt.param_groups[0]['lr'] != olr and not within(opt.param_groups[0]['lr'], olr, 0.000001):
            for param in opt.param_groups:
                param['lr'] -= 0.00001
        if opt.param_groups[0]['lr'] < olr:
            for pa in opt.param_groups:
                pa['lr'] = olr


def train_better(ob, trains, vals, epochs, optim='sgd', lr=0.01, loss='bce'):
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ob().to(device)
        if loss == 'bce':
            lossf = nn.BCELoss()
            fil_func = lambda x: x
        elif loss == 'logs':
            lossf = nn.BCEWithLogitsLoss()
            fil_func = lambda x: torch.sigmoid(x)
        else:
            lossf = nn.MSELoss()
            fil_func = lambda x: x
        if optim == 'sgd':
            o = torch.optim.SGD(model.parameters(), lr)
        elif optim == 'adam':
            o = torch.optim.Adam(model.parameters(), lr)
        elif optim == 'sadam':
            o = torch.optim.SparseAdam(model.parameters(), lr)
        losses = [float('inf'), float('inf'), float('inf')]
        valis = [float('inf'), float('inf'), float('inf')]
        for epoch in range(epochs):
            p = fbar(len(trains))
            for i, j in trains:
                p.step()
                pred = model(i)
                target = j.float().to(device)
                loss = lossf(pred.float(), target)
                model.zero_grad()
                o.zero_grad()
                loss.backward()
                o.step()
            model.eval()
            with torch.no_grad():
                count = 0
                for vi, vt in vals:
                    pred = model(vi)
                    if torch.round(fil_func(pred)) == vt.to(device):
                        count += 1
            model.train()
            print(count / len(vals), f' epoch: {epoch}')
            print(loss)
            valis.append(count / len(vals))
            losses.append(loss)
            flat(losses, valis, o, lr)
        return model
    except KeyboardInterrupt:
        return model


def already_train_better(model, trains, vals, epochs, optim='sgd', lr=0.01, loss='bce'):
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if loss == 'bce':
            lossf = nn.BCELoss()
            fil_func = lambda x: x
        elif loss == 'logs':
            lossf = nn.BCEWithLogitsLoss()
            fil_func = lambda x: torch.sigmoid(x)
        else:
            lossf = nn.MSELoss()
            fil_func = lambda x: x
        if optim == 'sgd':
            o = torch.optim.SGD(model.parameters(), lr)
        elif optim == 'adam':
            o = torch.optim.Adam(model.parameters(), lr)
        elif optim == 'sadam':
            o = torch.optim.SparseAdam(model.parameters(), lr)
        losses = [float('inf'), float('inf'), float('inf')]
        valis = [float('inf'), float('inf'), float('inf')]
        for epoch in range(epochs):
            p = fbar(len(trains))
            for i, j in trains:
                p.step()
                pred = model(i)
                target = j.float().to(device)
                loss = lossf(pred.float(), target)
                model.zero_grad()
                o.zero_grad()
                loss.backward()
                o.step()
            model.eval()
            with torch.no_grad():
                count = 0
                for vi, vt in vals:
                    pred = model(vi)
                    if torch.round(fil_func(pred)) == vt.to(device):
                        count += 1
            model.train()
            print(count / len(vals), f' epoch: {epoch}')
            print(loss)
            valis.append(count / len(vals))
            losses.append(loss)
            flat(losses, valis, o, lr)
        return model
    except KeyboardInterrupt:
        return model


def csv_reader(fname):
    with open(fname, 'r') as r:
        d = list(r)
        heads = [[] for i in d[0].split('|')]
        for line in d[1:]:
            for i in range(len(heads)):
                heads[i].append(line.split('|')[i])
        return {d[0].split('|')[i].strip('\n'): heads[i] for i in range(len(heads))}

all_text = [(text[i], tensorlabel[i]) for i in range(len(text)) if not isnan(text[i]) and tensorlabel[i] > -1]
all_titles=[(titles[i], tensorlabel[i]) for i in range(len(titles)) if not isnan(titles[i]) and tensorlabel[i] > -1]
print('data done')