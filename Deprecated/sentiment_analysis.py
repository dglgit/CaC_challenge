import torch
from torch import nn
from data_extraction import *
import sys as s
import random
print('start')


def p(thing):
    s.stdout.write(thing)


class pbar:
    def __init__(self, length, total, frac=False):
        self.length = length
        self.total = total
        if not frac:
            p('|' + '.' * length + '|')
        else:
            pass
        self.count = 0
        self.thresh = total // length
        self.all_count = 0
        self.n = 0

    def frac(self):
        if self.n == self.total - 1:
            p('\n')
            return
        if self.n > 0:
            p('\b' * self.len)
        string = f'{self.n}/{self.total}'
        p(string)
        self.len = len(string)
        self.n += 1


train_data = all_data[264:]
val_data = all_data[:264]


class Classifiergru(nn.Module):
    def __init__(self, mid=100, seq=550, out=1, embed_dim=10):
        super().__init__()
        self.seq = seq
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(len(vocab) + 1, embed_dim)
        self.gru = nn.GRU(embed_dim, mid)
        self.fc = nn.Linear(mid, out)

    def forward(self, x, hidden):
        x = self.embed(x.long()).float().reshape(self.seq, 1, -1)
        gruout, hidden = self.gru(x, hidden)
        return torch.sigmoid(self.fc(nn.functional.relu(hidden)))


model = Classifiergru()
lossf = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), 0.01)


def train1():
    for epoch in range(7):
        bar = pbar(1, len(train_data), True)
        for data1, data2, target in train_data:
            data = torch.cat((data1, data2))
            bar.frac()
            hidden = torch.randn(1, 1, 100)
            pred = model(data, hidden)
            loss = lossf(pred.reshape(-1), target.float())
            model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        correct = 0
        with torch.no_grad():
            for title, text, target in val_data:
                input = torch.cat((title, text))
                pred = model(input, torch.zeros(1, 1, 100))
                if torch.round(pred).reshape(-1) == target:
                    correct += 1
        model.train()
        print(correct / len(val_data))
        print(loss)


# train1()


class Linearclassifier(nn.Module):
    def __init__(self, embed_dim=10, seq=550, out=1):
        super().__init__()
        self.embed = nn.EmbeddingBag(len(vocab) + 1, embed_dim)
        self.fc1 = nn.Linear(embed_dim, out)
        self.fc2 = nn.Linear(seq, out)

    def forward(self, x):
        xe = self.embed(x.long(), torch.LongTensor([0, 550]))
        x1 = torch.sigmoid(self.fc1(xe))
        # x2=torch.sigmoid(self.fc2(x1))
        return torch.mean(x1)


linearmodel = Linearclassifier()
lossflinear = torch.nn.BCELoss()
optim_linear = torch.optim.Adam(linearmodel.parameters(), lr=0.01)


def train2():
    for epoch in range(200):
        counter = pbar(1, len(train_data), True)
        for data1, data2, target in train_data:
            data = torch.cat((data1, data2))
            counter.frac()
            pred = linearmodel(data)
            loss = lossflinear(pred.reshape(-1), target)
            linearmodel.zero_grad()
            optim_linear.zero_grad()
            loss.backward()
            optim_linear.step()
        if epoch % 10 == 0:
            linearmodel.eval()
            count = 0
            with torch.no_grad():
                for i1, i2, j in val_data:
                    i = torch.cat((i1, i2))
                    if torch.round(linearmodel(i)) == j:
                        count += 1
            print(count / len(val_data))
            linearmodel.train()
            print(loss)


# train2()

class Grudiscrete(nn.Module):
    def __init__(self, embed_dim=10, hidden_s=200, seq1=400, seq2=150):
        super().__init__()
        self.gru1 = nn.GRU(embed_dim, hidden_s)
        self.gru2 = nn.GRU(embed_dim, hidden_s)
        self.embed = nn.Embedding(len(vocab) + 1, embed_dim)
        self.fc1 = nn.Linear(hidden_s, 1)
        self.fc2 = nn.Linear(hidden_s, 1)
        self.fc3 = nn.Linear(2, 1)
        self.seq1 = seq1
        self.seq2 = seq2
        self.hidden_s = hidden_s

    def forward(self, x, x1, hidden):
        embedded1 = self.embed(x.long()).reshape(self.seq1, 1, -1)
        embedded2 = self.embed(x1.long()).reshape(self.seq2, 1, -1)
        encode1, h1 = self.gru1(embedded1, hidden)
        encode2, h2 = self.gru2(embedded2, hidden)
        out1 = nn.functional.relu(self.fc1(h1))
        out2 = nn.functional.relu(self.fc2(h2))
        return torch.sigmoid(self.fc3(torch.cat((out1, out2), dim=2)))


def train3():
    model_both = Grudiscrete()
    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(model_both.parameters())
    for i in range(50):
        bar = pbar(1, len(train_data), True)
        for title, text, target in train_data:
            title = title.reshape(150, -1)
            text = text.reshape(400, -1)
            pred = model_both(text, title, torch.zeros(1, 1, 200))
            loss = loss_func(pred.reshape(-1), target)
            model_both.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bar.frac()
        model_both.eval()
        count = 0
        with torch.no_grad():
            for test_title, test_text, test_target in val_data:
                title = title.reshape(150, -1)
                text = text.reshape(400, -1)
                prediction = model_both(test_text, test_title, torch.zeros(1, 1, 200))
                print(prediction.reshape(-1), test_target)
                if torch.round(prediction.reshape(-1)) == test_target:
                    count += 1
        model_both.train()
        print(count / len(val_data))
        print(loss)


# train3()

class Encode_decode_atten(nn.Module):
    def __init__(self, embed_size=10, mid_size=1, out=1, seq=550, decode_length=10):
        super().__init__()
        self.embed = nn.Embedding(len(vocab) + 1, embed_size)
        self.encoder = nn.GRU(embed_size, mid_size)
        self.encoder_fc = nn.Linear(2 * mid_size, mid_size)
        self.decoder = nn.GRUCell(1, out)
        self.seq = seq
        self.mid_size = mid_size
        self.decode_length = decode_length

    def forward(self, input):
        scores = torch.tensor([])
        embedded = self.embed(input.long())
        embedded = embedded.reshape(self.seq, 1, -1)
        oute, h = self.encoder(embedded)
        for i, j in enumerate(oute):
            scores = torch.cat((scores, self.encoder_fc(torch.cat((j, h.reshape(1, -1)), dim=1))))
        scores = scores.reshape(self.seq, self.mid_size)
        atten_weights = nn.functional.softmax(scores, dim=0)
        context = (atten_weights * oute.reshape(self.seq, -1)).sum().reshape(1, 1)

        h = self.decoder(context, torch.zeros(1, 1))
        for i in range(self.decode_length - 1):
            h = self.decoder(h, h)
        return torch.sigmoid(h)


def train_atten():
    attention = Encode_decode_atten()
    lossfunc = torch.nn.BCELoss()
    optim = torch.optim.Adam(attention.parameters(), 0.01)
    for wpoch in range(8):
        bar = pbar(1, len(train_data), True)
        for title, text, target in train_data:
            input = torch.cat((title, text))
            pred = attention(input)
            loss = lossfunc(pred.reshape(-1), target)
            attention.zero_grad()
            optim.zero_grad()
            loss.backward()
            optim.step()
            bar.frac()
        attention.eval()
        count = 0
        for val_title, val_text, val_target in val_data:
            with torch.no_grad():
                if torch.round(attention(torch.cat((val_title, val_text)))) == val_target:
                    count += 1
        attention.train()
        print(count / len(val_data))
        print(loss)


# train_atten()

class Lineardiscrete(nn.Module):
    def __init__(self, in_shape1=400, mid=100, in_shape2=150, out=1):
        super().__init__()
        # self.embed=nn.Embedding(len(vocab)+1,10)
        self.net1 = nn.Sequential(
            nn.Linear(in_shape1, mid),
            nn.ReLU(),
            nn.Linear(mid, out)
        )
        self.net2 = nn.Sequential(
            nn.Linear(in_shape2, mid),
            nn.ReLU(),
            nn.Linear(mid, out)
        )
        self.fc_combine = nn.Linear(2, 1)

    def forward(self, x=None, x2=None):
        if x2 is None:
            return torch.sigmoid(self.net1(x))
        elif x is None:
            return torch.sigmoid(self.net2(x2))
        else:
            out1 = torch.sigmoid(self.net1(x))
            out2 = torch.sigmoid(self.net2(x2))
            return torch.sigmoid(self.fc_combine(torch.cat((out1, out2))))


def is_zero(x):
    return x.sum() == 0


def lineartrain():
    lmodel = Lineardiscrete()
    loss_func = nn.BCELoss()
    opt = torch.optim.Adam(lmodel.parameters(), 0.5)
    for w in range(100):
        bar = pbar(1, len(train_data), True)
        for title, text, target in train_data:
            if is_zero(title):
                pred = lmodel(x=text.float())
            elif is_zero(text):
                pred = lmodel(x2=title.float())
            else:
                pred = lmodel(text.float(), title.float())
            loss = loss_func(pred.reshape(-1), target.float())
            lmodel.zero_grad()
            opt.zero_grad()
            loss.backward()
            opt.step()
            bar.frac()
        lmodel.eval()
        count = 0
        with torch.no_grad():
            for val_title, val_text, val_target in val_data:
                if is_zero(val_title):
                    predi = lmodel(x=val_text.float())
                elif is_zero(val_text):
                    predi = lmodel(x2=val_title.float())
                else:
                    predi = lmodel(val_text.float(), val_title.float())
                if torch.round(predi.reshape(-1)) == val_target:
                    count += 1
        print(count / len(val_data))
        print(loss)
        lmodel.train()


# lineartrain()

class Better(nn.Module):
    def __init__(self, embed_dim=20, mid=100, out=1):
        super().__init__()
        self.embed = nn.Embedding(len(vocab) + 1, embed_dim)
        self.gru = nn.GRUCell(embed_dim, mid)
        self.fc = nn.Linear(mid, out)
        self.embed_dim = embed_dim

    def forward(self, input):
        words = [torch.mean(self.embed(word2tensor(i).long()), dim=0) for i in input.split(' ')]
        h = torch.zeros(1, 100)
        for i in words:
            if not isnan(i[0]):
                h = self.gru(i.reshape(1, -1), h)
            else:
                h = self.gru(torch.zeros(1, self.embed_dim), h)
        return torch.sigmoid(self.fc(h).reshape(-1))


split = 100
val_titles = list(titles)[:split]
val_text = list(text)[:split]
train_titles = list(titles)[split:]
train_labels = tensorlabel[split:]
train_text = list(text)[split:]
vali_labels=list(tensorlabel)[:split]

def traintitle(data, vals, tlabel, vald=None):
    if vald is None:
        vald = data
    model = Better()
    lof = torch.nn.BCELoss()
    o = torch.optim.Adam(model.parameters(), 0.003)
    for k in range(25):
        b = fbar(len(data))
        for i in range(len(data)):
            b.step()
            if not isnan(data[i]) and int(tlabel[i]) > -1:
                pred = model(data[i])
                loss = lof(pred, tlabel[i])
                model.zero_grad()
                o.zero_grad()
                loss.backward()
                o.step()
        # validation
        model.eval()
        count = 0
        bad = 0
        with torch.no_grad():
            for i in range(vals):
                pick = random.randint(0, len(vald) - 1)
                if not isnan(vald[pick]) and int(tensorlabel[pick]) > -1:
                    pre = model(vald[pick])
                    print(vald[pick], torch.round(pre), int(tensorlabel[pick]))
                    if torch.round(pre) == tensorlabel[pick]:
                        count += 1
                else:
                    bad += 1
        model.train()
        valid_points = vals - bad
        print(f'Validation accuracy: {count / valid_points}')
        print(loss, f' epoch: {k}')
        # input()


# traintitle(train_titles, len(val_titles), train_labels, val_titles)

class Betteratten(nn.Module):
    def __init__(self, embed_dim=20, mid=100, out=1):
        super().__init__()
        self.embed = nn.Embedding(len(vocab) + 1, embed_dim)
        self.gru = nn.GRUCell(embed_dim, mid)
        self.fc = nn.Linear(mid, out)
        self.embed_dim = embed_dim
        self.hfc = nn.Linear(2 * mid, mid)
        self.decoder = nn.GRU(mid, mid)

    def forward(self, input):
        words = [torch.mean(self.embed(word2tensor(i).long()), dim=0) for i in input.split(' ')]
        scores = []
        h = torch.zeros(1, 100)
        for i in words:
            if not isnan(i[0]):
                h = self.gru(i.reshape(1, -1), h)
            else:
                h = self.gru(torch.zeros(1, self.embed_dim), h)
            scores.append(h)
        w = torch.tensor([])
        for score in range(len(scores)) :
            w=self.hfc(torch.cat((scores[score], scores[-1]),dim=1))
        atten_w = nn.functional.softmax(w, dim=-1)
        context = atten_w.sum(dim=0)
        #print(len(scores),'\n')
        return torch.sigmoid(self.fc(context).reshape(-1))


def train_attten(data, vals, tlabel, vald=None):
    if vald is None:
        vald = data
    model = Betteratten()
    lof = torch.nn.BCELoss()
    o = torch.optim.Adam(model.parameters(), 0.003)
    for k in range(25):
        b = fbar(len(data))
        for i in range(len(data)):
            b.step()
            if not isnan(data[i]) and int(tlabel[i]) > -1:
                pred = model(data[i])
                loss = lof(pred, tlabel[i])
                model.zero_grad()
                o.zero_grad()
                loss.backward()
                o.step()
        # validation
        model.eval()
        count = 0
        bad = 0
        with torch.no_grad():
            for i in range(vals):
                pick = random.randint(0, len(vald) - 1)
                if not isnan(vald[pick]) and int(tensorlabel[pick]) > -1:
                    pre = model(vald[pick])
                    print(vald[pick], torch.round(pre), int(tensorlabel[pick]))
                    if torch.round(pre) == tensorlabel[pick]:
                        count += 1
                else:
                    bad += 1
        model.train()
        valid_points = vals - bad
        print(f'Validation accuracy: {count / valid_points}')
        print(loss, f' epoch: {k}')


#train_attten(train_text, len(val_text), train_labels, val_text)


class sideways(nn.Module):
  def __init__(self,embed=20,c1=10,c2=5,mid=100,out=1,kernal=3):
    super().__init__()
    self.convs=nn.Sequential(
        nn.Conv1d(embed,c1,kernal),
        nn.Conv1d(c1,c2,kernal)
    )
    self.embed=nn.EmbeddingBag(len(vocab)+1,embed,mode='sum',sparse=True)
    self.gru=nn.GRU(c2,mid)
    self.fc=nn.Linear(mid,out)
  def forward(self,ins):
    w=torch.stack([self.embed(word2tensor(i).unsqueeze(0)) for i in ins.split(' ')])
    print(w.size())
    features=self.convs(w).permute(2,0,1)#shape 1,c2,n
    _,h=self.gru(features)
    return torch.sigmoid(self.fc(h).reshape(-1))


def main():
    traintext(sideways, train_text, train_labels, val_text, vali_labels, 25, lr=0.003)
main()
