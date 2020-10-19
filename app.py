import tkinter as tk
from tkinter import *
import tkinter.font as tkFont
from tkinter import ttk
import googlesearch as gs
from googlesearch import search
from bs4 import BeautifulSoup as bs
import requests
import torch

device = torch.device('cpu')
with open('./Data_and_models/stopwords.txt') as stops:
    for i in stops:
        stopwords = eval(i)


def remove_stops(x, tolist=False):
    if tolist:
        return list(filter(lambda x: x not in stopwords, x))
    else:
        return filter(lambda x: x not in stopwords, x)


def big_list2str(x):
    y = x
    return ''.join([str(i) + '\n\r' for i in y])

# deprecated
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


reliable_endings = ['.org', '.int', '.gov', 'PolitiFact', '.us', '.edu']


def check_if_reliable(x):
    reliable_endings = ['.org', '.int', '.gov', 'PolitiFact', '.us', '.edu']
    for i in reliable_endings:
        if i in x:
            return True
    return False


def googler(search_term, reliable=False, score=False):
    try:
        results = list(search(search_term,num=10,start=0,stop=9))
    except:
        return None
    print('googling')
    if not reliable:
        return results
    else:
        print('fghjkdkhgfh')
        for i in results:
            print(check_if_reliable(i),i)
        return list(filter(lambda x: check_if_reliable(x), results))


def scraper(url):
    r = requests.get(str(url))
    soup = bs(r.content)
    titles = str(soup.findAll('h'))
    p = ''.join([str(i) for i in soup.findAll('p') if len(str(i)) > 100])
    metas=None
    for i in soup.findALl('meta'):
        if i.attrs['name']=='description':
            metas=i.attrs['content']
            break
    return (titles, p, metas)


def isurl(x):
    return 'https://' in str(x) or 'http://' in str(x)


import torch.nn as nn
from collections import defaultdict
import string

device = torch.device('cpu')

letters = string.ascii_letters + ' !0123456789?'


def enum1(x):
    for i in range(len(x)):
        yield (i + 1, x[i])


vocab = {letter: i for i, letter in enum1(letters)}
vocab = defaultdict(lambda: 0, vocab)

with open('./Data_and_models/stopwords.txt') as stops:
    for i in stops:
        stopwords = eval(i)


def word2tensor(word):
    tens = [vocab[i] for i in word]
    return torch.tensor(tens) if len(tens) > 0 else torch.tensor([0])


def remove_stops(x, tolist=False):
    if tolist:
        return list(filter(lambda x: x not in stopwords, x))
    else:
        return filter(lambda x: x not in stopwords, x)

import math
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        super().__init__()
        self.d_model = d_model

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        x = x+self.pe[:, :seq_len]
        return x


class Thiccatten(nn.Module):
    def __init__(self, k, heads):
        super().__init__()
        self.qw = nn.Linear(k, k * heads)
        self.kw = nn.Linear(k, k * heads)
        self.vw = nn.Linear(k, k * heads)
        self.fc = nn.Linear(k * heads, k)
        self.heads = heads

    def forward(self, x):
        b, t, k = x.size()
        h = self.heads
        q = self.qw(x).view(b, t, h, k)
        key = self.kw(x).view(b, t, h, k)
        v = self.vw(x).view(b, t, h, k)
        keys = key.transpose(1, 2).contiguous().view(b * h, t, k)
        queries = q.transpose(1, 2).contiguous().view(b * h, t, k)
        values = v.transpose(1, 2).contiguous().view(b * h, t, k)
        keys = keys / (k ** 0.25)
        queries = queries / (k ** 0.25)
        dot = torch.bmm(keys, queries.transpose(1, 2))
        scaled_dot = torch.softmax(dot, dim=2)
        out = torch.bmm(scaled_dot, values).view(b, h, t, k)
        out = out.transpose(1, 2).contiguous().view(b, t, k * h)
        return self.fc(out)


class tblock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()

        self.attention = Thiccatten(k, heads=heads)

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.ff = nn.Sequential(
            nn.Linear(k, 4 * k),
            nn.ReLU(),
            nn.Linear(4 * k, k))

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)

        fedforward = self.ff(x)
        return self.norm2(fedforward + x)


class c_transformer(nn.Module):
    def __init__(self, heads=8, depth=7, word_embed=20, max_seq=6000):
        super().__init__()
        self.transformers = nn.Sequential(*[tblock(word_embed, heads) for i in range(depth)])
        self.w_embed = nn.EmbeddingBag(len(vocab) + 1, word_embed)
        self.pos_embed = nn.Embedding(max_seq + 1, word_embed)
        self.fc = nn.Linear(word_embed, 1)

    def forward(self, x):
        w = torch.stack(
            [self.w_embed(word2tensor(i).unsqueeze(0), None) for i in remove_stops(x.split(' '))]).transpose(0, 1).to(
            device)
        b, t, k = w.size()
        pos_embeddings = self.pos_embed(torch.arange(t)).expand(b, t, k)
        attended = self.transformers(pos_embeddings + w)
        classes = self.fc(attended).mean(dim=1)
        return torch.sigmoid(classes.reshape(-1))


class modified_transformer(nn.Module):
    def __init__(self, heads=8, depth=7, word_embed=20, max_seq=6000):
        super().__init__()
        self.transformers = nn.Sequential(*[tblock(word_embed, heads) for i in range(depth)])
        self.w_embed = nn.EmbeddingBag(len(vocab) + 1, word_embed)
        self.pos_embed = PositionalEncoder(word_embed, max_seq)
        self.fc = nn.Linear(word_embed, 1)

    def forward(self, x):
        w = torch.stack(
            [self.w_embed(word2tensor(i).unsqueeze(0), None) for i in remove_stops(x.split(' '))]).transpose(0, 1).to(
            device)
        b, t, k = w.size()
        attended = self.transformers(self.pos_embed(w))
        classes = self.fc(attended).mean(dim=1)
        return torch.sigmoid(classes.reshape(-1))


text_model = torch.load('./Data_and_models/model80')
title_model = torch.load('./Data_and_models/title_model85')
text_model.eval()
title_model.eval()
LARGE_FONT = ("TMR", 12)
Medium_FONT = ("TMR", 10)


class SampleApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        Tk.geometry(self, '690x700')

        self.frames = {}
        # Creating a loop to load all of the pages when buttons are pressed.
        for F in (StartPage, PageTwo, PageOne):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        print(cont.__name__)
        frame.tkraise()
        SampleApp.configure(self, bg="#4f4848")




class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Main Page", font=Medium_FONT)
        intro_msg = '''Hello! Welcome to Verify-19! This is the source of REAL Covid-19 News Validation! Click on our URL checker and validate your news in a flash!
        We use state of the art deep neural networks that makes a prediction on if your news is real or fake. What are you waiting for? Verify away!'''
        intro = tk.Label(self, text=intro_msg, wraplength=550)
        intro.pack()


        button = ttk.Button(self, text="Url Checker",
                            command=lambda: controller.show_frame(PageOne))
        button.pack()

        button2 = ttk.Button(self, text="Instructions",
                             command=lambda: controller.show_frame(PageTwo))
        button2.pack()
        StartPage.configure(self, bg="#4f4848")


class PageOne(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Url Checker", font=LARGE_FONT)
        label.pack()

        button = ttk.Button(self, text="Back to Home Screen",
                            command=lambda: controller.show_frame(StartPage))
        button.pack()

        button2 = ttk.Button(self, text="Instructions",
                             command=lambda: controller.show_frame(PageTwo))

        button2.pack(pady=2)
        self.title = ("Verify-19")
        self.entry = tk.Text(self, width=60, height=15, relief=GROOVE, borderwidth=3)
        self.clear_button = tk.Button(self, text='clear', command=self.clear)
        self.entry.pack(anchor=tk.CENTER)
        self.button = tk.Button(self, text="Submit", command=self.on_button,width=10)
        self.button.pack(anchor=CENTER,pady=4)
        self.clear_button.pack(pady=3)
        self.var = IntVar()
        self.show_var = IntVar()
        self.firstcheckbutton = Checkbutton(self, text="Sort by reliability", variable=self.var, command=self.firstcb)
        self.secondcheckbutton = Checkbutton(self, text="show google results", variable=self.show_var,
                                             command=self.second_cb)
        self.secondcheckbutton.pack(anchor=N)
        self.firstcheckbutton.pack(pady=2)
        self.result_label = Label(self, text='', bg='#4f4848')
        self.status = Label(self, text='awaiting input')
        self.status.pack()
        self.result_label.pack(side=TOP)
        self.query=None
        self.user_input = None

        PageOne.configure(self, bg="#4f4848")
    def clear(self):
        self.entry.delete(1.,'end')

    def on_button(self):
        user_input = self.entry.get('1.0', tk.END)
        display=True
        if isurl(user_input):
            try:
                self.titles, texts , metas= scraper(user_input)
                result = (text_model(texts) + title_model(self.titles)) / 2
                print(self.titles,metas)
            except:
                self.result_label.config(text='url retrieval failed, sorry :(')
                display=False
        elif len(user_input.split(' ')) > 160:
            result = text_model(user_input)
        else:
            result = title_model(user_input)
        if display:
            self.result_label.config(text=f'Probability of being true: {str(float(result))}%', fg='#d9ced0')
            if self.show_var.get():
                self.status.config(text='fetching reliable sources...')
                self.query = self.titles if isurl(user_input) else user_input
                print('something')
                sources = googler(self.query,self.var.get())
                self.entry.delete(1.0,'end')
                if sources is None:
                    self.entry.insert(1.0,'google search failed, try again later :(')
                else:
                    self.entry.insert(1.0,big_list2str(sources))
                self.status.config(text='done, now awaiting further input')
                self.user_input = user_input


    def firstcb(self):
        print(str(self.var.get()))
        if self.query is not None:
            self.entry.delete(1.0,'end')
            self.entry.insert(1.0,big_list2str(googler(self.query, reliable=self.var.get())[:10]))

    def second_cb(self):
        if not self.show_var.get():
            pass
        else:
            if self.user_input is not None:
                self.status.config(text='fetching reliable sources...')
                self.query = self.titles if isurl(self.user_input) else self.user_input
                sources = big_list2str(googler(self.query))
                print('check')
                print(self.query, sources)
                self.entry.config(text=sources, bg='white', height=40, width=40)
                self.status.config(text='done, now awaiting further input')


class PageTwo(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = ttk.Label(self,
                          text="Instructions: Please insert the URL for your news that you would like to validate. Make sure that it is less than 6000 characters. Then click the get button and you will get your result! If you would like to sort by mentions, then check that box. It's that easy!",
                          font=LARGE_FONT,
                          wraplength=690)
        label.pack()

        button = ttk.Button(self, text="Go to Url Checker",
                            command=lambda: controller.show_frame(PageOne))
        button.pack()
        button2 = ttk.Button(self, text="Back to Home Page",
                             command=lambda: controller.show_frame(StartPage))
        button2.pack()

        PageTwo.configure(self, bg="#4f4848")


app = SampleApp()
app.mainloop()

