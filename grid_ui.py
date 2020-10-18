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
    return ''.join([str(i) + '\n' for i in y])


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
    results = list(search(search_term,num=10,start=0,stop=9))
    print('googling')
    if not reliable:
        return results
    else:
        print('fghjkdkhgfh')
        for i in results:
            print(check_if_reliable(i),i)
        return list(filter(lambda x: check_if_reliable(x), results))


def scraper(url):
    r = requests.get(url)
    soup = bs(r.content)
    titles = str(soup.findAll('h'))
    p = ''.join([str(i) for i in soup.findAll('p') if len(str(i)) > 100])
    return (titles, p)


def isurl(x):
    return 'https://' in str(x)


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
        container.grid()
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

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


# Function to test and see of the button is working to switch screens
# def qf(quickPrint):
#     print(quickPrint)

class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Main Page", font=Medium_FONT)
        intro_msg = '''Hello! Welcome to Verify-19! This is the source of REAL Covid-19 News Validation! Click on our URL checker and validate your news in a flash!
        We use an advanced neural network that cross-checks your news with trusted sources, like the Mayo Clinic and CDC. What are you waiting for? Verify away!'''
        intro = tk.Label(self, text=intro_msg)
        intro.grid()
        label.grid()

        button = ttk.Button(self, text="Url Checker",
                            command=lambda: controller.show_frame(PageOne))
        button.grid()

        button2 = ttk.Button(self, text="Instructions",
                             command=lambda: controller.show_frame(PageTwo))
        button2.grid()
        StartPage.configure(self, bg="#4f4848")


class PageOne(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Url Checker", font=LARGE_FONT)
        label.grid()

        button = ttk.Button(self, text="Back to Home Screen",
                            command=lambda: controller.show_frame(StartPage))
        button.grid()

        button2 = ttk.Button(self, text="Instructions",
                             command=lambda: controller.show_frame(PageTwo))

        button2.grid()
        self.title = ("Verify-19")
        self.entry = tk.Text(self, width=40, height=15, relief=GROOVE, borderwidth=3)
        self.entry.grid(column=0)
        self.button = tk.Button(self, text="Get", command=self.on_button)
        self.button.grid()
        self.var = IntVar()
        self.show_var = IntVar()
        self.firstcheckbutton = Checkbutton(self, text="Sort by reliability", variable=self.var, command=self.firstcb)
        self.url_space = tk.Text(self, height=20, width=60, borderwidth=2, relief=GROOVE)
        self.secondcheckbutton = Checkbutton(self, text="show google results", variable=self.show_var,
                                             command=self.second_cb)
        self.secondcheckbutton.grid()
        self.url_space.grid()
        self.firstcheckbutton.grid()
        self.result_label = Label(self, text='', bg='#4f4848')
        self.result_label.grid()
        self.status = Label(self, text='awaiting input')
        self.status.grid()
        self.query=None
        self.user_input = None
        PageOne.configure(self, bg="#4f4848")

    def on_button(self):
        user_input = self.entry.get('1.0', tk.END)
        print(user_input)
        if isurl(user_input):
            self.titles, texts = scraper(user_input)
            result = (text_model(texts) + title_model(self.titles)) / 2
        elif len(user_input.split(' ')) > 160:
            result = text_model(user_input)
        else:
            result = title_model(user_input)
        self.result_label.config(text=f'Probability of being true: {str(100*float(result))}%')
        if self.show_var.get():
            self.status.config(text='fetching reliable sources...')
            self.query = self.titles if isurl(user_input) else user_input
            print('something')
            #print(googler(self.query))
            sources = big_list2str(googler(self.query,self.show_var.get()))
            #print(self.query, sources)
            self.url_space.delete(1.0,'end')
            self.url_space.insert(1.0,sources)
            self.status.config(text='done, now awaiting further input')
            self.user_input = user_input

    def firstcb(self):
        print(str(self.var.get()))
        if self.query is not None:
            self.url_space.delete(1.0,'end')
            self.url_space.insert(1.0,big_list2str(googler(self.query, reliable=self.var.get())[:10]))

    def second_cb(self):
        if not self.show_var.get():
            self.url_space.config(text='')
        else:
            if self.user_input is not None:
                self.status.config(text='fetching reliable sources...')
                self.query = self.titles if isurl(self.user_input) else self.user_input
                sources = big_list2str(googler(self.query))
                print('check')
                print(self.query, sources)
                self.url_space.config(text=sources, bg='white', height=40, width=40)
                self.status.config(text='done, now awaiting further input')


class PageTwo(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = ttk.Label(self,
                          text="Instructions: Please insert the URL for your news that you would like to validate. Make sure that it is less than 6000 characters. Then click the get button and you will get your result! If you would like to sort by mentions, then check that box. It's that easy!",
                          font=LARGE_FONT)
        label.grid()

        button = ttk.Button(self, text="Go to Url Checker",
                            command=lambda: controller.show_frame(PageOne))
        button.grid()
        button2 = ttk.Button(self, text="Back to Home Page",
                             command=lambda: controller.show_frame(StartPage))
        button2.grid()

        PageTwo.configure(self, bg="#4f4848")


app = SampleApp()
app.mainloop()
