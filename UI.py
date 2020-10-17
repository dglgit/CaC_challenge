import tkinter as tk
from tkinter import *
import tkinter.font as tkFont
from tkinter import ttk
LARGE_FONT = ("Verdana", 12)
Medium_FONT = ("Verdana", 10)

class SampleApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        # Creating a loop to load all of the pages when buttons are pressed.
        for F in (StartPage, PageTwo,PageOne):
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
        intro_msg='''Hello! Welcome to Verify-19! This is the source of REAL Covid-19 News Validation! Click on our URL checker and validate your news in a flash!
        We use an advanced neural network that cross-checks your news with trusted sources, like the Mayo Clinic and CDC. What are you waiting for? Verify away!'''
        intro=tk.Label(self, text=intro_msg)
        intro.pack()
        label.pack(pady=10, padx=10)

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

        button = ttk.Button(self, text="Back to Home Screen",
                           command=lambda: controller.show_frame(StartPage))
        button.pack()

        button2 = ttk.Button(self, text="Instructions",
                            command=lambda: controller.show_frame(PageTwo))

        button2.pack()
        self.title = ("Verify-19")
        self.entry = tk.Text(self, width=40, height=15, relief=GROOVE, borderwidth=3)
        self.entry.pack(side=tk.RIGHT)
        self.button = tk.Button(self, text="Get", command=self.on_button)
        self.button.pack()
        self.var = IntVar()
        self.firstcheckbutton = Checkbutton(self, text="Sort by mentions", variable=self.var, command=self.firstcb)
        self.url_space = tk.Label(self, text='', height=20, width=30, borderwidth=2, relief=GROOVE)
        self.url_space.pack(side=tk.LEFT)
        self.firstcheckbutton.pack()
        PageOne.configure(self, bg="#4f4848")



    def on_button(self):
        user_input = self.entry.get('1.0', tk.END)
        print(user_input)
        #print_to = tk.Label(text=[user_input, 'yeeet'])
        #print_to.pack()

    def firstcb(self):
        print(str(self.var.get()))


class PageTwo(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = ttk.Label(self, text="Instructions: Please insert the URL for your news that you would like to validate. Make sure that it is less than 6000 characters. Then click the get button and you will get your result! If you would like to sort by mentions, then check that box. It's that easy! , font=LARGE_FONT)

        button = ttk.Button(self, text="Go to Url Checker",
                           command=lambda: controller.show_frame(PageOne))
        button.pack()
        button2 = ttk.Button(self, text="Back to Home Page",
                            command=lambda: controller.show_frame(StartPage))
        button2.pack()

        PageTwo.configure(self, bg="#4f4848")

app = SampleApp()
app.mainloop()
