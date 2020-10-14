import tkinter as tk
from tkinter import *
import tkinter.font as tkFont

LARGE_FONT = ("Verdana", 12)


class SampleApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = { }
        # Creating a loop to load all of the pages when buttons are pressed.
        for F in (StartPage, PageOne, PageTwo):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()


# Function to test and see of the button is working to switch screens
# def qf(quickPrint):
#     print(quickPrint)

class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Main Page", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button = tk.Button(self, text="Url Checker",
                           command=lambda: controller.show_frame(PageOne))
        button.pack()

        button2 = tk.Button(self, text="Instructions",
                            command=lambda: controller.show_frame(PageTwo))
        button2.pack()


class PageOne(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Url Checker", font=LARGE_FONT)

        button = tk.Button(self, text="Back to Home Screen",
                           command=lambda: controller.show_frame(StartPage))
        button.pack()

        button2 = tk.Button(self, text="Instructions",
                            command=lambda: controller.show_frame(PageTwo))

        button2.pack()

    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Verify-19")
        self.entry = tk.Text(self, width=20, height=20, relief=GROOVE, borderwidth=3)
        self.entry.pack(side=tk.RIGHT)
        self.button = tk.Button(self, text="Get", command=self.on_button)
        self.button.pack()
        self.var = IntVar()
        self.firstcheckbutton = Checkbutton(text="Sort by mentions", variable=self.var, command=self.firstcb)
        self.firstcheckbutton.pack()
        self.url_space = tk.Label(text='', height=30, width=14, borderwidth=2, relief=GROOVE)
        self.url_space.pack(side=tk.LEFT)

    def on_button(self):
        user_input = self.entry.get('1.0', tk.END)
        print(user_input)
        print_to = tk.Label(text=[user_input, 'yeeet'])
        print_to.pack()

    def firstcb(self):
        str(print(self.var.get()))



class PageTwo(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Url Checker", font=LARGE_FONT)

        button = tk.Button(self, text="Go to Url Checker",
                           command=lambda: controller.show_frame(PageOne))
        button.pack()
        button2 = tk.Button(self, text="Back to Home Page",
                            command=lambda: controller.show_frame(StartPage))
        button2.pack()


app = SampleApp()
app.mainloop()
