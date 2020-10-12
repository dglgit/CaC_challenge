import tkinter as tk
from tkinter import *
#Create a class for all widgets so the Model can access the inputs.
class SampleApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Verify-19")
        self.entry = tk.Entry(self, width=50)
        self.button = tk.Button(self, text="Get", command=self.on_button)
        self.button.pack()
        self.entry.pack()
        self.var = IntVar()
        self.firstcheckbutton = Checkbutton(text="Sort by mentions", variable=self.var, command=self.firstcb)
        self.firstcheckbutton.pack()
        """ self.secondcheckbutton = Checkbutton(text="Sort by mentions", variable=self.var, onvalue="true", offvalue="false",command=self.secondcb)
        self.secondcheckbutton.pack() """

#Function to get the input value in the Entry Box
    def on_button(self):
        str(print(self.entry.get()))
#Function to get the int value associated with the checkbox as a string
    def firstcb(self):
        str(print(self.var.get()))
    """ def secondcb(self):
        str(print(self.var.get())) """
app = SampleApp()
app.mainloop()

