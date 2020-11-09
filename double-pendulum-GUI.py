#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 17:59:44 2020

@author: michal
"""

import tkinter as tk
import numpy as np

from matplotlib.figure import Figure

from matplotlib.backends.backend_tkagg import ( FigureCanvasTkAgg, NavigationToolbar2Tk)

def quiver_portrait(x_space, v_space, a_func, ax):
    v_results = []
    a_results = []

    for v_val in v_space:
        v_results.append([])
        a_results.append([])
        for x_val in x_space:
            v_results[-1].append(v_val)
            a_results[-1].append(a_func(x_val, v_val))

    #fig = plt.figure(figsize=(25, 12))
    #ax = fig.add_subplot(1, 1, 1)
    q = ax.quiver(x_space, v_space, v_results, a_results)
    return(q)


theta_space = np.linspace(-7.0, 7.0, 40)
omega_space = np.linspace(10.0, -10.0, 30)

#static quantities
k = 0.0







class MyGUI:
    def __init__(self):
        win = tk.Tk()
        win.title("Double Pendulum Phase Portrait GUI")
        win.geometry('1280x720')
        win.resizable(False,False)
        win.update()
        self.w = win.winfo_width()
        self.h = win.winfo_height()
        
        
        #Create layout

        self.fig = Figure(figsize=(self.w*0.78/100, self.h*0.6/100), dpi=100)
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=win)  # A tk.DrawingArea.
        self.canvas.draw()
        self.canvas.get_tk_widget().place(x=self.w*0.0, y=self.h*0.01)
        
        self.fig.canvas.callbacks.connect('button_press_event', self.on_click)
        # Parameter mapping init
        
        #param_map decides which two parameters are on the axes of the phase portrait. Other two are sliders
        #  0 = theta_1-omega_1, 1 = theta_1-theta_2, 2 = theta_1-omega_2, 3 = theta_2-omega_2, 4 = omega_1-omega_2, 5 = theta_2-omega_2
        self.set_param_map(5)
        
        #Parameter sliders
        
        self.param1 = tk.Scale(win, from_=-10, to=10, resolution=0.05, length=self.h*0.54, command=self.refreshParam)
        self.param1.place(x=self.w * 0.78, y=self.h*0.01)
        self.param1.update()
        
        self.param2 = tk.Scale(win, from_=-10, to=10, resolution=0.05, length=self.h*0.54, command=self.refreshParam)
        self.param2.place(x=self.w * 0.88, y=self.h*0.01)
        self.param2.update()
        
        self.param1text = tk.Label(win, text=self.param1label)
        self.param1text.place(x=self.w * 0.8, y=self.h * 0.57)
        self.param2text = tk.Label(win, text=self.param2label)
        self.param2text.place(x=self.w * 0.9, y=self.h * 0.57)
        
        #Parameter map radiobuttons
        self.param1text = tk.Label(win, text="Select dynamic variables")
        self.param1text.place(x=self.w * 0.05, y=self.h * 0.62)
        self.param_map_buttons = []
        self.param_map_var = tk.IntVar()
        self.param_map_text = [("theta_1 - omega_1", 0), ("theta_1 - theta_2", 1), ("theta_1 - omega_2", 2), ("theta_2 - omega_1", 3), ("omega_1 - omega_2", 4), ("theta_2 - omega_2", 5)]
        for i in range(len(self.param_map_text)):
            pm_text, pm_val = self.param_map_text[i]
            self.param_map_buttons.append(tk.Radiobutton(win, text=pm_text, variable=self.param_map_var, value=pm_val, command=self.change_param_map))
            self.param_map_buttons[-1].place(x=self.w * 0.05, y = self.h * (0.65 + i * 0.03))
        self.param_map_buttons[-1].select()
        
        #Action buttons (start iterating, save plot)
        
        
        #Global parameter sliders
        self.m_1 = 1.0
        self.m_2 = 1.0
        self.l_1 = 1.0
        self.l_2 = 1.0
        self.g = 9.8
        self.m_1_slider = tk.Scale(win, from_=0.1, to=10.0, resolution=0.1, orient=tk.HORIZONTAL, length=self.w*0.4, command=self.refreshParam, label="m_1")
        self.m_1_slider.place(x=self.w * 0.5, y=self.h * 0.62)
        self.m_1_slider.set(1.0)
        self.m_2_slider = tk.Scale(win, from_=0.1, to=10.0, resolution=0.1, orient=tk.HORIZONTAL, length=self.w*0.4, command=self.refreshParam, label="m_2")
        self.m_2_slider.place(x=self.w * 0.5, y=self.h * 0.69)
        self.m_2_slider.set(1.0)
        self.l_1_slider = tk.Scale(win, from_=0.1, to=10.0, resolution=0.1, orient=tk.HORIZONTAL, length=self.w*0.4, command=self.refreshParam, label="l_1")
        self.l_1_slider.place(x=self.w * 0.5, y=self.h * 0.76)
        self.l_1_slider.set(1.0)
        self.l_2_slider = tk.Scale(win, from_=0.1, to=10.0, resolution=0.1, orient=tk.HORIZONTAL, length=self.w*0.4, command=self.refreshParam, label="l_2")
        self.l_2_slider.place(x=self.w * 0.5, y=self.h * 0.83)
        self.l_2_slider.set(1.0)
        self.g_slider = tk.Scale(win, from_=0.0, to=20.0, resolution=0.1, orient=tk.HORIZONTAL, length=self.w*0.4, command=self.refreshParam, label="g")
        self.g_slider.place(x=self.w * 0.5, y=self.h * 0.9)
        self.g_slider.set(9.8)
        
        #Trigger startup
        self.phase_portrait(0.0, 0.0)
        
        #Start listening for events
        
        win.mainloop()

    #------------------------------------------------------- 
    def phase_portrait(self, my_param1=-1, my_param2=-1):
        
        self.set_static_param()
        self.ax.clear()
        self.ax.set_xlabel(self.x_lab)
        self.ax.set_ylabel(self.y_lab)
        #q = quiver_portrait(theta_space, omega_space, self.double_pend_f, self.ax)
        x_space = np.linspace(-4.0, 4.0, 40)
        y_space = np.linspace(10.0, -10.0, 30)
        x_results = []
        y_results = []

        for y_val in y_space:
            x_results.append([])
            y_results.append([])
            for x_val in x_space:
                #self.theta_2 = x_val
                #self.omega_2 = v_val
                x_res, y_res = self.get_dynamic_res(x_val, y_val)
                
                x_results[-1].append(x_res)
                y_results[-1].append(y_res)
                
                #fig = plt.figure(figsize=(25, 12))
                #ax = fig.add_subplot(1, 1, 1)
        q = self.ax.quiver(x_space, y_space, x_results, y_results)
        self.fig.tight_layout()
        self.canvas.draw()
        return 0,5
    
    def refreshParam(self, new_val=1.0):
        #This method is called whenever the user moves any slider. new_val is therefore useless
        self.m_1 = self.m_1_slider.get()
        self.m_2 = self.m_2_slider.get()
        self.l_1 = self.l_1_slider.get()
        self.l_2 = self.l_2_slider.get()
        self.g   = self.g_slider.get()
        self.phase_portrait(self.param1.get(), self.param2.get())
        #print(self.theta_1)
    
    def on_click(self, event):
        if event.inaxes is not None:
            print(event.xdata, event.ydata)
        else:
            print('Clicked ouside axes bounds but inside plot window')
    
    def set_param_map(self, new_param_map):
        self.param_map = new_param_map
        if new_param_map == 0:
            self.x_lab = "theta_1"
            self.y_lab = "omega_1"
            self.param1label = "theta_2"
            self.param2label = "omega_2"
        if new_param_map == 1:
            self.x_lab = "theta_1"
            self.y_lab = "theta_2"
            self.param1label = "omega_1"
            self.param2label = "omega_2"
        if new_param_map == 2:
            self.x_lab = "theta_1"
            self.y_lab = "omega_2"
            self.param1label = "theta_2"
            self.param2label = "omega_1"
        if new_param_map == 3:
            self.x_lab = "theta_2"
            self.y_lab = "omega_1"
            self.param1label = "theta_1"
            self.param2label = "omega_2"
        if new_param_map == 4:
            self.x_lab = "omega_1"
            self.y_lab = "omega_2"
            self.param1label = "theta_1"
            self.param2label = "theta_2"
        if new_param_map == 5:
            self.x_lab = "theta_2"
            self.y_lab = "omega_2"
            self.param1label = "theta_1"
            self.param2label = "omega_1"
    
    def change_param_map(self):
        self.set_param_map(self.param_map_var.get())
        self.param1text['text'] = self.param1label
        self.param2text['text'] = self.param2label
        self.refreshParam()
    
    def set_static_param(self):
        if self.param_map == 0:
            self.theta_2 = self.param1.get()
            self.omega_2 = self.param2.get()
        if self.param_map == 1:
            self.omega_1 = self.param1.get()
            self.omega_2 = self.param2.get()
        if self.param_map == 2:
            self.theta_2 = self.param1.get()
            self.omega_1 = self.param2.get()
        if self.param_map == 3:
            self.theta_1 = self.param1.get()
            self.omega_2 = self.param2.get()
        if self.param_map == 4:
            self.theta_1 = self.param1.get()
            self.theta_2 = self.param2.get()
        if self.param_map == 5:
            self.theta_1 = self.param1.get()
            self.omega_1 = self.param2.get()
            #print("kek")
    
    def get_dynamic_res(self, val_1, val_2):
        if self.param_map == 0:
            #self.theta_1 = self.val_1
            #self.omega_1 = self.val_2
            x_res = val_2
            y_res = self.get_epsilon_1(val_1, val_2, self.theta_2, self.omega_2)
        if self.param_map == 1:
            #self.theta_1 = self.val_1
            #self.theta_2 = self.val_2
            x_res = self.omega_1
            y_res = self.omega_2
        if self.param_map == 2:
            #self.theta_1 = self.val_1
            #self.omega_2 = self.val_2
            x_res = self.omega_1
            y_res = self.get_epsilon_2(val_1, self.omega_1, self.theta_2, val_2)
        if self.param_map == 3:
            #self.theta_2 = self.val_1
            #self.omega_1 = self.val_2
            x_res = self.omega_2
            y_res = self.get_epsilon_1(self.theta_1, val_2, val_1, self.omega_2)
        if self.param_map == 4:
            #self.omega_1 = self.val_1
            #self.omega_2 = self.val_2
            x_res = self.get_epsilon_1(self.theta_1, val_1, self.theta_2, val_2)
            y_res = self.get_epsilon_2(self.theta_1, val_1, self.theta_2, val_2)
        if self.param_map == 5:
            #self.theta_2 = self.val_1
            #self.omega_2 = self.val_2
            x_res = val_2
            y_res = self.get_epsilon_2(self.theta_1, self.omega_1, val_1, val_2)
        return(x_res, y_res)
    
    #-------------- PHYSICS METHODS -----------------------
    
    #def double_pend_f(self, theta_2, omega_2):
    #    
    
    def get_epsilon_1(self, my_theta_1, my_omega_1, my_theta_2, my_omega_2):
        numerator = self.m_2*self.l_1*np.sin(my_theta_1-my_theta_2)*(my_omega_2**2+my_omega_1**2*np.cos(my_theta_1-my_theta_2))+self.g*(self.m_1*np.sin(my_theta_1)+self.m_2*np.cos(my_theta_2)*np.sin(my_theta_1-my_theta_2))
        denominator = self.l_1*(self.m_2*np.cos(my_theta_1-my_theta_2)*np.cos(my_theta_1-my_theta_2)-(self.m_1+self.m_2))
        return(numerator/denominator)
    
    def get_epsilon_2(self, my_theta_1, my_omega_1, my_theta_2, my_omega_2):
        numerator = (self.m_1+self.m_2)*self.l_1*my_omega_1**2+self.m_2*self.l_2*my_omega_2**2*np.cos(my_theta_1-my_theta_2)+(self.m_1+self.m_2)*self.g*np.cos(my_theta_1)*np.sin(my_theta_1-my_theta_2)
        denominator = self.l_2*((self.m_1+self.m_2)-self.m_2*np.cos(my_theta_1-my_theta_2)*np.cos(my_theta_1-my_theta_2))
        return(numerator/denominator)


#------------------------------------------------------------------        
if __name__ =='__main__':   
    MyApp=MyGUI()