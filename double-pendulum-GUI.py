#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 17:59:44 2020

@author: michal
"""

import tkinter as tk
import numpy as np
import scipy.integrate as spi

from matplotlib.figure import Figure

from matplotlib.backends.backend_tkagg import ( FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.cm as cm
from matplotlib.colors import Normalize

import csv

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

def line_integral(r_0,r_1,func):
    partial_f = []
    for i in range(len(r_0)):
        partial_f.append( lambda l, r_0 : func(r_0[:i] + [l] + r_0[i:])[i] )
    my_res = 0
    my_err = 0
    for i in range(len(r_0)):
        cur_r = r_1[:i] + r_0[i+1:]
        cur_res, cur_err = spi.quad(partial_f[i], r_0[i], r_1[i], args=(cur_r))
        my_res += cur_res
        my_err += cur_err
    return((my_res, my_err))

theta_space = np.linspace(-7.0, 7.0, 40)
omega_space = np.linspace(10.0, -10.0, 30)

#static quantities
k = 0.0







class MyGUI:
    def __init__(self):
        self.win = tk.Tk()
        self.win.title("Double Pendulum Phase Portrait GUI")
        self.win.geometry('1280x720')
        self.win.resizable(False,False)
        self.win.update()
        self.w = self.win.winfo_width()
        self.h = self.win.winfo_height()
        
        #Colormap objects
        self.norm = Normalize()
        self.colormap = cm.inferno
        
        #Create layout

        self.dpi = 100
        self.fig = Figure(figsize=(self.w*0.78/self.dpi, self.h*0.6/self.dpi), dpi=self.dpi)
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.win)  # A tk.Draself.wingArea.
        self.canvas.draw()
        self.canvas.get_tk_widget().place(x=self.w*0.0, y=self.h*0.01)
        
        self.fig.canvas.callbacks.connect('button_press_event', self.on_click)
        
        #Parameter sliders
        self.theta_1 = 0.0
        self.omega_1 = 0.0
        self.theta_2 = 0.0
        self.omega_2 = 0.0
        
        self.param1 = tk.Scale(self.win, from_=-10, to=10, resolution=0.05, length=self.h*0.54, command=self.refreshParam)
        self.param1.place(x=self.w * 0.78, y=self.h*0.01)
        self.param1.update()
        
        self.param2 = tk.Scale(self.win, from_=-10, to=10, resolution=0.05, length=self.h*0.54, command=self.refreshParam)
        self.param2.place(x=self.w * 0.88, y=self.h*0.01)
        self.param2.update()
        
        # Parameter mapping init
        
        #param_map decides which two parameters are on the axes of the phase portrait. Other two are sliders
        #  0 = theta_1-omega_1, 1 = theta_1-theta_2, 2 = theta_1-omega_2, 3 = theta_2-omega_2, 4 = omega_1-omega_2, 5 = theta_2-omega_2
        self.set_param_map(5)
        
        self.param1text = tk.Label(self.win, text=self.param1label)
        self.param1text.place(x=self.w * 0.8, y=self.h * 0.57)
        self.param2text = tk.Label(self.win, text=self.param2label)
        self.param2text.place(x=self.w * 0.9, y=self.h * 0.57)
        
        
        
        #Parameter map radiobuttons
        self.param_map_label = tk.Label(self.win, text="Select dynamic variables")
        self.param_map_label.place(x=self.w * 0.01, y=self.h * 0.62)
        self.param_map_buttons = []
        self.param_map_var = tk.IntVar()
        self.param_map_text = [("theta_1 - omega_1", 0), ("theta_1 - theta_2", 1), ("theta_1 - omega_2", 2), ("theta_2 - omega_1", 3), ("omega_1 - omega_2", 4), ("theta_2 - omega_2", 5)]
        for i in range(len(self.param_map_text)):
            pm_text, pm_val = self.param_map_text[i]
            self.param_map_buttons.append(tk.Radiobutton(self.win, text=pm_text, variable=self.param_map_var, value=pm_val, command=self.change_param_map))
            self.param_map_buttons[-1].place(x=self.w * 0.01, y = self.h * (0.65 + i * 0.03))
        self.param_map_buttons[-1].select()
        
        #Visualisation style radiobuttons
        self.style = 0
        self.style_label = tk.Label(self.win, text="Select visual style")
        self.style_label.place(x=self.w * 0.01, y=self.h * 0.83)
        self.style_buttons = []
        self.style_var = tk.IntVar()
        self.style_text = [ ('Plain quiver', 0), ('Gradient quiver', 1), ('Gradient only', 2) ]
        for i in range(len(self.style_text)):
            st_text, st_val = self.style_text[i]
            self.style_buttons.append(tk.Radiobutton(self.win, text=st_text, variable=self.style_var, value=st_val, command=self.change_style))
            self.style_buttons[-1].place(x=self.w * 0.01, y = self.h * (0.86 + i * 0.03))
        self.style_buttons[0].select()
        
        #Dynamic parameters
        self.t = 0.0
        #self.dt = 0.01
        self.dynamic_param_label = tk.Label(self.win, text="Values of dynamic variables")
        self.dynamic_param_label.place(x=self.w * 0.16, y=self.h * 0.62)
        self.dynamic_param1_label = tk.Label(self.win, text="theta 2 = 0.0")
        self.dynamic_param1_label.place(x=self.w * 0.17, y=self.h * 0.65)
        self.dynamic_param2_label = tk.Label(self.win, text="omega 2 = 0.0")
        self.dynamic_param2_label.place(x=self.w * 0.17, y=self.h * 0.68)
        
        #Action buttons (start iterating, save plot)
        self.play = False
        self.is_writing = False
        self.tick_btn = tk.Button(self.win,text='Tick',command=self.tick_btn_listener, width = int(np.floor(self.w * 0.01)))
        self.tick_btn.place(x=self.w * 0.16, y=self.h * 0.71)
        self.tick_btn.update()
        self.play_btn = tk.Button(self.win,text='Play',command=self.play_btn_listener, width = int(np.floor(self.w * 0.01)))
        self.play_btn.place(x=self.w * 0.16, y=self.h * 0.75)
        self.play_btn.update()
        self.write_btn = tk.Button(self.win,text='Start writing',command=self.write_btn_listener, width = int(np.floor(self.w * 0.01)))
        self.write_btn.place(x=self.w * 0.16, y=self.h * 0.79)
        self.write_btn.update()
        
        #Global parameter sliders
        self.m_1 = 1.0
        self.m_2 = 1.0
        self.l_1 = 1.0
        self.l_2 = 1.0
        self.g = 9.8
        self.global_change = False
        self.m_1_slider = tk.Scale(self.win, from_=0.1, to=10.0, resolution=0.1, orient=tk.HORIZONTAL, length=self.w*0.4, command=self.refresh_global_param, label="m_1")
        self.m_1_slider.place(x=self.w * 0.5, y=self.h * 0.62)
        self.m_1_slider.set(1.0)
        self.m_2_slider = tk.Scale(self.win, from_=0.1, to=10.0, resolution=0.1, orient=tk.HORIZONTAL, length=self.w*0.4, command=self.refresh_global_param, label="m_2")
        self.m_2_slider.place(x=self.w * 0.5, y=self.h * 0.69)
        self.m_2_slider.set(1.0)
        self.l_1_slider = tk.Scale(self.win, from_=0.1, to=10.0, resolution=0.1, orient=tk.HORIZONTAL, length=self.w*0.4, command=self.refresh_global_param, label="l_1")
        self.l_1_slider.place(x=self.w * 0.5, y=self.h * 0.76)
        self.l_1_slider.set(1.0)
        self.l_2_slider = tk.Scale(self.win, from_=0.1, to=10.0, resolution=0.1, orient=tk.HORIZONTAL, length=self.w*0.4, command=self.refresh_global_param, label="l_2")
        self.l_2_slider.place(x=self.w * 0.5, y=self.h * 0.83)
        self.l_2_slider.set(1.0)
        self.g_slider = tk.Scale(self.win, from_=0.0, to=20.0, resolution=0.1, orient=tk.HORIZONTAL, length=self.w*0.4, command=self.refresh_global_param, label="g")
        self.g_slider.place(x=self.w * 0.5, y=self.h * 0.9)
        self.g_slider.set(9.8)
        
        #Open output text file
        self.data_output_file  = open('data_output.csv', mode='w')
        self.param_output_file = open('param_output.csv', mode='w')
        self.data_output_writer  = csv.writer(self.data_output_file,  delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        self.param_output_writer = csv.writer(self.param_output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        self.data_output_writer.writerow( ['time', 'theta_1', 'omega_1', 'theta_2', 'omega_2'])
        self.param_output_writer.writerow(['time', 'm_1', 'm_2', 'l_1', 'l_2', 'g'])
        
        #Trigger startup
        self.phase_portrait(0.0, 0.0)
        
        #Start listening for events
        
        self.win.mainloop()

    #------------------------------------------------------- 
    def phase_portrait(self, my_param1=-1, my_param2=-1):
        
        cur_dynam_param1, cur_dynam_param2 = self.set_static_param(self.param1.get(), self.param2.get())
        self.ax.clear()
        self.ax.set_xlabel(self.x_lab)
        self.ax.set_ylabel(self.y_lab)
        #q = quiver_portrait(theta_space, omega_space, self.double_pend_f, self.ax)
        if self.style != 2:
            x_space = np.linspace(-7.0, 7.0, 40)
            y_space = np.linspace(10.0, -10.0, 30)
        else:
            x_space = np.linspace(-7.0, 7.0, 80)
            y_space = np.linspace(10.0, -10.0, 60)
        x_results = []
        y_results = []
        if self.style != 0:
            colors = []
            def H_field(r):
                x_res, y_res = self.get_dynamic_res(r[0], r[1])
                return( [y_res, -x_res] )
            H_field_x = lambda x, y : H_field([x, y])[0]
            H_field_y = lambda y, x : H_field([x, y])[1]
            potential_y = []
            for i in range(len(y_space)):
                potential_y.append( spi.quad(H_field_y, 0.0, y_space[i], args=(0.0))[0] )
        for y_index in range(len(y_space)):
            y_val = y_space[y_index]
            x_results.append([])
            y_results.append([])
            if self.style != 0:
                colors.append([])
            for x_index in range(len(x_space)):
                x_val = x_space[x_index]
                x_res, y_res = self.get_dynamic_res(x_val, y_val)     
                x_results[-1].append(x_res)
                y_results[-1].append(y_res)
                if self.style != 0:
                    cur_potential_x = spi.quad(H_field_x, 0.0, x_val, args=(y_val))[0]
                #if self.style == 1:
                    #colors[-1].append(line_integral([0,0], [x_val, y_val], H_field))
                    colors[-1].append(cur_potential_x + potential_y[y_index])
        if self.style == 0:
            self.ax.quiver(x_space, y_space, x_results, y_results)
        elif self.style == 1:
            flat_colors = list(np.concatenate(colors).flatten('F'))
            self.norm.autoscale(flat_colors)
            self.ax.quiver(x_space, y_space, x_results, y_results, color=self.colormap(self.norm(flat_colors)))
        elif self.style == 2:
            self.ax.pcolormesh(colors, cmap = self.colormap)
        self.ax.plot(cur_dynam_param1, cur_dynam_param2, 'x')
        self.fig.tight_layout()
        self.canvas.draw()
        return 0,5
    
    def refreshParam(self, new_val=1.0):
        #This method is called whenever any of the params change. new_val is therefore useless
        self.phase_portrait(self.param1.get(), self.param2.get())
        #print(self.theta_1)
    
    def refresh_global_param(self, new_val=0.0):
        #An umbrella method for global param sliders
        self.global_change = True
        self.m_1 = self.m_1_slider.get()
        self.m_2 = self.m_2_slider.get()
        self.l_1 = self.l_1_slider.get()
        self.l_2 = self.l_2_slider.get()
        self.g   = self.g_slider.get()
        self.refreshParam()
    
    def on_click(self, event):
        if event.inaxes is not None:
            print(event.xdata, event.ydata)
            self.set_dynamic_param(event.xdata, event.ydata)
        else:
            print('Clicked ouside axes bounds but inside plot self.window')
    
    def set_param_map(self, new_param_map):
        self.param_map = new_param_map
        if new_param_map == 0:
            self.x_lab = "theta_1"
            self.y_lab = "omega_1"
            self.param1label = "theta_2"
            self.param2label = "omega_2"
            self.param1.set(self.theta_2)
            self.param2.set(self.omega_2)
        if new_param_map == 1:
            self.x_lab = "theta_1"
            self.y_lab = "theta_2"
            self.param1label = "omega_1"
            self.param2label = "omega_2"
            self.param1.set(self.omega_1)
            self.param2.set(self.omega_2)
        if new_param_map == 2:
            self.x_lab = "theta_1"
            self.y_lab = "omega_2"
            self.param1label = "theta_2"
            self.param2label = "omega_1"
            self.param1.set(self.theta_2)
            self.param2.set(self.omega_1)
        if new_param_map == 3:
            self.x_lab = "theta_2"
            self.y_lab = "omega_1"
            self.param1label = "theta_1"
            self.param2label = "omega_2"
            self.param1.set(self.theta_1)
            self.param2.set(self.omega_2)
        if new_param_map == 4:
            self.x_lab = "omega_1"
            self.y_lab = "omega_2"
            self.param1label = "theta_1"
            self.param2label = "theta_2"
            self.param1.set(self.theta_1)
            self.param2.set(self.theta_2)
        if new_param_map == 5:
            self.x_lab = "theta_2"
            self.y_lab = "omega_2"
            self.param1label = "theta_1"
            self.param2label = "omega_1"
            self.param1.set(self.theta_1)
            self.param2.set(self.omega_1)
    
    def change_param_map(self):
        self.set_param_map(self.param_map_var.get())
        self.param1text['text'] = self.param1label
        self.param2text['text'] = self.param2label
        dynam_1, dynam_2 = self.set_static_param()
        self.dynamic_param1_label['text'] = self.x_lab + " = " + str(dynam_1)
        self.dynamic_param2_label['text'] = self.y_lab + " = " + str(dynam_2)
        self.refreshParam()
    
    def set_static_param(self, val_1, val_2, backtrack = False):
        if backtrack:
            self.param1.set(val_1)
            self.param2.set(val_2)
        if self.param_map == 0:
            self.theta_2 = val_1
            self.omega_2 = val_2
            return self.theta_1, self.omega_1
        if self.param_map == 1:
            self.omega_1 = val_1
            self.omega_2 = val_2
            return self.theta_1, self.theta_2
        if self.param_map == 2:
            self.theta_2 = val_1
            self.omega_1 = val_2
            return self.theta_1, self.omega_2
        if self.param_map == 3:
            self.theta_1 = val_1
            self.omega_2 = val_2
            return self.theta_2, self.omega_1
        if self.param_map == 4:
            self.theta_1 = val_1
            self.theta_2 = val_2
            return self.omega_1, self.omega_2
        if self.param_map == 5:
            self.theta_1 = val_1
            self.omega_1 = val_2
            return self.theta_2, self.omega_2
    
    def set_dynamic_param(self, val_1, val_2):
        if self.param_map == 0:
            self.theta_1 = val_1
            self.omega_1 = val_2
        if self.param_map == 1:
            self.theta_1 = val_1
            self.theta_2 = val_2
        if self.param_map == 2:
            self.theta_1 = val_1
            self.omega_2 = val_2
        if self.param_map == 3:
            self.theta_2 = val_1
            self.omega_1 = val_2
        if self.param_map == 4:
            self.omega_1 = val_1
            self.omega_2 = val_2
        if self.param_map == 5:
            self.theta_2 = val_1
            self.omega_2 = val_2
        self.dynamic_param1_label['text'] = self.x_lab + " = " + str(val_1)
        self.dynamic_param2_label['text'] = self.y_lab + " = " + str(val_2)
        self.refreshParam()
    
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
    
    def change_style(self):
        self.style = self.style_var.get()
        self.refreshParam()
    
    #------------ SIMULATION LISTENERS --------------------
    
    def tick_btn_listener(self):
        t1, o1, t2, o2 = self.step()
        """self.theta_1 = t1
        self.omega_1 = o1
        self.theta_2 = t2
        self.omega_2 = o2"""
        #insert theta restrictions here
        """if t1 > 2*np.pi:
            t1 -= 2*np.pi
        if t1 < -2*np.pi:
            t1 += 2*np.pi
        if t2 > 2*np.pi:
            t2 -= 2*np.pi
        if t2 < -2*np.pi:
            t2 += 2*np.pi"""
        
        if self.param_map == 0:
            self.set_static_param(t2, o2, True)
            self.set_dynamic_param(t1, o1)
        if self.param_map == 1:
            self.set_static_param(o1, o2, True)
            self.set_dynamic_param(t1, t2)
        if self.param_map == 2:
            self.set_static_param(t2, o1, True)
            self.set_dynamic_param(t1, o2)
        if self.param_map == 3:
            self.set_static_param(t1, o2, True)
            self.set_dynamic_param(t2, o1)
        if self.param_map == 4:
            self.set_static_param(t1, t2, True)
            self.set_dynamic_param(o1, o2)
        if self.param_map == 5:
            self.set_static_param(t1, o1, True)
            self.set_dynamic_param(t2, o2)
        #self.refreshParam()
        
    def play_btn_listener(self):
        if self.play == True:
            #Enable all disabled interactive stuff
            self.tick_btn.configure(state = "normal")
            #Change the label
            self.play_btn['text'] = 'Play'
            self.play = False
        elif self.play == False:
            #Disable all the interactive stuff
            self.tick_btn.configure(state = "disabled")
            #Change the label
            self.play_btn['text'] = 'Pause'
            self.play = True
            self.play_func()
        
    def write_btn_listener(self):
        if self.is_writing == True:
            #Change the label
            self.write_btn['text'] = 'Start writing'
            self.is_writing = False
        elif self.is_writing == False:
            #Change the label
            self.write_btn['text'] = 'Stop writing'
            self.is_writing = True
    
    def play_func(self):
        self.tick_btn_listener()
        if self.play:
            self.win.after(10, self.play_func)
    
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
    
    def step(self):
        dt = 0.01
        k1a = dt*(self.omega_1)
        k1b = dt*self.get_epsilon_1(self.theta_1, self.omega_1, self.theta_2, self.omega_2)
        k1c = dt*(self.omega_2)
        k1d = dt*self.get_epsilon_2(self.theta_1, self.omega_1, self.theta_2, self.omega_2)
        
        k2a = dt*(self.omega_1 + k1b/2)
        k2b = dt*self.get_epsilon_1(self.theta_1 + k1a/2, self.omega_1 + k1b/2, self.theta_2 + k1c/2, self.omega_2 + k1d/2)
        k2c = dt*(self.omega_2 + k1d/2)
        k2d = dt*self.get_epsilon_2(self.theta_1 + k1a/2, self.omega_1 + k1b/2, self.theta_2 + k1c/2, self.omega_2 + k1d/2)
        
        k3a = dt*(self.omega_1 + k2b/2)
        k3b = dt*self.get_epsilon_1(self.theta_1 + k2a/2, self.omega_1 + k2b/2, self.theta_2 + k2c/2, self.omega_2 + k2d/2)
        k3c = dt*(self.omega_2 + k2d/2)
        k3d = dt*self.get_epsilon_2(self.theta_1 + k2a/2, self.omega_1 + k2b/2, self.theta_2 + k2c/2, self.omega_2 + k2d/2)
        
        k4a = dt*(self.omega_1 + k3b)
        k4b = dt*self.get_epsilon_1(self.theta_1 + k3a, self.omega_1 + k3b, self.theta_2 + k3c, self.omega_2 + k3d)
        k4c = dt*(self.omega_2 + k3d)
        k4d = dt*self.get_epsilon_2(self.theta_1 + k3a, self.omega_1 + k3b, self.theta_2 + k3c, self.omega_2 + k3d)
        
        theta_1_new = self.theta_1 + (k1a + 2*k2a + 2*k3a + k4a)/6
        omega_1_new = self.omega_1 + (k1b + 2*k2b + 2*k3b + k4b)/6
        theta_2_new = self.theta_2 + (k1c + 2*k2c + 2*k3c + k4c)/6
        omega_2_new = self.omega_2 + (k1d + 2*k2d + 2*k3d + k4d)/6
        self.t += dt
        
        if self.is_writing:
            if self.global_change:
                self.param_output_writer.writerow([self.t, self.m_1, self.m_2, self.l_1, self.l_2, self.g])
                self.global_change = False
            self.data_output_writer.writerow([self.t, theta_1_new, omega_1_new, theta_2_new, omega_2_new])
        
        return(theta_1_new, omega_1_new, theta_2_new, omega_2_new)


#------------------------------------------------------------------        
if __name__ =='__main__':   
    MyApp=MyGUI()