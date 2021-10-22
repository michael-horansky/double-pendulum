#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 17:59:44 2020

@author: michal
"""

import tkinter as tk
import numpy as np
import scipy.integrate as spi
import statistics

from matplotlib.figure import Figure

from matplotlib.backends.backend_tkagg import ( FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.cm as cm
from matplotlib.colors import Normalize

import csv

import io
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc

from pathlib import Path


# ------------- BASIC FUNCTIONS -------------

def nonzero_sign(x):
    if x >= 0.0:
        return(1)
    else:
        return(-1)

def base_theta(x):
    res = x
    while res <= - np.pi:
        res += 2.0 * np.pi
    while res > np.pi:
        res -= 2.0 * np.pi
    return(res)




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

def fig_to_img(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

theta_space = np.linspace(-7.0, 7.0, 40)
omega_space = np.linspace(10.0, -10.0, 30)

#static quantities
k = 0.0


# --------------- CLASSES -------------

class popupWindow(object):
    def __init__(self,master):
        top=self.top=tk.Toplevel(master)
        self.l=tk.Label(top,text="Enter the filename (without extension)")
        self.l.pack()
        self.e=tk.Entry(top)
        self.e.pack()
        self.b=tk.Button(top,text='Submit',command=self.cleanup)
        self.b.pack()
    def cleanup(self):
        self.value=self.e.get()
        self.top.destroy()



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
        self.fig = Figure(figsize=(self.w*0.58/self.dpi, self.h*0.6/self.dpi), dpi=self.dpi)
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
        
        self.traces = []
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        """self.param1 = tk.Scale(self.win, from_=-10, to=10, resolution=0.05, length=self.h*0.54, command=self.refreshParam)
        self.param1.place(x=self.w * 0.78, y=self.h*0.01)
        self.param1.update()
        
        self.param2 = tk.Scale(self.win, from_=-10, to=10, resolution=0.05, length=self.h*0.54, command=self.refreshParam)
        self.param2.place(x=self.w * 0.88, y=self.h*0.01)
        self.param2.update()"""
        self.param1 = tk.Scale(self.win, from_=-10, to=10, resolution=0.05, orient=tk.HORIZONTAL, length=self.h*0.54, command=self.refreshParam, label="theta_1")
        self.param1.place(x=self.w * 0.4, y=self.h*0.62)
        self.param1.update()
        
        self.param2 = tk.Scale(self.win, from_=-10, to=10, resolution=0.05, orient=tk.HORIZONTAL, length=self.h*0.54, command=self.refreshParam, label="omega_1")
        self.param2.place(x=self.w * 0.4, y=self.h*0.69)
        self.param2.update()
        
        # Parameter mapping init
        
        #param_map decides which two parameters are on the axes of the phase portrait. Other two are sliders
        #  0 = theta_1-omega_1, 1 = theta_1-theta_2, 2 = theta_1-omega_2, 3 = theta_2-omega_2, 4 = omega_1-omega_2, 5 = theta_2-omega_2
        self.set_param_map(5)
        
        """self.param1text = tk.Label(self.win, text=self.param1label)
        self.param1text.place(x=self.w * 0.8, y=self.h * 0.57)
        self.param2text = tk.Label(self.win, text=self.param2label)
        self.param2text.place(x=self.w * 0.9, y=self.h * 0.57)
        """
        
        # Driving params (treat as global params)
        self.F = 0.0
        self.omega_F = 0.0
        self.omega_F_range = 0.0
        self.driving_text = tk.Label(self.win, text="Parameters of the driving force; F(t) = F.sin(omega_F.t)")
        self.driving_text.place(x=self.w * 0.4, y=self.h * 0.76)
        
        self.F_slider = tk.Scale(self.win, from_=0, to=10, resolution=0.05, orient=tk.HORIZONTAL, length=self.h*0.54, command=self.refresh_global_param, label="F")
        self.F_slider.place(x=self.w * 0.4, y=self.h*0.79)
        self.F_slider.update()
        
        self.F_omega_slider = tk.Scale(self.win, from_=0, to=4.0, resolution=0.01, orient=tk.HORIZONTAL, length=self.w*0.15, command=self.refresh_global_param, label="omega_F")
        self.F_omega_slider.place(x=self.w * 0.4, y=self.h*0.86)
        self.F_omega_slider.update()
        self.F_range_slider = tk.Scale(self.win, from_=0, to=2.5, resolution=0.05, orient=tk.HORIZONTAL, length=self.w*0.15, command=self.refresh_global_param, label="omega_F range")
        self.F_range_slider.place(x=self.w * (0.4 + 0.15) + 5, y=self.h*0.86)
        self.F_range_slider.update()
        
        self.damping_analysis_btn = tk.Button(self.win,text='Damping analysis',command=self.damping_analysis_btn_listener, width = 14)
        self.damping_analysis_btn.place(x=self.w * 0.4, y=self.h * 0.93)
        self.damping_analysis_btn.update()
        
        self.damp_progress_label = tk.Label(self.win, text="No damping analysis in progress")
        self.damp_progress_label.place(x=self.w * 0.5, y=self.h * 0.94)
        
        
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
        
        #Load previous static parameters
        self.load_prev_params_btn = tk.Button(self.win,text='Load prev. params',command=self.load_prev_params_btn_listener, width = 14)
        self.load_prev_params_btn.place(x=self.w * 0.01, y=self.h * 0.95)
        self.load_prev_params_btn.update()
        
        #Dynamic parameters
        self.t = 0.0
        #self.dt = 0.01
        self.dynamic_param_label = tk.Label(self.win, text="Values of relevant variables")
        self.dynamic_param_label.place(x=self.w * 0.16, y=self.h * 0.62)
        self.static_param1_label = tk.Label(self.win, text="theta 1 = 0.0")
        self.static_param1_label.place(x=self.w * 0.17, y=self.h * 0.65)
        self.static_param2_label = tk.Label(self.win, text="omega 1 = 0.0")
        self.static_param2_label.place(x=self.w * 0.17, y=self.h * 0.68)
        self.dynamic_param1_label = tk.Label(self.win, text="theta 2 = 0.0")
        self.dynamic_param1_label.place(x=self.w * 0.17, y=self.h * 0.71)
        self.dynamic_param2_label = tk.Label(self.win, text="omega 2 = 0.0")
        self.dynamic_param2_label.place(x=self.w * 0.17, y=self.h * 0.74)
        self.t_label = tk.Label(self.win, text="      t = 0.0")
        self.t_label.place(x=self.w * 0.17, y=self.h * 0.77)
        
        #Action buttons (start iterating, save plot)
        self.play = False
        self.is_writing = False
        self.tick_btn = tk.Button(self.win,text='Tick',command=self.tick_btn_listener, width = 8)
        self.tick_btn.place(x=self.w * 0.16, y=self.h * 0.80)
        self.tick_btn.update()
        self.play_btn = tk.Button(self.win,text='Play',command=self.play_btn_listener, width = 8)
        self.play_btn.place(x=self.w * 0.16 + 84, y=self.h * 0.80)
        self.play_btn.update()
        self.reset_btn = tk.Button(self.win,text='Reset',command=self.reset_btn_listener, width = 8)
        self.reset_btn.place(x=self.w * 0.16 + 167, y=self.h * 0.80)
        self.reset_btn.update()
        
        self.write_btn = tk.Button(self.win,text='Start writing',command=self.write_btn_listener, width = 14)
        self.write_btn.place(x=self.w * 0.16, y=self.h * 0.84)
        self.write_btn.update()
        #Characteristic properties
        self.characterisation = False
        self.prop_btn = tk.Button(self.win,text='Enable meta', command=self.prop_btn_listener, width = 14)
        self.prop_btn.place(x=self.w * 0.16 + 125, y=self.h * 0.84)
        self.prop_btn.update()
        #T = total kinetic energy. V = total potential energy. E = total energy. div P = divergence of the phase portrait (Louville's theorem)
        self.prop_T = tk.Label(self.win, text="T = 0.0")
        self.prop_T.place(x=self.w * 0.17, y=self.h * 0.88)
        self.prop_V = tk.Label(self.win, text="V = 0.0")
        self.prop_V.place(x=self.w * 0.17, y=self.h * 0.91)
        self.prop_E = tk.Label(self.win, text="E = 0.0")
        self.prop_E.place(x=self.w * 0.17, y=self.h * 0.94)
        self.prop_div = tk.Label(self.win, text="div P = 0.0")
        self.prop_div.place(x=self.w * 0.17, y=self.h * 0.97)
        
        #Global parameter sliders
        self.m_1 = 1.0
        self.m_2 = 1.0
        self.l_1 = 1.0
        self.l_2 = 1.0
        self.g = 9.8
        self.global_change = False
        self.m_1_slider = tk.Scale(self.win, from_=0.1, to=10.0, resolution=0.1, orient=tk.HORIZONTAL, length=self.w*0.22, command=self.refresh_global_param, label="m_1")
        self.m_1_slider.place(x=self.w * 0.75, y=self.h * 0.62)
        self.m_1_slider.set(1.0)
        self.m_2_slider = tk.Scale(self.win, from_=0.1, to=10.0, resolution=0.1, orient=tk.HORIZONTAL, length=self.w*0.22, command=self.refresh_global_param, label="m_2")
        self.m_2_slider.place(x=self.w * 0.75, y=self.h * 0.69)
        self.m_2_slider.set(1.0)
        self.l_1_slider = tk.Scale(self.win, from_=0.1, to=10.0, resolution=0.1, orient=tk.HORIZONTAL, length=self.w*0.22, command=self.refresh_global_param, label="l_1")
        self.l_1_slider.place(x=self.w * 0.75, y=self.h * 0.76)
        self.l_1_slider.set(1.0)
        self.l_2_slider = tk.Scale(self.win, from_=0.1, to=10.0, resolution=0.1, orient=tk.HORIZONTAL, length=self.w*0.22, command=self.refresh_global_param, label="l_2")
        self.l_2_slider.place(x=self.w * 0.75, y=self.h * 0.83)
        self.l_2_slider.set(1.0)
        self.g_slider = tk.Scale(self.win, from_=0.0, to=20.0, resolution=0.1, orient=tk.HORIZONTAL, length=self.w*0.22, command=self.refresh_global_param, label="g")
        self.g_slider.place(x=self.w * 0.75, y=self.h * 0.9)
        self.g_slider.set(9.8)
        
        #Open output text file
        self.data_output_file  = open('data_output.csv', mode='w')
        self.param_output_file = open('param_output.csv', mode='w')
        self.data_output_writer  = csv.writer(self.data_output_file,  delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        self.param_output_writer = csv.writer(self.param_output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        self.data_output_writer.writerow( ['time', 'theta_1', 'omega_1', 'theta_2', 'omega_2'])
        self.param_output_writer.writerow(['time', 'm_1', 'm_2', 'l_1', 'l_2', 'g'])
        
        #Visual output canvas
        self.is_drawing = False
        self.is_tracing = True
        self.o_c_w = int(self.w * 0.4)
        self.o_c_h = int(self.h * 0.6)
        self.o_c_x = int(self.w * 0.99 - self.o_c_w)
        self.o_c_y = int(self.h * 0.01)
        self.output_canvas = tk.Canvas(self.win, width = self.o_c_w, height = self.o_c_h)
        self.output_canvas.place(x = self.o_c_x, y = self.o_c_y)
        self.drawing_scale = (self.o_c_h / 2.2) / (self.l_1 + self.l_2)
        self.x0 = self.o_c_w / 2.0
        self.y0 = self.o_c_h / 2.0
        x1 = self.x0 + self.l_1 * self.drawing_scale * np.sin(self.theta_1)
        y1 = self.y0 + self.l_1 * self.drawing_scale * np.cos(self.theta_1)
        x2 = x1 + self.l_2 * self.drawing_scale * np.sin(self.theta_2)
        y2 = y1 + self.l_2 * self.drawing_scale * np.cos(self.theta_2)
        self.rod1 = self.output_canvas.create_line(self.x0, self.y0, x1, y1)
        self.rod2 = self.output_canvas.create_line(x1, y1, x2, y2)
        self.mass1 = self.output_canvas.create_oval(x1 - self.m_1, y1 - self.m_1, x1 + self.m_1, y1 + self.m_1, fill = "red")
        self.mass2 = self.output_canvas.create_oval(x2 - self.m_2, y2 - self.m_2, x2 + self.m_2, y2 + self.m_2, fill = "red")
        
        self.draw_btn = tk.Button(self.win,text='Enable drawing', command=self.draw_btn_listener, width = 14)
        self.draw_btn.place(x= self.w * 0.99 - self.o_c_w, y=self.h * 0.01)
        self.draw_btn.update()
        self.tracing_btn = tk.Button(self.win,text='Disable tracing', command=self.tracing_btn_listener, width = 14)
        self.tracing_btn.place(x= self.w * 0.99 - self.o_c_w + 125, y=self.h * 0.01)
        self.tracing_btn.update()
        self.trace_erase_btn = tk.Button(self.win,text='Erase tracing', command=self.trace_erase_btn_listener, width = 14)
        self.trace_erase_btn.place(x= self.w * 0.99 - self.o_c_w + 250, y=self.h * 0.01)
        self.trace_erase_btn.update()
        self.new_output_btn = tk.Button(self.win,text='New output', command=self.new_output_btn_listener, width = 14)
        self.new_output_btn.place(x= self.w * 0.99 - self.o_c_w + 375, y=self.h * 0.01)
        self.new_output_btn.update()
        
        #OpenCV visual output luggage
        self.FPS = 100
        self.fourcc = VideoWriter_fourcc(*'MP42')
        self.number_of_videos = 1
        self.video = VideoWriter('./double_pendulum_video_output' + str(self.number_of_videos) + '.avi', self.fourcc, float(self.FPS), (self.w, self.h))
        
        self.output_frames = []
        self.layout_frame = np.zeros((self.h, self.w, 3), dtype=np.uint8) + 255
        self.trace_frame = np.zeros((self.o_c_h, self.o_c_w, 3), dtype=np.uint8) + 255
        
        #Install closing handler protocol
        self.win.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.theta_1 = 0.0
        self.omega_1 = 3.0
        self.theta_2 = -2.0
        self.omega_2 = 0.0
        
        #Trigger startup
        self.phase_portrait(self.theta_1, self.omega_1)
        
        
        #Start listening for events
        
        self.win.mainloop()

    #--------------- OUTPUT METHODS ----------------------
    
    def phase_portrait(self, static_param1, static_param2):
        
        cur_dynam_param1, cur_dynam_param2 = self.set_static_param(static_param1, static_param2)
        self.ax.clear()
        self.ax.set_xlabel(self.x_lab)
        self.ax.set_ylabel(self.y_lab)
        x_ran = 5.0
        y_ran = 10.0
        if self.style != 2:
            x_space = np.linspace(cur_dynam_param1 - x_ran, cur_dynam_param1 + x_ran, 40)
            y_space = np.linspace(cur_dynam_param2 + y_ran, cur_dynam_param2 - y_ran, 30)
        else:
            x_space = np.linspace(cur_dynam_param1 - x_ran, cur_dynam_param1 + x_ran, 80)
            y_space = np.linspace(cur_dynam_param2 + y_ran, cur_dynam_param2 - y_ran, 60)
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
        
        #Characterisation and drawing
        if self.characterisation or self.is_drawing:
            cur_T, cur_V, cur_E = self.characterize()
        if self.is_drawing:
            self.draw_self(cur_T, cur_V, cur_E)
        
        return 0,5

    def draw_self(self, cur_T, cur_V, cur_E):
        self.output_canvas.delete(self.rod1)
        self.output_canvas.delete(self.mass1)
        self.output_canvas.delete(self.rod2)
        self.output_canvas.delete(self.mass2)
        x1 = self.x0 + self.l_1 * self.drawing_scale * np.sin(self.theta_1)
        y1 = self.y0 + self.l_1 * self.drawing_scale * np.cos(self.theta_1)
        x2 = x1 + self.l_2 * self.drawing_scale * np.sin(self.theta_2)
        y2 = y1 + self.l_2 * self.drawing_scale * np.cos(self.theta_2)
        if self.is_tracing:
            self.traces.append(self.output_canvas.create_oval(x2 - 1, y2 - 1, x2 + 1, y2 + 1, fill = "blue")) #tracer
        self.rod1 = self.output_canvas.create_line(self.x0, self.y0, x1, y1)
        self.rod2 = self.output_canvas.create_line(x1, y1, x2, y2)
        self.mass1 = self.output_canvas.create_oval(x1 - self.m_1_r, y1 - self.m_1_r, x1 + self.m_1_r, y1 + self.m_1_r, fill = "red")
        self.mass2 = self.output_canvas.create_oval(x2 - self.m_2_r, y2 - self.m_2_r, x2 + self.m_2_r, y2 + self.m_2_r, fill = "red")
        #OpenCV output
        cur_frame = self.layout_frame.copy()
        cur_frame[self.o_c_y:self.o_c_y+self.o_c_h, self.o_c_x:self.o_c_x+self.o_c_w] = self.trace_frame
        
        new_img = fig_to_img(self.fig, self.dpi)
        width = len(new_img[0])
        height = len(new_img)
        cur_frame[int(self.h * 0.01):int(self.h * 0.01)+height, 0:0+width] = new_img
        
        offset_x = self.w * 0.99 - self.o_c_w
        offset_y = self.h * 0.01
        
        cv2.line(cur_frame, (int(offset_x + self.x0), int(offset_y + self.y0)), (int(offset_x + x1), int(offset_y + y1)), (0,0,0), 2)
        cv2.line(cur_frame, (int(offset_x + x1), int(offset_y + y1)), (int(offset_x + x2), int(offset_y + y2)), (0,0,0), 2)
        
        cv2.circle(cur_frame, (int(offset_x + x1), int(offset_y + y1)), int(self.m_1_r), (0,0,0), 2 )
        cv2.circle(cur_frame, (int(offset_x + x1), int(offset_y + y1)), int(self.m_1_r), (0,0,255), -1 )
        cv2.circle(cur_frame, (int(offset_x + x2), int(offset_y + y2)), int(self.m_2_r), (0,0,0), 2 )
        cv2.circle(cur_frame, (int(offset_x + x2), int(offset_y + y2)), int(self.m_2_r), (0,0,255), -1 )
        
        if self.is_tracing:
            cv2.circle(self.trace_frame, (int(x2), int(y2)), 2, (255,0,0), -1 )
        
        text_offset_x = int(self.w * 0.05)
        text_offset_y = int(self.h * 0.65) + 20
        cv2.putText(cur_frame, "t = " + str(round(self.t, 3)), (text_offset_x, text_offset_y), self.font, 1, (0,0,0))
        text_offset_x += 300
        cv2.putText(cur_frame, "T = " + str(round(cur_T, 3)), (text_offset_x, text_offset_y + 80), self.font, 1, (0,0,0))
        cv2.putText(cur_frame, "V = " + str(round(cur_V, 3)), (text_offset_x, text_offset_y + 120), self.font, 1, (0,0,0))
        cv2.putText(cur_frame, "E = " + str(round(cur_E, 3)), (text_offset_x, text_offset_y + 160), self.font, 1, (0,0,0))
        text_offset_x += 300        
        cv2.putText(cur_frame, "theta_1 = " + str(round(self.theta_1, 3)), (text_offset_x, text_offset_y), self.font, 1, (0,0,0))
        cv2.putText(cur_frame, "omega_1 = " + str(round(self.omega_1, 3)), (text_offset_x, text_offset_y + 40), self.font, 1, (0,0,0))
        cv2.putText(cur_frame, "theta_2 = " + str(round(self.theta_2, 3)), (text_offset_x, text_offset_y + 80), self.font, 1, (0,0,0))
        cv2.putText(cur_frame, "omega_2 = " + str(round(self.omega_2, 3)), (text_offset_x, text_offset_y + 120), self.font, 1, (0,0,0))
        
        
        self.output_frames.append(cur_frame)
    
    def release_video(self):
        for frame in self.output_frames:
            self.video.write(frame)
        self.video.release()
        
        self.output_frames = []
        self.trace_frame = np.zeros((self.h, self.w, 3), dtype=np.uint8) + 255
    
    #------------ PARAMETER MANAGEMENT --------------------
    
    def refreshParam(self, new_val=1.0):
        #This method is called whenever any of the params change. new_val is therefore useless
        self.phase_portrait(self.param1.get(), self.param2.get())
    
    def refresh_global_param(self, new_val=0.0):
        #An umbrella method for global param sliders
        self.global_change = True
        self.m_1           = self.m_1_slider.get()
        self.m_2           = self.m_2_slider.get()
        self.l_1           = self.l_1_slider.get()
        self.l_2           = self.l_2_slider.get()
        self.g             = self.g_slider.get()
        self.F             = self.F_slider.get()
        self.omega_F       = self.F_omega_slider.get()
        self.omega_F_range = self.F_range_slider.get()
        
        #Drawing scale changed
        old_scale = self.drawing_scale
        #new_traces = []
        self.drawing_scale = (self.o_c_h / 2.2) / (self.l_1 + self.l_2)
        for my_trace in self.traces:
            self.output_canvas.scale(my_trace, self.x0, self.y0, self.drawing_scale / old_scale, self.drawing_scale / old_scale)
        self.m_1_r = np.sqrt(self.m_1 * 10.0)
        self.m_2_r = np.sqrt(self.m_2 * 10.0)
        self.layout_frame = np.zeros((self.h, self.w, 3), dtype=np.uint8) + 255
        text_offset_x = int(self.w * 0.05)
        text_offset_y = int(self.h * 0.65) + 60
        cv2.putText(self.layout_frame, "m_1 = " + str(self.m_1), (text_offset_x, text_offset_y), self.font, 1, (0,0,0))
        cv2.putText(self.layout_frame, "m_2 = " + str(self.m_2), (text_offset_x, text_offset_y + 40), self.font, 1, (0,0,0))
        cv2.putText(self.layout_frame, "l_1 = " + str(self.l_1), (text_offset_x, text_offset_y + 80), self.font, 1, (0,0,0))
        cv2.putText(self.layout_frame, "l_2 = " + str(self.l_2), (text_offset_x, text_offset_y + 120), self.font, 1, (0,0,0))
        cv2.putText(self.layout_frame, "g = " + str(self.g), (text_offset_x, text_offset_y + 160), self.font, 1, (0,0,0))
        text_offset_x += 300
        cv2.putText(self.layout_frame, "F = " + str(self.F), (text_offset_x, text_offset_y - 40), self.font, 1, (0,0,0))
        cv2.putText(self.layout_frame, "omega_F = " + str(self.omega_F), (text_offset_x, text_offset_y), self.font, 1, (0,0,0))
        
        static_param1 = 0.0
        static_param2 = 0.0
        if self.param_map == 0:
            static_param1 = self.theta_2
            static_param2 = self.omega_2
        elif self.param_map == 1:
            static_param1 = self.omega_1
            static_param2 = self.omega_2
        elif self.param_map == 2:
            static_param1 = self.theta_2
            static_param2 = self.omega_1
        elif self.param_map == 3:
            static_param1 = self.theta_1
            static_param2 = self.omega_2
        elif self.param_map == 4:
            static_param1 = self.theta_1
            static_param2 = self.theta_2
        elif self.param_map == 5:
            static_param1 = self.theta_1
            static_param2 = self.omega_1
        
        if not self.play:
            self.phase_portrait(static_param1, static_param2)
    
    def set_param_map(self, new_param_map):
        self.param_map = new_param_map
        if new_param_map == 0:
            self.x_lab = "theta_1"
            self.y_lab = "omega_1"
            self.param1['label'] = "theta_2"
            self.param2['label'] = "omega_2"
            self.param1.set(self.theta_2)
            self.param2.set(self.omega_2)
        if new_param_map == 1:
            self.x_lab = "theta_1"
            self.y_lab = "theta_2"
            self.param1['label'] = "omega_1"
            self.param2['label'] = "omega_2"
            self.param1.set(self.omega_1)
            self.param2.set(self.omega_2)
        if new_param_map == 2:
            self.x_lab = "theta_1"
            self.y_lab = "omega_2"
            self.param1['label'] = "theta_2"
            self.param2['label'] = "omega_1"
            self.param1.set(self.theta_2)
            self.param2.set(self.omega_1)
        if new_param_map == 3:
            self.x_lab = "theta_2"
            self.y_lab = "omega_1"
            self.param1['label'] = "theta_1"
            self.param2['label'] = "omega_2"
            self.param1.set(self.theta_1)
            self.param2.set(self.omega_2)
        if new_param_map == 4:
            self.x_lab = "omega_1"
            self.y_lab = "omega_2"
            self.param1['label'] = "theta_1"
            self.param2['label'] = "theta_2"
            self.param1.set(self.theta_1)
            self.param2.set(self.theta_2)
        if new_param_map == 5:
            self.x_lab = "theta_2"
            self.y_lab = "omega_2"
            self.param1['label'] = "theta_1"
            self.param2['label'] = "omega_1"
            self.param1.set(self.theta_1)
            self.param2.set(self.omega_1)
    
    def change_param_map(self):
        self.set_param_map(self.param_map_var.get())
        dynam_1, dynam_2 = self.set_static_param(self.param1.get(), self.param2.get())
        self.dynamic_param1_label['text'] = self.x_lab + " = " + str(dynam_1)
        self.dynamic_param2_label['text'] = self.y_lab + " = " + str(dynam_2)
        self.refreshParam()
    
    def set_static_param(self, val_1, val_2, backtrack = True):
        #if backtrack:
            #self.param1.set(val_1)
            #self.param2.set(val_2)
        if self.param_map == 0:
            self.theta_2 = val_1
            self.omega_2 = val_2
            self.static_param1_label['text'] = "theta_2 = " + str(val_1)
            self.static_param2_label['text'] = "omega_2 = " + str(val_2)
            return self.theta_1, self.omega_1
        if self.param_map == 1:
            self.omega_1 = val_1
            self.omega_2 = val_2
            self.static_param1_label['text'] = "omega_1 = " + str(val_1)
            self.static_param2_label['text'] = "omega_2 = " + str(val_2)
            return self.theta_1, self.theta_2
        if self.param_map == 2:
            self.theta_2 = val_1
            self.omega_1 = val_2
            self.static_param1_label['text'] = "theta_2 = " + str(val_1)
            self.static_param2_label['text'] = "omega_1 = " + str(val_2)
            return self.theta_1, self.omega_2
        if self.param_map == 3:
            self.theta_1 = val_1
            self.omega_2 = val_2
            self.static_param1_label['text'] = "theta_1 = " + str(val_1)
            self.static_param2_label['text'] = "omega_2 = " + str(val_2)
            return self.theta_2, self.omega_1
        if self.param_map == 4:
            self.theta_1 = val_1
            self.theta_2 = val_2
            self.static_param1_label['text'] = "theta_1 = " + str(val_1)
            self.static_param2_label['text'] = "theta_2 = " + str(val_2)
            return self.omega_1, self.omega_2
        if self.param_map == 5:
            self.theta_1 = val_1
            self.omega_1 = val_2
            self.static_param1_label['text'] = "theta_1 = " + str(val_1)
            self.static_param2_label['text'] = "omega_1 = " + str(val_2)
            return self.theta_2, self.omega_2
    
    def set_dynamic_param(self, val_1, val_2):
        static_param1 = 0.0
        static_param2 = 0.0
        if self.param_map == 0:
            self.theta_1 = val_1
            self.omega_1 = val_2
            static_param1 = self.theta_2
            static_param2 = self.omega_2
        if self.param_map == 1:
            self.theta_1 = val_1
            self.theta_2 = val_2
            static_param1 = self.omega_1
            static_param2 = self.omega_2
        if self.param_map == 2:
            self.theta_1 = val_1
            self.omega_2 = val_2
            static_param1 = self.theta_2
            static_param2 = self.omega_1
        if self.param_map == 3:
            self.theta_2 = val_1
            self.omega_1 = val_2
            static_param1 = self.theta_1
            static_param2 = self.omega_2
        if self.param_map == 4:
            self.omega_1 = val_1
            self.omega_2 = val_2
            static_param1 = self.theta_1
            static_param2 = self.theta_2
        if self.param_map == 5:
            self.theta_2 = val_1
            self.omega_2 = val_2
            static_param1 = self.theta_1
            static_param2 = self.omega_1
        self.dynamic_param1_label['text'] = self.x_lab + " = " + str(val_1)
        self.dynamic_param2_label['text'] = self.y_lab + " = " + str(val_2)
        self.phase_portrait(static_param1, static_param2)
    
    def get_dynamic_res(self, val_1, val_2):
        if self.param_map == 0:
            x_res = val_2
            y_res = self.get_epsilon_1(val_1, val_2, self.theta_2, self.omega_2)
        if self.param_map == 1:
            x_res = self.omega_1
            y_res = self.omega_2
        if self.param_map == 2:
            x_res = self.omega_1
            y_res = self.get_epsilon_2(val_1, self.omega_1, self.theta_2, val_2)
        if self.param_map == 3:
            x_res = self.omega_2
            y_res = self.get_epsilon_1(self.theta_1, val_2, val_1, self.omega_2)
        if self.param_map == 4:
            x_res = self.get_epsilon_1(self.theta_1, val_1, self.theta_2, val_2)
            y_res = self.get_epsilon_2(self.theta_1, val_1, self.theta_2, val_2)
        if self.param_map == 5:
            x_res = val_2
            y_res = self.get_epsilon_2(self.theta_1, self.omega_1, val_1, val_2)
        return(x_res, y_res)
    
    def change_style(self):
        self.style = self.style_var.get()
        self.refreshParam()
    
    #------------ SIMULATION LISTENERS --------------------
    
    def on_click(self, event):
        if event.inaxes is not None:
            #print(event.xdata, event.ydata)
            self.set_dynamic_param(event.xdata, event.ydata)
        else:
            print('Clicked ouside axes bounds but inside plot self.window')
    
    def tick_btn_listener(self):
        t1, o1, t2, o2 = self.step()
        self.t_label['text'] = "      t = " + str(round(self.t, 3))
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
        elif self.param_map == 1:
            self.set_static_param(o1, o2, True)
            self.set_dynamic_param(t1, t2)
        elif self.param_map == 2:
            self.set_static_param(t2, o1, True)
            self.set_dynamic_param(t1, o2)
        elif self.param_map == 3:
            self.set_static_param(t1, o2, True)
            self.set_dynamic_param(t2, o1)
        elif self.param_map == 4:
            self.set_static_param(t1, t2, True)
            self.set_dynamic_param(o1, o2)
        elif self.param_map == 5:
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
    
    def reset_btn_listener(self):
        self.set_static_param(0.0, 0.0)
        self.set_dynamic_param(0.0, 0.0)
        self.param1.set(0.0)
        self.param2.set(0.0)
        self.t = 0.0
        if self.play:
            self.tick_btn.configure(state = "normal")
            self.play_btn['text'] = 'Play'
            self.play = False
            self.t -= 0.01
        self.phase_portrait(0.0, 0.0)
        
    def write_btn_listener(self):
        if self.is_writing == True:
            #Change the label
            self.write_btn['text'] = 'Start writing'
            self.is_writing = False
        elif self.is_writing == False:
            #Change the label
            self.write_btn['text'] = 'Stop writing'
            self.is_writing = True
        
    def prop_btn_listener(self):
        if self.characterisation == True:
            #Change the label
            self.prop_btn['text'] = 'Enable meta'
            self.characterisation = False
        elif self.characterisation == False:
            #Change the label
            self.prop_btn['text'] = 'Disable meta'
            self.characterisation = True
    
    def draw_btn_listener(self):
        if self.is_drawing == True:
            #Change the label
            self.draw_btn['text'] = 'Enable drawing'
            self.is_drawing = False
        elif self.is_drawing == False:
            #Change the label
            self.draw_btn['text'] = 'Disable drawing'
            self.is_drawing = True
    
    def tracing_btn_listener(self):
        if self.is_tracing == True:
            #Change the label
            self.tracing_btn['text'] = 'Enable tracing'
            self.is_tracing = False
        elif self.is_tracing == False:
            #Change the label
            self.tracing_btn['text'] = 'Disable tracing'
            self.is_tracing = True
    
    def trace_erase_btn_listener(self):
        for my_trace in self.traces:
            self.output_canvas.delete(my_trace)
        self.traces = []
        self.trace_frame = np.zeros((self.o_c_h, self.o_c_w, 3), dtype=np.uint8) + 255
    
    def new_output_btn_listener(self):
        self.release_video()
        self.number_of_videos += 1
        self.video = VideoWriter('./double_pendulum_video_output' + str(self.number_of_videos) + '.avi', self.fourcc, float(self.FPS), (self.w, self.h))
        
    
    def play_func(self):
        self.tick_btn_listener()
        if self.play:
            self.win.after(1, self.play_func)
    
    def damping_analysis_btn_listener(self):
        print("Damping analysis started with the following parameters:")
        omega_0 = np.sqrt(self.g / self.l_1)
        param_string1 = "F = " + str(self.F) + "; omega_F = " + str(self.omega_F) + "; omega_F_range = " + str(self.omega_F_range) + "; omega_0 = " + str(omega_0)
        param_string2 = "m_1 = " + str(self.m_1) + "; l_1 = " + str(self.l_1) + "; m_2 = " + str(self.m_2) + "; l_2 = " + str(self.l_2) + "; g = " + str(self.g)
        print(param_string1)
        print(param_string2)
        
        
        self.damp_output_popup = popupWindow(self.win)
        self.win.wait_window(self.damp_output_popup.top)
        output_filename = str(self.damp_output_popup.value)
        
        #self.damp_output_file = open('damping_analysis_output.csv', mode='w')

        # We write all aggregate properties into damp_output_file
        # and create a subfolder in Data called /trajectory_[filename] where we save each trajectory into 200 textfiles called trajectory_[number]
        # and put global parameters into param_config_file at Data/config/__PARAM_MAP__.txt
        # when damping-analyzer analyzes the output for the first time, it purges the /trajectory folder and leaves only the trajectories at resonant frequencies
        # and it flips a switch property called "purge_trajectories" which is at index 9 in the __PARAM_MAP__
        
        # idea to make it cleaner: save aggregate outputs in the respective folders as well?

        self.damp_output_file = open("Data/" + output_filename + '.csv', mode='w')
        self.damp_output_writer = csv.writer(self.damp_output_file,  delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # HEADER CHANGELOG: 't'->'max_theta_t', added 'avg_E' (average mechanical energy = <T + U>)
        self.damp_output_writer.writerow( ['omega_F', 'max_theta', 'max_theta_t', 'avg_E', 'ang_f_1', 'ang_f_2', 'hp_1_std', 'hp_2_std'])

        self.param_config_file = open("Data/config/__PARAM_MAP__.txt", mode="a+")
        self.param_config_file.write("\n" + output_filename + ": " + param_string1 + "; " + param_string2 + "; purge_trajectories = 1")
        self.param_config_file.close()

        Path("Data/trajectory_" + output_filename).mkdir(parents=True, exist_ok=True)
        
        omega_space_min = self.omega_F - self.omega_F_range
        omega_space_max = self.omega_F + self.omega_F_range
        if omega_space_min < 0:
            omega_space_min = 0.0
        omega_space_datapoints = 200
        omega_space = np.linspace(omega_space_min, omega_space_max, omega_space_datapoints)
        max_t = 500.0
        min_theta = 0.01
        t_period_threshold = 100.0
        
        self.damp_progress_label['text'] = "Analysis in progress: 0%"
        cycle_index = 0
        percentage_done = 0
        for omega_val in omega_space:

            # Initialize dynamic properties and open the writer

            cycle_index += 1
            self.t = 0.0
            self.set_static_param(0.0, 0.0)
            self.set_dynamic_param(0.0, 0.0)
            self.omega_F = omega_val

            self.trajectory_output_file = open("Data/trajectory_" + output_filename + "/trajectory_" + str(cycle_index) + '.csv', mode='w')
            self.trajectory_output_writer = csv.writer(self.trajectory_output_file,  delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            self.trajectory_output_writer.writerow( ['t', 'theta_1', 'theta_2', 'n'])
            
            # Quantities to be determined for each config
            
            max_theta = 0.0
            max_theta_t = 0.0
            average_mechanical_energy = 0.0
            total_halfperiods_1 = 0
            total_halfperiods_2 = 0
            halfperiod_1_length_std = 0
            halfperiod_2_length_std = 0

            # Helping properties

            last_halfperiod_1_t = 0.0
            last_halfperiod_2_t = 0.0
            halfperiod_1_length_list = []
            halfperiod_2_length_list = []
            number_of_active_cycles = 0
            
            # When to start counting the periods
            period_sign_1 = 0
            period_sign_2 = 0
            period_signs_init = False
            
            # Perform the simulation for a particular configuration of parameters
            
            while self.t < max_t:
                t1, o1, t2, o2 = self.step()
                self.theta_1 = t1
                self.omega_1 = o1
                self.theta_2 = t2
                self.omega_2 = o2
            
                if t1 > max_theta:
                    max_theta = t1
                    max_theta_t = self.t
                
                
                if self.t > t_period_threshold:
                    
                    number_of_active_cycles += 1

                    # Check for increase of halfperiods
                    # Remember that a halfperiod is also when the segment flips over - we need to check a modulo-ed version of the parameter
                    if not period_signs_init:
                        period_sign_1 = nonzero_sign(base_theta(t1))
                        period_sign_2 = nonzero_sign(base_theta(t2))
                        period_signs_init = True
                        
                    if (period_sign_1 * nonzero_sign(base_theta(t1)) == -1):
                        period_sign_1 *= -1
                        total_halfperiods_1 += 1
                        if last_halfperiod_1_t != 0.0:
                            halfperiod_1_length_list.append(self.t - last_halfperiod_1_t)
                        last_halfperiod_1_t = self.t
                        
                    if (period_sign_2 * nonzero_sign(base_theta(t2)) == -1):
                        period_sign_2 *= -1
                        total_halfperiods_2 += 1
                        if last_halfperiod_2_t != 0.0:
                            halfperiod_2_length_list.append(self.t - last_halfperiod_2_t)
                        last_halfperiod_2_t = self.t

                    # Write dynamic properties into the correct output
                    if self.theta_1 * self.theta_1 > min_theta * min_theta:
                        self.trajectory_output_writer.writerow( [self.t, self.theta_1, self.theta_2, self.theta_2 / self.theta_1])
                    # Write other to-be-aggregated data
                    kinetic_energy, potential_energy, mechanical_energy = self.get_energy()
                    average_mechanical_energy += mechanical_energy
                    
                    
            #print(omega_val, max_theta, max_theta_t)
            percentage_done = cycle_index * 100 / omega_space_datapoints
            self.damp_progress_label['text'] = "Analysis in progress: " + str(round(percentage_done, 1)) + "%"
            
            average_mechanical_energy /= number_of_active_cycles
            ang_f_1 = total_halfperiods_1 / (2.0 * (max_t - t_period_threshold)) * 2.0 * np.pi
            ang_f_2 = total_halfperiods_2 / (2.0 * (max_t - t_period_threshold)) * 2.0 * np.pi
            halfperiod_1_length_std = statistics.stdev(halfperiod_1_length_list)
            halfperiod_2_length_std = statistics.stdev(halfperiod_2_length_list)
            self.damp_output_writer.writerow([omega_val, max_theta, max_theta_t, average_mechanical_energy, ang_f_1, ang_f_2, halfperiod_1_length_std, halfperiod_2_length_std])

            self.trajectory_output_file.close()
        
        self.damp_output_file.close()
        self.damp_progress_label['text'] = "Data exported to file " + output_filename
    
    def load_prev_params_btn_listener(self):
        print("Loading last saved session...")
        meta_config_file = open("meta_config/GUI_META_CONFIG.txt", mode="r")
        prev_params_string = meta_config_file.readlines()[0]
        meta_config_file.close()
        
        my_prev_params_strings = prev_params_string.split(' ')
        my_prev_params = []
        for element in my_prev_params_strings:
            my_prev_params.append(float(element))
        #print(my_prev_params)
        self.m_1_slider.set(my_prev_params[0])
        self.m_2_slider.set(my_prev_params[1])
        self.l_1_slider.set(my_prev_params[2])
        self.l_2_slider.set(my_prev_params[3])
        self.g_slider.set(my_prev_params[4])
        
        self.F_slider.set(my_prev_params[5])
        self.F_omega_slider.set(my_prev_params[6])
        self.F_range_slider.set(my_prev_params[7])
        
    def on_closing(self):
        #Close output files
        self.data_output_file.close()
        self.param_output_file.close()
        
        #Release the last OpenCV video
        self.release_video()
        
        
        #Save the global param config
        meta_config_file = open("meta_config/GUI_META_CONFIG.txt", mode="w")
        static_params_string = str(self.m_1) + " " + str(self.m_2) + " " + str(self.l_1) + " " + str(self.l_2) + " " + str(self.g)
        force_params_string = str(self.F) + " " + str(self.F_omega_slider.get()) + " " + str(self.omega_F_range)
        meta_config_file.write(static_params_string + " " + force_params_string)
        meta_config_file.close()
        
        #Close down Tkinter
        self.win.destroy()
    
    #-------------- PHYSICS METHODS -----------------------
    
    #def double_pend_f(self, theta_2, omega_2):
    #    
    
    def get_epsilon_1(self, my_theta_1, my_omega_1, my_theta_2, my_omega_2):
        numerator = self.F*np.sin(self.omega_F * self.t) - self.m_2*np.sin(my_theta_1-my_theta_2)*(self.l_2*my_omega_2**2+self.l_1*my_omega_1**2*np.cos(my_theta_1-my_theta_2)) - self.g*(self.m_1*np.sin(my_theta_1)+self.m_2*np.cos(my_theta_2)*np.sin(my_theta_1-my_theta_2))
        denominator = self.l_1*((self.m_1+self.m_2)-self.m_2*np.cos(my_theta_1-my_theta_2)*np.cos(my_theta_1-my_theta_2))
        return(numerator/denominator)
    
    def get_epsilon_2(self, my_theta_1, my_omega_1, my_theta_2, my_omega_2):
        numerator = self.F*np.sin(self.omega_F*self.t)*np.cos(my_theta_1-my_theta_2) - ((self.m_1+self.m_2)*self.l_1*my_omega_1**2+self.m_2*self.l_2*my_omega_2**2*np.cos(my_theta_1-my_theta_2)+(self.m_1+self.m_2)*self.g*np.cos(my_theta_1))*np.sin(my_theta_1-my_theta_2)
        denominator = self.l_2*(self.m_2*np.cos(my_theta_1-my_theta_2)*np.cos(my_theta_1-my_theta_2) - (self.m_1+self.m_2))
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
    
    def get_energy(self):
        # Returns a tuple in the form (kinetic e., potential e., mechanical e.)
        T_1 = 0.5 * (self.m_1 + self.m_2) * self.l_1 * self.l_1 * self.omega_1 * self.omega_1
        T_2 = 0.5 * self.m_2 * self.l_2 * self.l_2 * self.omega_2 * self.omega_2
        T_3 = self.m_2 * self.l_1 * self.l_2 * self.omega_1 * self.omega_2 * np.cos(self.theta_2 - self.theta_1)
        cur_T = T_1 + T_2 + T_3
        cur_V = - (self.m_1 + self.m_2) * self.g * self.l_1 * np.cos(self.theta_1) - self.m_2 * self.g * self.l_2 * np.cos(self.theta_2)
        return(cur_T, cur_V, cur_T + cur_V)
    
    
    def characterize(self):
        
        kinetic_energy, potential_energy, mechanical_energy = self.get_energy()
        self.prop_T['text'] = "T = " + str(kinetic_energy)
        self.prop_V['text'] = "V = " + str(potential_energy)
        self.prop_E['text'] = "E = " + str(mechanical_energy)
        
        return()


#------------------------------------------------------------------        
if __name__ =='__main__':   
    MyApp=MyGUI()
