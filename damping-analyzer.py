#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 11:42:16 2020

@author: michal
"""

import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import pandas as pd

import os


# ------------------------------------------------
# ---------------- FUNCTIONS ---------------------
# ------------------------------------------------

# -------------- graph functions -----------------

# autoscale omitter
def autoscale_based_on(ax, lines):
    ax.dataLim = mtransforms.Bbox.unit()
    for line in lines:
        xy = np.vstack(line[0].get_data()).T
        ax.dataLim.update_from_data_xy(xy, ignore=False)
    ax.autoscale_view()

# subplot dimensions organizer
def subplot_dimensions(number_of_plots):
    # 1x1, 2x1, 3x1, 2x2, 3x2, 4x2, 3x3, 4x3, 5x3
    if number_of_plots == 1:
        return(1, 1)
    if number_of_plots == 2:
        return(2, 1)
    if number_of_plots == 3:
        return(3, 1)
    if number_of_plots == 4:
        return(2, 2)
    if number_of_plots <= 6:
        return(3, 2)
    if number_of_plots <= 8:
        return(4, 2)
    if number_of_plots <= 9:
        return(3, 3)
    if number_of_plots <= 12:
        return(4, 3)
    if number_of_plots <= 15:
        return(5, 3)
    return(np.ceil(np.sqrt(number_of_plots)), np.ceil(np.sqrt(number_of_plots)))


# ------------- physics functions ----------------

#Simple pendulum amplitude func
def simple_amp(omega_F, omega_0, F, m):
    #if omega_0 == omega_F:
    #    return(0)
    return(F / (m * abs(omega_0 ** 2 - omega_F ** 2)))

# ------------- The smoothing (rolling avg) analysis -------------------

# A collection of spread functions

def rect(x, spread_parameter):
    # The spread parameter is the total width
    if np.abs(x) > spread_parameter / 2.0:
        return(0.0)
    if np.abs(x) == spread_parameter / 2.0:
        return(0.5 / spread_parameter)
    if np.abs(x) < spread_parameter / 2.0:
        return(1.0 / spread_parameter)

def gaussian(x, spread_parameter):
    # The spread parameter is the standard deviation of the gaussian
    distribution_value = np.exp(-0.5 * x * x / (spread_parameter * spread_parameter))
    normalizing_factor = 1.0 / (spread_parameter * np.sqrt(2.0 * np.pi))
    return(normalizing_factor * distribution_value)

# The convolution func takes dimension array, data numpy array (must have equal dimension), spread function, and a single parameter defining
# relative spread (funcs get normalized to int_infty = 1). Returns a single array of same dimension as data array

def convolute(dimension_array, data_array, spread_function, spread_parameter):

    # We assume that the dimension array is ORDERED and ASCENDING    
    def data_function(x):
        if x < dimension_array[0] or x >= dimension_array[-1]:
            return(0.0)
        i = 0
        while(dimension_array[i] < x):
            i += 1
        i -= 1
        return(data_array[i] + (data_array[i+1] - data_array[i]) * (x - dimension_array[i]) / (dimension_array[i+1] - dimension_array[i]))

    my_xspace = np.linspace(dimension_array[0] - 10.0, dimension_array[-1] + 10.0, 4000)
    my_yspace = np.zeros(len(my_xspace))
    for i in range(len(my_xspace)):
        my_yspace[i] = data_function(my_xspace[i])
    plt.plot(my_xspace, my_yspace)
    plt.savefig("hoho.png")
    plt.show()

    quit()


def smoothing(df_name, df, spread_function, spread_function_param):

    print("  Beginning smoothing procedure...")

    smooth_df = pd.DataFrame(columns=['omega_F', 'max_theta', 't', 'ang_f_1', 'ang_f_2', 'hp_1_std', 'hp_2_std'])

    print("  Analyzing trajectory files...")
    trajectory_file_index = 65
    trajectory_file_exists = True
    while(trajectory_file_exists):
        try:
            cur_trajectory_df = pd.read_csv("Data/trajectory_" + df_name + "/trajectory_" + str(trajectory_file_index + 1) + '.csv')
            trajectory_file_index += 1

            # Smooth it out
            # read t, theta_1, theta_2
            cur_t_space       = cur_trajectory_df['t'      ].to_numpy()
            cur_theta_1_space = cur_trajectory_df['theta_1'].to_numpy()
            cur_theta_2_space = cur_trajectory_df['theta_2'].to_numpy()
            
            smooth_theta_1_space = np.zeros(len(cur_theta_1_space))

            convolute(cur_t_space, cur_theta_2_space, spread_function, spread_function_param)
            quit()


        except FileNotFoundError:
            trajectory_file_exists = False

    print("  " + str(trajectory_file_index) + " trajectory files analyzed")

    # This is basically a convolution of theta_1 and a spread function (first we try rect(t))
    #if spread_type == "rect":
        


def main_analysis():
    
    try:
        config_file = open("Data/config/__PARAM_MAP__.txt", 'r')
    except FileNotFoundError:
        print("Config file not found. Aborting...")
        quit()

    # Writing meta-config file

    meta_config_file = open("meta_config/damping_analyzer_META_CONFIG.txt", mode="w")
    #filename_list_stringed = 'L'.join([str(elem) for elem in my_filename_list])
    #meta_config_file.write(filename_list_stringed)
    meta_config_file.write(my_filename_list)
    meta_config_file.close()


    # Reading config file

    config_lines = config_file.readlines()
    my_lines = {}
    for line in config_lines:
        elements = line.split(": ")
        if elements[0] in my_filenames:
            #my_lines.append([elements[0], elements[1]])
            my_lines[elements[0]] = [elements[0], elements[1]]
    config_file.close()

    global_parameters = {}
    simple_amps = {}
    for i in range(len(my_filenames)):
        line = my_lines[my_filenames[i]]
        print("---------------------------------------")
        print(line[0] + ": " + line[1])
        # read the global parameters
        split_line = line[1].split('; ')
        global_parameters[my_filenames[i]] = {}
        global_parameters[my_filenames[i]]["F"            ] = float(split_line[0].split(' = ')[1])
        global_parameters[my_filenames[i]]["omega_F"      ] = float(split_line[1].split(' = ')[1])
        global_parameters[my_filenames[i]]["omega_F_range"] = float(split_line[2].split(' = ')[1])
        global_parameters[my_filenames[i]]["omega_0"      ] = float(split_line[3].split(' = ')[1])
        global_parameters[my_filenames[i]]["m_1"          ] = float(split_line[4].split(' = ')[1])
        global_parameters[my_filenames[i]]["l_1"          ] = float(split_line[5].split(' = ')[1])
        global_parameters[my_filenames[i]]["m_2"          ] = float(split_line[6].split(' = ')[1])
        global_parameters[my_filenames[i]]["l_2"          ] = float(split_line[7].split(' = ')[1])
        global_parameters[my_filenames[i]]["g"            ] = float(split_line[8].split(' = ')[1])


        global_parameters[my_filenames[i]]["purge_trajectories"] = float(split_line[9].split(' = ')[1])
    
        if analyze[my_filenames[i]]:
            #cur_F = float(line[1].split('; ')[0].split(' = ')[1])
            #cur_m = float(line[1].split('; ')[4].split(' = ')[1])
            #cur_omega_0 = float(line[1].split('; ')[3].split(' = ')[1])
            min_w = df[my_filenames[i]]['omega_F'][0]
            max_w = df[my_filenames[i]]['omega_F'][len(df[my_filenames[i]]['omega_F'])-1]
            cur_xspace = np.linspace(min_w, max_w, 100)
            cur_yspace = simple_amp(cur_xspace, global_parameters[my_filenames[i]]["omega_0"], global_parameters[my_filenames[i]]["F"], global_parameters[my_filenames[i]]["m_1"])
            simple_amps[my_filenames[i]] = [cur_xspace, cur_yspace]
        else:
            simple_amps[my_filenames[i]] = [0,0]

    print("---------------------------------------")

    #Find dimensions of the graph
    #print(type(df['vary_l0']))
    #global max_theta_val
    max_theta_val = 0
    for key, item in df.items():
        if max_theta_val < max(item['max_theta']):
            max_theta_val = max(item['max_theta'])
    """max_ang_f_val = 0
    for key, item in df.items():
        if max_ang_f_val < max(item['ang_f_1']):
            max_ang_f_val = max(item['ang_f_1'])
    for key, item in df.items():
        if max_ang_f_val < max(item['ang_f_2']):
            max_ang_f_val = max(item['ang_f_2'])"""
    max_ang_f_1_val = 0
    for key, item in df.items():
        if max_ang_f_1_val < max(item['ang_f_1']):
            max_ang_f_1_val = max(item['ang_f_1'])
    max_ang_f_2_val = 0
    for key, item in df.items():
        if max_ang_f_2_val < max(item['ang_f_2']):
            max_ang_f_2_val = max(item['ang_f_2'])
    min_ang_f_1_val = max_ang_f_1_val
    for key, item in df.items():
        if min_ang_f_1_val > min(item['ang_f_1']):
            min_ang_f_1_val = min(item['ang_f_1'])
    min_ang_f_2_val = max_ang_f_2_val
    for key, item in df.items():
        if min_ang_f_2_val > min(item['ang_f_2']):
            min_ang_f_2_val = min(item['ang_f_2'])

    if max_ang_f_2_val > max_ang_f_1_val:
        max_ang_f_val = max_ang_f_2_val
    else:
        max_ang_f_val = max_ang_f_1_val
    if min_ang_f_2_val < min_ang_f_1_val:
        min_ang_f_val = min_ang_f_2_val
    else:
        min_ang_f_val = min_ang_f_1_val


    # ------------------------------------------------
    # ------------- PHYSICAL ANALYSIS ----------------
    # ------------------------------------------------


    # Find the two resonant frequencies
    # Local maximum -> f'(x) * f'(x+1) < -1   (the tangent rotates by more than pi/2 rad)

    max_theta_list = {}
    resonant_frequency_list = {}
    resonant_frequency_index_list = {}
    for key, item in df.items():
        cur_max_thetas = item['max_theta'].to_numpy()
        cur_omega_F    = item['omega_F'  ].to_numpy()
        max_theta_list[key] = cur_max_thetas
        print("Analyzing the resonant frequencies of", key)
        resonant_frequency_list[key] = []
        resonant_frequency_index_list[key] = []
        for i in range(len(cur_max_thetas) - 2):
            if (cur_max_thetas[i + 2] - cur_max_thetas[i + 1]) * (cur_max_thetas[i + 1] - cur_max_thetas[i]) < (-1 * (cur_omega_F[i + 2] - cur_omega_F[i + 1]) * (cur_omega_F[i + 1] - cur_omega_F[i]) ):
                resonant_frequency_list[key].append(cur_omega_F[i + 1])
                resonant_frequency_index_list[key].append(i + 1)
                #print(cur_omega_F[i + 1])
        if len(resonant_frequency_list[key]) >= 2:
            print("1st =%6.3f" % resonant_frequency_list[key][0], "rad/s; 2nd =%6.3f" % resonant_frequency_list[key][1], "rad/s")
        print(resonant_frequency_index_list[key])
    
    # Use the algebraically derived formulas to compute the predicted resonant frequencies

    predicted_resonant_frequency_list = {}
    for key, item in df.items():
        predicted_resonant_frequency_list[key] = []

        g = global_parameters[key]["g"]
        m_1 = global_parameters[key]["m_1"]
        l_1 = global_parameters[key]["l_1"]
        m_2 = global_parameters[key]["m_2"]
        l_2 = global_parameters[key]["l_2"]
    
        predicted_resonant_frequency_list[key].append(np.sqrt(g / (l_1 + l_2)))
        #predicted_resonant_frequency_list[key].append(np.sqrt((m_1 * l_1 + m_2 * (l_1 - l_2)) * g / (m_1 * l_1 * l_1 + m_2 * (l_1 * l_1 - l_2 * l_2))))

    # Read the trajectories of the observed resonant datapoints (we expect these to behave as normal modes)

    print("---------------------------------------")
    trajectory_df = {}
    n_estimate_list = {}

    for key, item in df.items():
        trajectory_df[key] = []
        n_estimate_list[key] = []
    
        print("Approximating the n value for resonant trajectories of", key)
        for res_f_i_i in range(len(resonant_frequency_index_list[key])):
            res_f_i = resonant_frequency_index_list[key][res_f_i_i]
            cur_filename = "Data/trajectory_" + key + "/trajectory_" + str(res_f_i + 1) + '.csv'
            trajectory_df[key].append(pd.read_csv(cur_filename))
            # Curve fit absolute value
            params, cov = optimize.curve_fit(lambda x, a: a, trajectory_df[key][-1]['theta_1'], trajectory_df[key][-1]['n'])
            n_estimate_list[key].append(params[0])
            print("n_" + str(res_f_i_i + 1) + " =", params[0])


    # Do smoothing analysis

    #for key, item in df.items():
    #    smoothing(key, item, rect, 2.0)



    # ------------------------------------------------
    # ------------------ GRAPHS ----------------------
    # ------------------------------------------------

    # --------------------------------------------------------------
    # ----- Graph functions (supporting dynamic configuration) -----
    # --------------------------------------------------------------

    graph_titles = ["Maximal amplitude", "Angular frequencies", "Angular frequency space", "Normal mode amplitude coefficient", "Normal mode amplitude space", "Standard deviation of half-periods"]
    graph_function_list = {}

    # ------------ maximal amplitude graph -----------

    def max_amp_graph_func(plt):

        plt.title("Maximal amplitude")
        plt.ylim(0, max_theta_val * 1.1)
        plt.xlabel("$\omega_F$ [rad.s$^{-1}$]")
        plt.ylabel("maximal $\\theta_1$ [rad]")

        max_amp_plotline_list = {}
        max_amp_plotline_color_list = {}

        for i in range(len(my_filenames)):
            item = df[my_filenames[i]]
            if analyze[my_filenames[i]]:
                #print(simple_amps[i][0])
                #print(simple_amps[i][1])
                plt.plot(simple_amps[my_filenames[i]][0], simple_amps[my_filenames[i]][1], '-', label=my_filenames[i] + " resonance")
            max_amp_plotline_list[my_filenames[i]] = plt.plot(item['omega_F'], item['max_theta'], '.-', label=my_filenames[i] + " data")
            max_amp_plotline_color_list[my_filenames[i]] = max_amp_plotline_list[my_filenames[i]][0].get_color()
            # add resonant frequencies
            # observed resonant frequencies
            #for res_f in resonant_frequency_list[my_filenames[i]]:
            #    plt.axvline(x=res_f, linestyle='dotted')
            # predicted resonant frequencies
            for pred_res_f in predicted_resonant_frequency_list[my_filenames[i]]:
                plt.axvline(x=pred_res_f, linestyle='dotted', color=max_amp_plotline_color_list[my_filenames[i]], label = my_filenames[i] + " predicted res f")
        plt.legend()


    graph_function_list['max_amp_graph'] = max_amp_graph_func

    # --------- average mechanical energy graph --------

    def avg_E_graph_func(plt):

        plt.title("Time-average mechanical energy")
        plt.xlabel("$\omega_F$ [rad.s$^{-1}$]")
        plt.ylabel("$\langle E\\rangle$ [J]")
        for i in range(len(my_filenames)):
            item = df[my_filenames[i]]
            plt.plot(item['omega_F'], item['avg_E'], '.-', label=my_filenames[i] + " data")
            # observed resonant frequencies
            for res_f in resonant_frequency_list[my_filenames[i]]:
                plt.axvline(x=res_f, linestyle='dotted')
            # predicted resonant frequencies
            #for pred_res_f in predicted_resonant_frequency_list[my_filenames[i]]:
            #    plt.axvline(x=pred_res_f, linestyle='dotted', color=max_amp_plotline_color_list[my_filenames[i]], label = my_filenames[i] + " predicted res f")
        plt.legend()

    graph_function_list['avg_E_graph'] = avg_E_graph_func

    # ------------ angular frequency graph -----------

    def ang_f_graph_func(plt):

        plt.title("Angular frequencies")
        plt.ylim(0, max_ang_f_val * 1.1)
        plt.xlabel("$\omega_F$ [rad.s$^{-1}$]")
        plt.ylabel("ang. freq. of pend. [rad.s$^{-1}$]")
        for i in range(len(my_filenames)):
            item = df[my_filenames[i]]
            plt.plot(item['omega_F'], item['ang_f_1'], '.', label=my_filenames[i] + " data - 1st pend.")
            plt.plot(item['omega_F'], item['ang_f_2'], '.', label=my_filenames[i] + " data - 2nd pend.")
            # add resonant frequencies
            for res_f in resonant_frequency_list[my_filenames[i]]:
                plt.axvline(x=res_f, linestyle='dotted')
                plt.axhline(y=res_f, linestyle='dotted')
        plt.legend()

    graph_function_list['ang_f_graph'] = ang_f_graph_func

    # --------- angular frequency space graph ---------

    def ang_f_space_graph_func(plt):

        plt.title("Angular frequency space")
        border_coefficient = 1.05
        plt.xlim(min_ang_f_1_val - max_ang_f_1_val * (border_coefficient - 1.0), max_ang_f_1_val * border_coefficient)
        plt.ylim(min_ang_f_2_val - max_ang_f_1_val * (border_coefficient - 1.0), max_ang_f_2_val * border_coefficient)
        plt.xlabel("ang. freq. of 1st pend. [rad.s$^{-1}$]")
        plt.ylabel("ang. freq. of 2nd pend. [rad.s$^{-1}$]")
        ang_f_space = np.linspace(0, max_ang_f_val * border_coefficient, 50)
        plt.plot(ang_f_space, ang_f_space, linestyle='dashed', label = "Equal frequency line")
        for i in range(len(my_filenames)):
            item = df[my_filenames[i]]
            plt.plot(item['ang_f_1'], item['ang_f_2'], '-', label=my_filenames[i] + " data")
            # add resonant frequencies
            for res_f_i in resonant_frequency_index_list[my_filenames[i]]:
                plt.plot(item['ang_f_1'][res_f_i],item['ang_f_2'][res_f_i],'x',markersize=15, markeredgewidth=2)

        plt.legend()

    graph_function_list['ang_f_space_graph'] = ang_f_space_graph_func

    # ---------- normal mode amplitude coefficient graph ---------

    def NM_amp_coef_graph_func(plt):

        plt.title("Normal mode amplitude coefficient")
        #plt.ylim(0, max_ang_f_val * 1.1)
        plt.xlabel("$\\theta_1$ [rad]")
        plt.ylabel("n = $\\theta_2$ / $\\theta_1$ [dimensionless]")
        for i in range(len(my_filenames)):
            for trajectory_index in range(1, len(trajectory_df[my_filenames[i]])):
                trajectory = trajectory_df[my_filenames[i]][trajectory_index]
                plt.plot(trajectory['theta_1'], trajectory['n'], '.', label=my_filenames[i] + ": res. traj. " + str(trajectory_index + 1))
        
                plt.axhline(y=n_estimate_list[my_filenames[i]][trajectory_index], linestyle='dotted')
        plt.legend()

    graph_function_list['NM_amp_coef_graph'] = NM_amp_coef_graph_func

    # ------------ normal mode amplitude space graph -----------

    def NM_amp_space_graph_func(plt):

        plt.title("Normal mode amplitude space")
        #plt.ylim(0, max_ang_f_val * 1.1)
        plt.xlabel("$\\theta_1$ [rad]")
        plt.ylabel("$\\theta_2$ [rad]")
        theta_n_space = np.linspace(-max_theta_val * 1.1, max_theta_val * 1.1, 100)
        scales = []
        for i in range(len(my_filenames)):
            for trajectory_index in range(0, len(trajectory_df[my_filenames[i]])):
                trajectory = trajectory_df[my_filenames[i]][trajectory_index]
                scales.append(plt.plot(trajectory['theta_1'], trajectory['theta_2'], '.', label=my_filenames[i] + ": res. traj. " + str(trajectory_index + 1)))
            
                plt.plot(theta_n_space, theta_n_space * n_estimate_list[my_filenames[i]][trajectory_index], linestyle='dashed')
        plt.plot(theta_n_space, theta_n_space * 1.0, linestyle='dashed')
        autoscale_based_on(plt.gca(), scales)
        plt.legend()

    graph_function_list['NM_amp_space_graph'] = NM_amp_space_graph_func

    # --------- standard deviation of halfperiods graph --------

    def std_hp_graph_func(plt):

        plt.title("Standard deviation of half-periods")
        plt.xlabel("$\omega_F$ [rad.s$^{-1}$]")
        plt.ylabel("$\sigma$ [dimensionless]")
        for i in range(len(my_filenames)):
            item = df[my_filenames[i]]
            plt.plot(item['omega_F'], item['hp_1_std'], '.-', label=my_filenames[i] + " data - 1st pend.")
            plt.plot(item['omega_F'], item['hp_2_std'], '.-', label=my_filenames[i] + " data - 2nd pend.")
            # observed resonant frequencies
            for res_f in resonant_frequency_list[my_filenames[i]]:
                plt.axvline(x=res_f, linestyle='dotted')
            # predicted resonant frequencies
            #for pred_res_f in predicted_resonant_frequency_list[my_filenames[i]]:
            #    plt.axvline(x=pred_res_f, linestyle='dotted', color=max_amp_plotline_color_list[my_filenames[i]], label = my_filenames[i] + " predicted res f")
        plt.legend()

    graph_function_list['std_hp_graph'] = std_hp_graph_func

    # --------- Graphing the given configuration -----------

    active_graph_list = []
    for key, item in graph_configuration.items():
        if item == 'Y':
            active_graph_list.append(key)

    plt.figure(figsize=(15, 8))
    subplot_x_dimension, subplot_y_dimension = subplot_dimensions(len(active_graph_list))


    for i in range(len(active_graph_list)):

        plt.subplot(subplot_y_dimension, subplot_x_dimension, i + 1)
        graph_function_list[active_graph_list[i]](plt)


    plt.tight_layout()
    plt.savefig("Outputs/my_output.png")
    plt.show()


# ------------------------------------------------
# --------------- LOADING DATA -------------------
# ------------------------------------------------


df = {}
analyze = {}

graph_configuration = {'max_amp_graph' : 'Y', 'avg_E_graph' : 'Y', 'ang_f_graph' : 'Y', 'ang_f_space_graph' : 'Y', 'NM_amp_coef_graph' : 'Y', 'NM_amp_space_graph' : 'Y', 'std_hp_graph' : 'Y'}

command_doc = [
            ['title'     , '1 argument: new graph title'                      , "doesn't do anything"],
            ['rename'    , '2 arguments: old datafile name, new datafile name', "renames a datafile and updates all important metadata"],
            ['visible'   , '0 arguments'                                      , "lists visible and hidden graphs"],
            ['configure' , '0 arguments'                                      , "configure the visibility settings of available graphs"],
            ['about'     , '0 arguments'                                      , "provides a longer documentation of this program, as well as its context"],
            ['copyright' , '0 arguments'                                      , "provides the copyright information about this program"],
            ['help'      , '0 arguments'                                      , "provides a list of available commands"],
            ['exit'      , '0 arguments'                                      , "exits the program"]
        ]

title_delim = ' '

def get_yn(question):
    while(True):
        response = input(question).upper()
        if response == 'Y' or response == 'YES':
            return(True)
        if response == 'N' or response == 'NO':
            return(False)
        print("Type either 'Y' (meaning 'yes') or 'N' (meaning 'no')")

while True:
    my_filename_list = input("Please enter a valid filename list (Searching in 'Data/'): ")
    if my_filename_list.split(title_delim)[0] == 'title':
        graph_title = str(title_delim.join(my_filename_list.split(' ')[1:]))
        print("Graph title updated!")
    elif my_filename_list.split(title_delim)[0] == 'rename':
        try:
            old_title = my_filename_list.split(' ')[1]
            new_title = my_filename_list.split(' ')[2]
            # We need to change the aggregate data filename, trajectory data folder name, and parameter map name
            try:
                try:
                    rename_config_file = open("Data/config/__PARAM_MAP__.txt", 'r')
                    rename_config_lines = rename_config_file.readlines()
                    rename_my_lines = []
                    for line in rename_config_lines:
                        elements = line.split(": ", 1)
                        if elements[0] == old_title:
                            rename_my_lines.append(new_title + ": " + elements[1])
                            print("Old filename found in parameter map and updated")
                        else:
                            rename_my_lines.append(line)
                    rename_config_file.close()
                    corrected_config_file = open("Data/config/__PARAM_MAP__.txt", 'w')
                    for line in rename_my_lines:
                        corrected_config_file.write(line)
                    corrected_config_file.close()
                except FileNotFoundError:
                    print("Config file not found")
                    quit()
                os.rename("Data/" + old_title + ".csv", "Data/" + new_title + ".csv")
                print("Aggregate data filename updated")
                os.rename("Data/trajectory_" + old_title, "Data/trajectory_" + new_title)
                print("Trajectory data folder name updated")
                
            except FileNotFoundError:
                print("Datafile and trajectory folder not found")
        except IndexError:
            print("ERROR: Two string inputs after the command expected")
    elif my_filename_list.split(title_delim)[0] == 'visible':
        print("")
        cur_visible_graphs = ""
        cur_hidden_graphs  = ""
        for key, item in graph_configuration.items():
            if item == 'Y':
                cur_visible_graphs += key + ", "
            else:
                cur_hidden_graphs += key + ", "
        print("Currently visible graphs: " + cur_visible_graphs[:-2])
        print("Currently hidden graphs:  " + cur_hidden_graphs[:-2])
        print("")
    elif my_filename_list.split(title_delim)[0] == 'configure':
        print("")
        print("For each available graph, determine whether it should be visible [Y/N]")
        g_c_copy = dict(graph_configuration)
        for key, item in g_c_copy.items():
            if get_yn(key + ": "):
                graph_configuration[key] = 'Y'
            else:
                graph_configuration[key] = 'N'
        print("")
    elif my_filename_list.split(title_delim)[0] == 'help':
        print("\nInput a list of filenames (omitting the '.csv' extension) separated by a comma and a space.")
        print("Adding an asterisk in front of a filename will omit some derived plotlines, such as the projected motional spectrum of the upper segment, for that specific datafile.")
        print("Inputting an empty line will load the filename list used in last session.")
        print("Alternatively, type one of the following commands followed by the required arguments, all in a single line separated by a space.")
        for command_info in command_doc:
            print("  '" + command_info[0] + "':")
            print("    " + command_info[1])
            print("    " + command_info[2])
    elif my_filename_list.split(title_delim)[0] == 'about':
        try:
            about_file  = open("meta_config/damping_analyzer_about_file.txt", 'r')
            about_lines = about_file.readlines()
            print("")
            for line in about_lines:
                print(line[:-1])
            about_file.close()
            print("")
        except FileNotFoundError:
            print("'About' file not found")
    elif my_filename_list.split(title_delim)[0] == 'copyright':
        try:
            copyright_file  = open("meta_config/damping_analyzer_copyright_file.txt", 'r')
            copyright_lines = copyright_file.readlines()
            print("")
            for line in copyright_lines:
                print(line[:-1])
            copyright_file.close()
            print("")
        except FileNotFoundError:
            print("'Copyright' file not found")
    elif my_filename_list.split(title_delim)[0] == 'exit' or my_filename_list.split(title_delim)[0] == 'quit' or my_filename_list.split(title_delim)[0] == 'q':
        print("Exiting...")
        quit()

    else:
        if my_filename_list == '':
            print("Loading last saved session...")
            meta_config_file = open("meta_config/damping_analyzer_META_CONFIG.txt", mode="r")
            my_filename_list = meta_config_file.readlines()[0]
            meta_config_file.close()
        try:
            my_filenames = my_filename_list.split(', ')
            for i in range(len(my_filenames)):
                if my_filenames[i][0] == '*':
                    my_filenames[i] = my_filenames[i][1:]
                    analyze[my_filenames[i]] = False
                else:
                    analyze[my_filenames[i]] = True
            #df.append(pd.read_csv("Data/" + str(my_filenames[i]) + ".csv"))
                df[my_filenames[i]] = pd.read_csv("Data/" + str(my_filenames[i]) + ".csv")
            # --------- perform the main body of code --------
            main_analysis()
            # --------- reset variables ------------
            df = {}
            analyze = {}
        except FileNotFoundError:
            print("Datafiles not found. Try typing 'help' to see the list of available commands and documentation")
            df = {}



