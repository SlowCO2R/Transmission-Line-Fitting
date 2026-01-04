# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 13:02:40 2024

@author: Mike Carroll
"""
import pandas as pd
import math
import matplotlib
import numpy as np
import os
import matplotlib.pyplot as plt
import re
from scipy.optimize import curve_fit
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl import load_workbook
from scipy.optimize import fsolve


#%conda install xlrdcv

#%%
#Load the desired folder
folder_path = r"Y:/5900/HydrogenTechFuelCellsGroup/CO2R/Nhan P/Experiments/CO2 Cell Testing/TS2/2NP10"

#%% Read DTA files

def read_DTA(path):
    """
    Inputs: path to desired .DTA data file
    Outputs: ZCurve from DTA file
    """
    # Initialize variables to store table data
    table_data = []
    in_table = False

    with open(path, encoding='windows-1252') as fp:
        for i, line in enumerate(fp):
            if 'ZCURVE' in line:
                in_table = True
                starting_line = i + 4  # Assuming the table starts four lines after 'ZCURVE'
            elif in_table and line.strip() and not line.startswith('#'):  # Skip lines starting with '#'
                table_data.append(line.strip().split('\t'))
#Look for Zcurve line in the data file
    # Check if 'ZCURVE' table is found
    if not in_table:
        raise ValueError("ZCURVE table not found in DTA file")

    # Create a DataFrame from the table data, skipping lines starting with '#'
    table_data_cleaned = [row for row in table_data if not any(entry.startswith('#') for entry in row)]
    
    # Check if there is any non-numeric data in the table
    try:
        df = pd.DataFrame(table_data_cleaned[1:], columns=table_data_cleaned[0]).astype(float)
    except ValueError as e:
        #print(f"Error converting to float: {e}")
        #print(f"Data causing the error: {table_data_cleaned}")
        raise  # Re-raise the exception

    return df


all_data_dict = {}

for filename in os.listdir(folder_path):
    if filename.endswith(".DTA"):
        file_path = os.path.join(folder_path, filename)
        try:
            df = read_DTA(file_path)

            # Multiply 'Zimag' by -1 if it exists in columns
            if 'Zimag' in df.columns:
                df['Zimag'] *= -1

            # Store the DataFrame in the dictionary with the file name as the key
            all_data_dict[filename] = df
        except ValueError as e:
            print(f"Skipping file {filename} due to error: {e}")

def custom_sort_key(file_name):
    # Extract the numbers before "RH" and after "#"
    match = re.search(r'(\d+)RH_FI_#(\d+)', file_name)
    if match:
        num_before_rh = int(match.group(1))
        num_after_hash = int(match.group(2))
        return (num_before_rh, num_after_hash)
    else:
        # Handle cases where the pattern is not found
        return (float('inf'), float('inf'))

# Sort the keys based on the custom sorting function
sorted_keys = sorted(all_data_dict.keys(), key=custom_sort_key)

# Create a new dictionary with sorted keys
sorted_data_dict = {key: all_data_dict[key] for key in sorted_keys}

# Now sorted_data_dict contains the items sorted based on the custom criteria
#sorted_data_dict

#%% SECTION 2 - Create Function to read DTA files into Python
low_freq_min = 100
low_freq_max = 200
mid_freq_min = 200
mid_freq_max = 500
#mid_freq_min2 = 100
#mid_freq_max2 = 200
high_freq_min = 1000
high_freq_max = 10000
plot_high_freq_min = high_freq_min / 5
cell_area = 25
for file_name, df in sorted_data_dict.items():
  

    # Calculate 'omega' from 'Freq'
    df['omega'] = 2 * np.pi * df['Freq']

    fig, (ax, ax1) = plt.subplots(1, 2, figsize=(12, 5))
    ax.plot(df['Zreal'], df['Zimag'], 'rx')
    ax.plot(df['Zreal'][(df['Freq'] > high_freq_min) & (df['Freq'] < high_freq_max)],
            df['Zimag'][(df['Freq'] > high_freq_min) & (df['Freq'] < high_freq_max)],
            'kx', label='Current High Frequency Selection')
    ax.plot(df['Zreal'][(df['Freq'] > mid_freq_min) & (df['Freq'] < mid_freq_max)],
            df['Zimag'][(df['Freq'] > mid_freq_min) & (df['Freq'] < mid_freq_max)],
            's', label='Current Mid Frequency Selection')
    #ax.plot(df['Zreal'][(df['Freq'] > mid_freq_min2) & (df['Freq'] < mid_freq_max2)],
            #df['Zimag'][(df['Freq'] > mid_freq_min2) & (df['Freq'] < mid_freq_max2)],
            #'s', label='Current Mid Frequency 2 Selection')
    ax.plot(df['Zreal'][(df['Freq'] < low_freq_max) & (df['Freq'] > low_freq_min)],
            df['Zimag'][(df['Freq'] < low_freq_max) & (df['Freq'] > low_freq_min)],
            'bx', label='Current Low Frequency Selection')
    ax.legend()
    ax.set_xlabel('Zreal (ohm)')
    ax.set_ylabel('Zimag (ohm)')
    ax.set_title(f'Nyquist Plot for {file_name}')
    ax.plot(xlim=[0.01, 0.1])

    ax1.plot(1 / df['omega'][(df['Freq'] < low_freq_max) & (df['Freq'] > low_freq_min)]**2,
             1 / (df['omega'][(df['Freq'] < low_freq_max) & (df['Freq'] > low_freq_min)] *
                  np.abs(df['Zimag'][(df['Freq'] < low_freq_max) & (df['Freq'] > low_freq_min)])),
             'rx')
    ax1.set_title('Low Frequency Spectrum')
    ax1.set_xlabel('$1/\omega^2$')
    ax1.set_ylabel('$1/\omega*|Z_{imag}|$')

    mid_freq_df = df[(df['Freq'] > mid_freq_min) & (df['Freq'] < mid_freq_max)]
    high_freq_df = df[(df['Freq'] > high_freq_min) & (df['Freq'] < high_freq_max)]

    fig2, (ax, ax1) = plt.subplots(1, 2, figsize=(12, 5))
    ax.plot(1 / mid_freq_df['omega']**2,
            1 / (mid_freq_df['omega'] * np.abs(mid_freq_df['Zimag'])),
            'rx')
    ax.set_title('Mid Frequency Spectrum')
    ax.set_xlabel('$1/\omega^2$')
    ax.set_ylabel('$1/\omega*|Z_{imag}|$')

    ax1.plot(mid_freq_df['Zreal'],
             mid_freq_df['Zimag'],
             'kx', markersize=10, label='Current Selection')
    ylim = np.max(np.abs(df['Zimag'][(df['Freq'] > low_freq_min) & (df['Freq'] < mid_freq_max)]))
    ylim_high_freq = np.max(np.abs(df['Zimag'][(df['Freq'] > plot_high_freq_min) & (df['Freq'] < high_freq_max)]))
    sc = ax1.scatter(mid_freq_df['Zreal'][(mid_freq_df['Zimag'] <= ylim_high_freq)],
                     mid_freq_df['Zimag'][(mid_freq_df['Zimag'] <= ylim_high_freq)],
                     c=mid_freq_df['Freq'][(mid_freq_df['Zimag'] <= ylim_high_freq)])
    plt.colorbar(sc, label='Frequency [Hz]')
    ax1.set_xlabel('$Z_{real}$ $[\Omega]$')
    ax1.set_ylabel('$Z_{imag}$ $[\Omega]$')
    ax1.set_title('Mid Frequency Portion of Nyquist Plot')
    ax1.legend()
    
#%% Fitting
EIS_parameters = pd.DataFrame
dfs = []
for file_name, df in sorted_data_dict.items():
    
    # Calculate 'omega' from 'Freq'
    df['omega'] = 2 * np.pi * df['Freq']

    idx = np.abs(df['Zimag']).argmin() #Find the index where Zimag crosses Zreal axis (closest to 0)
    
    #This codeline will fail if idx = 0 and counter = -1 --> Keyerror:-1 --> Out of bound index
    if df['Zimag'][idx] > 0:
        counter = 1
    elif df['Zimag'][idx] < 0:
        counter = -1
    else:
        counter = 0    
    
    slope = (df['Zimag'][idx + counter] - df['Zimag'][idx]) / (df['Zreal'][idx + counter] - df['Zreal'][idx]) #Calculate the slope between the zero-crossing point and the next point
    b = df['Zimag'][idx + counter] - slope * df['Zreal'][idx + counter]
    x_0 = -b / slope # solve for Zreal value where Zimag = 0, or HFR

    #if df['Zimag'][idx] > 0:
        #counter = 1
        #if idx + counter >= len(df):
            #counter = -1
    #elif df['Zimag'][idx] < 0:
        #counter = -1
        #if idx + counter < 0:
            #counter = 1
    #else:       # Zimag is exactly zero
        #x_0 = df['Zreal'][idx]
        #slope = None
        #b = None
    #else_block_executed = False
    
    #if 'x_0' not in locals():
        #try:
            #delta_zimag = df['Zimag'].iloc[idx + counter] - df['Zimag'].iloc[idx]
            #delta_zreal = df['Zreal'].iloc[idx + counter] - df['Zreal'].iloc[idx]
            #slope = delta_zimag / delta_zreal
            #b = df['Zimag'].iloc[idx + counter] - slope * df['Zreal'].iloc[idx + counter]
            #x_0 = -b / slope
        #except ZeroDivisionError:
            #x_0 = df['Zreal'].iloc[idx]  # fallback if division by zero
            #slope = None
            #b = None

    print("Data for", file_name)
    print(f"The ohmic resistance calculated from linear interpolation around Zimag=0 is: {x_0 * cell_area:.4f} Ω-cm2")

    #Define the impedance from specified frequency ranges
    high_freq_Zreal = df['Zreal'][(df['Freq'] < high_freq_max) & (df['Freq'] > high_freq_min)]
    high_freq_Zimag = df['Zimag'][(df['Freq'] < high_freq_max) & (df['Freq'] > high_freq_min)]
    mid_freq_Zreal = df['Zreal'][(df['Freq'] < mid_freq_max) & (df['Freq'] > mid_freq_min)]
    mid_freq_Zimag = df['Zimag'][(df['Freq'] < mid_freq_max) & (df['Freq'] > mid_freq_min)]
    #mid_low_freq_Zreal = df['Zreal'][(df['Freq'] < mid_freq_max2) & (df['Freq'] > mid_freq_min2)]
    #mid_low_freq_Zimag = df['Zimag'][(df['Freq'] < mid_freq_max2) & (df['Freq'] > mid_freq_min2)]
    low_freq_Zreal = df['Zreal'][(df['Freq'] < low_freq_max) & (df['Freq'] > low_freq_min)]
    low_freq_Zimag = df['Zimag'][(df['Freq'] < low_freq_max) & (df['Freq'] > low_freq_min)]
    omega_low_freq = df['omega'][(df['Freq'] < low_freq_max) & (df['Freq'] > low_freq_min)]
    Zimag_low_freq = df['Zimag'][(df['Freq'] < low_freq_max) & (df['Freq'] > low_freq_min)]

    #fit transmission line
    def fit_TL(x, m, c):
        y = m * x - c #Can change function for curve fit
        return y

    parameters_TL, covariance_TL = curve_fit(fit_TL, high_freq_Zreal, high_freq_Zimag) #Fit the custom linear function to HFR- parameters_TL = slope + intercept, coveariance = fit parameters
    Std_TL = np.sqrt(np.diag(covariance_TL)) #Standard deviation of slope and y-intercept
    Res_TL = high_freq_Zimag - fit_TL(np.array(high_freq_Zreal), parameters_TL[0], parameters_TL[1]) #Goodness of fit between actual and predicted Zimag
    Sq_res_TL = np.sum(Res_TL**2)
    Sq_sum_TL = np.sum((high_freq_Zimag - np.mean(high_freq_Zimag))**2)
    R2_TL = 1 - (Sq_res_TL / Sq_sum_TL)
    low_freq_x_var = 1 / (omega_low_freq**2)            #Convert Zimag to 
    low_freq_y_var = 1 / (omega_low_freq * Zimag_low_freq)

    
    print(f"The fit angle at specified high frequency is: {180 * np.arctan(parameters_TL[0]) / np.pi:.2f}") #convert slope to fit angle
    print(f"Calculated normalized ohmic resistance (R_ohmic) from the fit is: {cell_area * parameters_TL[1] / parameters_TL[0]:.3f} Ω-cm2")
    print(f"R2 of the fit is: {R2_TL:.2f}")
    print("Parameters_TL:", parameters_TL)

    #Fit low frequency to acquare CDL and RCT
    def fit_low_freq(x, m, c):
        y = m * x + c
        return y

    parameters_low_freq, covariance_low_freq = curve_fit(fit_low_freq, low_freq_x_var, low_freq_y_var)

    capacitance = parameters_low_freq[1] / cell_area
    R_CT = np.sqrt(1 / (parameters_low_freq[0] * parameters_low_freq[1])) * cell_area

    Std1 = np.sqrt(np.diag(covariance_low_freq))
    Res_low_freq = low_freq_y_var - fit_low_freq(np.array(low_freq_x_var), parameters_low_freq[0],
                                                 parameters_low_freq[1])
    Sq_res_low_freq = np.sum(Res_low_freq**2)
    Sq_sum_low_freq = np.sum((low_freq_y_var - np.mean(low_freq_y_var))**2)
    R2_low_freq = 1 - (Sq_res_low_freq / Sq_sum_low_freq)

    print(f"Area normalized capacitance (C): {capacitance * 1e6:.6f} uF/cm2")
    print(f"Area normalized R_CT: {R_CT:.2f} Ω-cm2")
    print(f'R2 of the fit is: {R2_low_freq:.2f}')

    def fit_mid_freq(x, m, c):
        y = m * x - c
        return y

    parameters_mid_freq, covariance_mid_freq = curve_fit(fit_mid_freq, mid_freq_Zreal, mid_freq_Zimag)

    # Generate additional x-values for the extended line
    extended_x_values = np.linspace(min(df['Zreal']), max(df['Zreal']), 100)

    # Calculate corresponding y-values for the extended line
    extended_y_values_mid = fit_mid_freq(extended_x_values, parameters_mid_freq[0], parameters_mid_freq[1])

    #Find intercept of low vs mid frequency
    def find_intersection(x, m1, c1, m2, c2):
        return m1 * x - c1 - (m2 * x - c2)

    x_intersection = fsolve(find_intersection, x0=0, args=(parameters_mid_freq[0], parameters_mid_freq[1], parameters_low_freq[0], parameters_low_freq[1]))

    R_sheet = ((x_intersection - x_0) * cell_area)
    
    # Define the function for the mid-low frequency fit
    #def fit_low_freq(x, m, c):
        #y = m * x - c
        #return y
    
    #parameters_mid_low_freq, covariance_mid_low_freq = curve_fit(fit_mid_low_freq, mid_low_freq_Zreal, mid_low_freq_Zimag)
    #parameters_low_freq, covariance_low_freq = curve_fit(fit_low_freq, low_freq_Zreal, low_freq_Zimag)
    # Calculate corresponding y-values for the mid-low frequency line
    y_values_low_freq = fit_low_freq(extended_x_values, parameters_low_freq[0], parameters_low_freq[1])
    
    # R2 calculation for mid-low frequency
    Std_low_freq = np.sqrt(np.diag(covariance_low_freq))
    Res_low_freq = low_freq_Zimag - fit_low_freq(np.array(low_freq_Zreal), parameters_low_freq[0], parameters_low_freq[1])
    Sq_res_low_freq = np.sum(Res_low_freq**2)
    Sq_sum_low_freq = np.sum((low_freq_Zimag - np.mean(low_freq_Zimag))**2)
    R2_low_freq = 1 - (Sq_res_low_freq / Sq_sum_low_freq)
    
    # Define the function for finding intersection
    #def find_intersection(x, m1, c1, m2, c2):
        #return m1 * x - c1 - (m2 * x - c2)
    
    # Find the intersection point
    #x_intersection = fsolve(find_intersection, x0=0, args=(parameters_mid_freq[0], parameters_mid_freq[1], parameters_low_freq[0], parameters_low_freq[1]))
    
    # Calculate the sheet resistance
    #R_sheet = ((x_intersection - (parameters_TL[1] / parameters_TL[0])) * cell_area)


    print("The sheet resistance is:", float(R_sheet), "\u03A9cm2")
    print(f"The fit angle at mid frequency is: {180 * np.arctan(parameters_mid_freq[0]) / np.pi:.2f}")
    plt.show()  # Display the plot
    # Plotting the High-Frequency Plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # Create a 1x3 grid of subplots
    
    # Create a dictionary with relevant parameters
    parameters_dict = {
        'file_name': file_name,
        'R_HFR_interpolation': x_0 * cell_area,
        'fit_angle': 180 * np.arctan(parameters_mid_freq[0]) / np.pi,
        'R_HFR_extrapolation': cell_area * parameters_TL[1] / parameters_TL[0],
        'capacitance': capacitance * 1e6,
        'R_CT': R_CT,
        'R_sheet': float(R_sheet),
    }

    # Convert the dictionary to a DataFrame and append it to the list
    dfs.append(pd.DataFrame([parameters_dict]))

    # Concatenate all DataFrames in the list
    All_data = pd.concat(dfs, ignore_index=True)
    EIS_data = All_data[['file_name', 'R_HFR_interpolation', 'R_sheet', 'capacitance', 'R_CT']]

    # Plot 1: High-Frequency Portion
    axs[0].scatter(df['Zreal'][(df['Zimag'] <= ylim_high_freq)], df['Zimag'][(df['Zimag'] <= ylim_high_freq)],
                   c=df['Freq'][(df['Zimag'] <= ylim_high_freq)], label='Data')
    axs[0].plot(df['Zreal'][(df['Freq'] > high_freq_min) & (df['Freq'] < high_freq_max)],
                df['Zimag'][(df['Freq'] > high_freq_min) & (df['Freq'] < high_freq_max)],
                'kx', markersize=10, label='Current Selection')
    axs[0].plot(high_freq_Zreal, (high_freq_Zreal * parameters_TL[0] - parameters_TL[1]), 'r--', label='Linear Fit')
    axs[0].set_xlabel('$Z_{real}$ $[\Omega]$')
    axs[0].set_ylabel('$Z_{imag}$ $[\Omega]$')
    axs[0].set_title(f'High Frequency Portion of Nyquist Plot for {file_name}')
    axs[0].legend()

    # Plot 2: Low-Frequency Spectrum
    axs[1].plot(low_freq_x_var, low_freq_y_var, 'ko', label='Low Frequency Data')
    axs[1].plot(low_freq_x_var, parameters_low_freq[0] * low_freq_x_var + parameters_low_freq[1], 'r--',
                label='Linear Fit')
    axs[1].set_title('Low Frequency Spectrum')
    axs[1].set_xlabel('$1/\omega^2$')
    axs[1].set_ylabel('$-1/\omega*Z_{imag}$')
    axs[1].legend()

    # Plot 3: Mid and Low Frequency Linear Fits
    sc = axs[2].scatter(df['Zreal'], df['Zimag'], c='blue', label='Data')  # Set color to blue
    axs[2].plot(df['Zreal'][(df['Freq'] > mid_freq_max) & (df['Freq'] < low_freq_min)],
                df['Zimag'][(df['Freq'] > mid_freq_max) & (df['Freq'] < low_freq_min)],
                )
    axs[2].plot(extended_x_values, extended_y_values_mid, 'r--', label='Mid-Frequency Linear Fit')  # Plot the mid-frequency line
    axs[2].plot(extended_x_values, y_values_low_freq, 'g--', label='Low-Frequency Linear Fit')  # Plot the low-frequency line
    axs[2].set_xlabel('$Z_{real}$ $[\Omega]$')
    axs[2].set_ylabel('$Z_{imag}$ $[\Omega]$')
    axs[2].set_title('Mid and Low Frequency Linear Fits')
    axs[2].legend()

    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.show()  # Display the plot
    
EIS_data.to_clipboard(index=False, header=False, excel=True)

#%% High Frequency Calculation - Sheet and Ohmic Resistance
Cell_Values=pd.DataFrame({"HFR (\u03A9 cm2)":[x_0*cell_area],
                         "R_CL (\u03A9 cm2)":float(R_sheet),
                         "Capacitance (uF/cm2)":[capacitance*1e6],
                         "Charge Transfer Resistance (\u03A9 cm2)":(R_CT)})
Cell_Values.to_csv(folder_path + 'hello.csv')
Cell_Values.to_clipboard(index=False, header=False, excel=True)