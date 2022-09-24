# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 12:28:23 2022

@author: Diego Castillo
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import factorial
import scipy.stats as st



def extractData(file):
    """
    Extracts data from csv file and returns nunpy array of all columns of data, which correspond to one run 
    
    """
    file_data = pd.read_csv(file)
    headers = file_data.columns.values
    data_runs = []
    unique_values=[]
    for i in range(1, len(headers)):
        data_runs.append(file_data[headers[i]])
    
    data_runs = np.array(data_runs)
    
    return data_runs

def horiz_add(data_array):
    """        
    Parameters
    ----------
    data_array : numpy.array
    
    Takes array of runs/data sets and horizontally adds the corresponding value of each run
    
    Returns: array of horizontally summed data sets
    -------
    """
    summed_data = np.sum(data_array, axis = 0)
    
    return summed_data

def vertical_add(data_array):
    """        
    Parameters
    ----------
    data_array : numpy.array
    
    Takes array of runs/data sets and vertically adds the values of each run
    
    Returns: array of vertically summed data sets
    -------
    """
    summed_data = np.sum(data_array, axis = 1)
    
    return summed_data
        

def poisson_fit(x, mu):
    #return st.poisson.pmf(x, mu)   
    return (mu**x * np.exp(-mu))/(factorial(x))

def gaussian_fit(x, mean, std):
    return st.norm.pdf(x, mean, std)
    
def plot_hist(data, title):
    #determining the bins by finding the unique values in the data set
    unique_data = set(data)
    unique_counts = list(unique_data)
    max_count = max(unique_counts)
    min_count = min(unique_counts)
    temp = np.arange(min_count, max_count, 1)
    bins_arr = [bins - 0.5 for bins in temp]
    
    #mean, std dev and mu
    std = np.std(data, ddof=1)
    mean = np.mean(data)
    mu = std**2
        
    #plotting histogram
    plt.figure(figsize=(16,12))
    occurences, bin_edges, patches = plt.hist(data, bins_arr, edgecolor='k', color = 'c')
    plt.xlabel("Counts per time bin", fontsize = 16)
    plt.ylabel("Number of occurences", fontsize = 16)
    plt.grid()
    
    bin_middles = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    best_mu, cov = curve_fit(poisson_fit, bin_middles, occurences)

    #establishing domain for Gaussian and Poisson distributions
    domain = np.linspace(min(data)-0.5, max(data)-0.5, 1000)
        
    #plotting poisson fit with best fit count rate mu
    y_poisson = sum(occurences) * poisson_fit(domain, mu)    
    plt.plot(domain, y_poisson, ls='-',lw = 3,   color='r', label='Poisson best fit' )
    
    #determining error for histogram errorbars at middle of bins
    y_errors = []
    
    for y in occurences :
        y_errors.append(np.sqrt(y))    
        
    #plt.errorbar(bin_middles, occurences, yerr = y_errors, ecolor = 'b', elinewidth = 2.5, capsize=10, fmt='none')
        
    #plotting Gaussian normal fit
    plt.plot(domain, sum(occurences) * gaussian_fit(domain, mean, std), lw = 3, color = 'm', label = 'Gaussian fit')
    plt.legend()
    plt.title(title, fontweight ="bold", fontsize = 20)
    
    best_fit = poisson_fit(bin_middles, mu)
    #Plotting residuals
    plt.figure(figsize = (15,3))
    plt.plot(bin_middles, (sum(occurences) * best_fit) - occurences, 'k.')
    plt.xlabel("Counts per time bin", fontsize=16)  
    plt.ylabel("Residuals", fontsize=16)
    plt.title("Plot of residuals, " + title,
              fontweight ="bold")
    plt.axhline(color = "r", lw = 2.5)
    plt.show
    
    return mu
    
e1 = "OneDrive - McGill University\WINTER 2022\PHYS 258\LAB6_258\E1_lab6.csv"
e2 = "OneDrive - McGill University\WINTER 2022\PHYS 258\LAB6_258\E2_lab6.csv"
e3 = "OneDrive - McGill University\WINTER 2022\PHYS 258\LAB6_258\E3_lab6.csv"
e4 = "OneDrive - McGill University\WINTER 2022\PHYS 258\LAB6_258\E4_lab6.csv"
tech = "OneDrive - McGill University/WINTER 2022/PHYS 258/LAB6_258/tech.csv"


#==================================E1 analysis========================================  
e1_data = extractData(e1)
summed_e1 = horiz_add(e1_data)

#fitting least square of Gaussian and Poissonian distributions for all three runs of E1
#obtaining their respective counting rate mu and then calculating the avg counting rate
e1_rates = []
stds_e1 = []
for i in range(len(e1_data)):
    e1_rates.append(plot_hist(e1_data[i], "E1 Gaussian and Poisson fits, Run " + str(i+1)))
    stds_e1.append(np.std(e1_data[i]))
e1_avgrate = np.mean(e1_rates)

#fitting summed least square of Gaussian and Poissonian distributions for summed data set of E1
E1_sumrate = plot_hist(summed_e1, "E1 horizontally summed data Gaussian and Poissonian distribution fits")

#===============================E2 analysis===============================================
e2_data = extractData(e2)
summed_e2 = horiz_add(e2_data)

E2_rate = plot_hist(e2_data.flatten(),"E2 Gaussian and Poisson fits" )

#fitting least square of Gaussian and Poissonian distributions for all three runs of E1
#obtaining their respective counting rate mu and then calculating the avg counting rate
e2_rates = []
stds_e2=[]
for i in range(len(e2_data)):
    e2_rates.append(plot_hist(e2_data[i], "E2 Gaussian and Poisson fits, Run " + str(i+1)))
    stds_e2.append(np.std(e2_data[i]))
e2_avgrate = np.mean(e2_rates)

#fitting summed least square of Gaussian and Poissonian distributions for summed data set of E1
E2_sumrate = plot_hist(summed_e2, "E2 horizontally summed data Gaussian and Poissonian distribution fits")

#===============================E3 analysis===============================================
e3_data = extractData(e3)
horizsummed_e3 = horiz_add(e3_data)
verticalsummed_e3 = vertical_add(e3_data)

#fitting least square of Gaussian and Poissonian distributions for all three runs of E1
#obtaining their respective counting rate mu and then calculating the avg counting rate


#for i in range(len(e3_data)):
#    e3_rates.append(plot_hist(e3_data[i], "E3 Gaussian and Poisson fits, Run " + str(i+1)))
#e3_avgrate = np.mean(e3_rates)

E3_rate = plot_hist(e3_data.flatten(),"E3 Gaussian and Poisson fits" )

#fitting summed least square of Gaussian and Poissonian distributions for summed data set of E1
E3_horizontalsum_rate = plot_hist(horizsummed_e3, "E3 horizontally summed data Gaussian and Poissonian distribution fits")
E3_verticalsum_rate = plot_hist(verticalsummed_e3, "E3 vertically summed data Gaussian and Poissonian distribution fits")


#===============================E4 analysis===============================================
e4_data = extractData(e4)
summed_e4 = horiz_add(e4_data)

#E4_rate = plot_hist(e2_data.flatten(),"E2 General Gaussian and Poisson fits" )

#fitting least square of Gaussian and Poissonian distributions for all three runs of E1
#obtaining their respective counting rate mu and then calculating the avg counting rate
e4_rates = []
stds_e4=[]
for i in range(len(e4_data)):
    e4_rates.append(plot_hist(e4_data[i], "E4 Gaussian and Poisson fits, Run " + str(i+1)))
    stds_e4.append(np.std(e4_data[i]))
e4_avgrate = np.mean(e4_rates)

#fitting summed least square of Gaussian and Poissonian distributions for summed data set of E1
E4_sumrate = plot_hist(summed_e4, "E4 horizontally summed data Gaussian and Poissonian distribution fits")


#===============================tech analysis===============================================
file_data = pd.read_csv(tech)

tech_data = np.array(file_data["Run 1"])



#Plot of technician data
tech_rate = plot_hist(tech_data, "Technician data Gaussian and Poissonian fits")

#Re-binning data by two, adding two consecutive data
bin_by2 = [sum(tech_data[x:x+2]) for x in range (0,len(tech_data),2)]
tech_rate_2 = plot_hist(bin_by2, "Technician data Gaussian and Poissonian fits, binned by two")

#Re-binning data by 5, adding five consecutive data points
bin_by5 = [sum(tech_data[x:x+5]) for x in range (0,len(tech_data),5)]
tech_rate_5 = plot_hist(bin_by5, "Technician data Gaussian and Poissonian fits, binned by five")

#Re-binning data by 20, adding 20 consecutive data points
bin_by20 = [sum(tech_data[x:x+20]) for x in range (0,len(tech_data),20)]
tech_rate_20 = plot_hist(bin_by20, "Technician data Gaussian and Poissonian fits, binned by 20")

    