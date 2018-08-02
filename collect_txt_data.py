from astropy.io import fits
import numpy as np
import glob
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from astropy.table import Table, Column 
from matplotlib.font_manager import FontProperties

'''
Author:
Myles McKay

About:
This script plots the internal flat field median count-rate data from output text file from med_countrate.py.

Requirements:
1. Must run med_countrate.py first

Parameters:
--path:
    The path to directory with internal flat files
    required

Output:
1. Plot of all the filters median count-rate ratios and a function of time(Date-obs) for chip1
2. Suplot of all the filters median count-rate ratio as a function of time sperated by Filter type(W, LP, M, N)
3. Plot of all the filters median count-rate ratios and a function of time(Date-obs) for chip2
4. Suplot of all the filters median count-rate ratio as a function of time sperated by Filter type(W, LP, M, N)

Terminal output:
1. Number of files for each type of filter for chip1 and chip2

Calling Example: python collect_txt_data --path='/grp/hst/wfc3v/mmckay/internal_flats/calibrated_files/'

'''

def collect_txt_data(txt):
    date_list=[]
    chip1_mcr_ratio = []
    chip2_mcr_ratio = []
    filter_list = []
    #Reading in text file data to plot
    with open(txt) as f:
        content= f.readlines(1)
        content= f.readlines(2)
        #print(content)
        date=content[0].split()[1]
        #print(date)
        filenames = np.loadtxt(txt,usecols=(0),dtype=str)
        date_obs = np.loadtxt(txt,usecols=(1),dtype=str)
        Filter = np.loadtxt(txt,usecols=(2),dtype=str)

        chip1_mcr_eps = np.loadtxt(txt,usecols=(3),dtype=float)
        chip2_mcr_eps = np.loadtxt(txt,usecols=(4),dtype=float)

        date_list=np.append(date_list,date_obs)
        
        chip1_mcr_ratio=np.append(chip1_mcr_ratio, chip1_mcr_eps)
        chip2_mcr_ratio=np.append(chip2_mcr_ratio, chip2_mcr_eps)
        filter_list=np.append(filter_list,Filter)    
        
    return chip1_mcr_ratio, chip2_mcr_ratio, date_list, Filter


def parse_args():
    """Parses command line arguments.

    Parameters:
        nothing

    Returns:
        args : argparse.Namespace object
            An argparse object containing all of the added arguments.

    Outputs:
        nothing
    """

    #Create help string:
    path_help = 'Path to the folder with files to run tweakreg.'
    # Add arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-path', dest = 'path', action = 'store',
                        type = str, required = True, help = path_help)


    # Parse args:
    args = parser.parse_args()

    return args
# -------------------------------------------------------------------
if __name__ == '__main__':
    args = parse_args()
    path=args.path
    os.chdir(path)

    #Collecting data from each txt file
    
    #---------------------------
    #LP filters
    #---------------------------
    txt = 'F200LP_medratio_eps.txt'
    F200LP_cp1_mcr_data, F200LP_cp2_mcr_data, F200LP_date_list, F200LP_Filter = collect_txt_data(txt)
    F200LP_file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in F200LP_date_list]
    
    txt = 'F350LP_medratio_eps.txt'
    F350LP_cp1_mcr_data, F350LP_cp2_mcr_data, F350LP_date_list, F350LP_Filter = collect_txt_data(txt)
    F350LP_file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in F350LP_date_list]

    txt = 'F850LP_medratio_eps.txt'
    F850LP_cp1_mcr_data, F850LP_cp2_mcr_data, F850LP_date_list, F850LP_Filter = collect_txt_data(txt)
    F850LP_file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in F850LP_date_list]

    txt = 'F600LP_medratio_eps.txt'
    F600LP_cp1_mcr_data, F600LP_cp2_mcr_data, F600LP_date_list, F600LP_Filter = collect_txt_data(txt)
    F600LP_file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in F600LP_date_list]


    #---------------------------
    #Wide band filters
    #---------------------------

    txt = 'F336W_medratio_eps.txt'
    F336W_cp1_mcr_data, F336W_cp2_mcr_data, F336W_date_list, F336W_Filter = collect_txt_data(txt)
    F336W_file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in F336W_date_list]

    txt = 'F390W_medratio_eps.txt'
    F390W_cp1_mcr_data, F390W_cp2_mcr_data, F390W_date_list, F390W_Filter = collect_txt_data(txt)
    F390W_file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in F390W_date_list]

    txt = 'F438W_medratio_eps.txt'
    F438W_cp1_mcr_data, F438W_cp2_mcr_data, F438W_date_list, F438W_Filter = collect_txt_data(txt)
    F438W_file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in F438W_date_list]

    txt = 'F606W_medratio_eps.txt'
    F606W_cp1_mcr_data, F606W_cp2_mcr_data, F606W_date_list, F606W_Filter = collect_txt_data(txt)
    F606W_file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in F606W_date_list]

    txt = 'F475W_medratio_eps.txt'
    F475W_cp1_mcr_data, F475W_cp2_mcr_data, F475W_date_list, F475W_Filter = collect_txt_data(txt)
    F475W_file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in F475W_date_list]
    
    txt = 'F555W_medratio_eps.txt'
    F555W_cp1_mcr_data, F555W_cp2_mcr_data, F555W_date_list, F555W_Filter = collect_txt_data(txt)
    F555W_file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in F555W_date_list]

    txt = 'F625W_medratio_eps.txt'
    F625W_cp1_mcr_data, F625W_cp2_mcr_data, F625W_date_list, F625W_Filter = collect_txt_data(txt)
    F625W_file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in F625W_date_list]

    txt = 'F775W_medratio_eps.txt'
    F775W_cp1_mcr_data, F775W_cp2_mcr_data, F775W_date_list, F775W_Filter = collect_txt_data(txt)
    F775W_file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in F775W_date_list]

    txt = 'F814W_medratio_eps.txt'
    F814W_cp1_mcr_data, F814W_cp2_mcr_data, F814W_date_list, F814W_Filter = collect_txt_data(txt)
    F814W_file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in F814W_date_list]

    #---------------------------
    #Medium band filters
    #---------------------------
    txt = 'F390M_medratio_eps.txt'
    F390M_cp1_mcr_data, F390M_cp2_mcr_data, F390M_date_list, F390M_Filter = collect_txt_data(txt)
    F390M_file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in F390M_date_list]

    txt = 'F467M_medratio_eps.txt'
    F467M_cp1_mcr_data, F467M_cp2_mcr_data, F467M_date_list, F467M_Filter = collect_txt_data(txt)
    F467M_file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in F467M_date_list]

    txt = 'F547M_medratio_eps.txt'
    F547M_cp1_mcr_data, F547M_cp2_mcr_data, F547M_date_list, F547M_Filter = collect_txt_data(txt)
    F547M_file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in F547M_date_list]

    txt = 'F621M_medratio_eps.txt'
    F621M_cp1_mcr_data, F621M_cp2_mcr_data, F621M_date_list, F621M_Filter = collect_txt_data(txt)
    F621M_file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in F621M_date_list]

    txt = 'F689M_medratio_eps.txt'
    F689M_cp1_mcr_data, F689M_cp2_mcr_data, F689M_date_list, F689M_Filter = collect_txt_data(txt)
    F689M_file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in F689M_date_list]

    txt = 'F763M_medratio_eps.txt'
    F763M_cp1_mcr_data, F763M_cp2_mcr_data, F763M_date_list, F763M_Filter = collect_txt_data(txt)
    F763M_file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in F763M_date_list]

    txt = 'F845M_medratio_eps.txt'
    F845M_cp1_mcr_data, F845M_cp2_mcr_data, F845M_date_list, F845_Filter = collect_txt_data(txt)
    F845M_file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in F845M_date_list]

    txt = 'F410M_medratio_eps.txt'
    F410M_cp1_mcr_data, F410M_cp2_mcr_data, F410M_date_list, F410M_Filter = collect_txt_data(txt)
    F410M_file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in F410M_date_list]
    
    #---------------------------
    #Narrow band filters
    #---------------------------
    txt = 'F343N_medratio_eps.txt'
    F343N_cp1_mcr_data, F343N_cp2_mcr_data, F343N_date_list, F343N_Filter = collect_txt_data(txt)
    F343N_file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in F343N_date_list]

    txt = 'F469N_medratio_eps.txt'
    F469N_cp1_mcr_data, F469N_cp2_mcr_data, F469N_date_list, F469N_Filter = collect_txt_data(txt)
    F469N_file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in F469N_date_list]

    txt = 'F487N_medratio_eps.txt'
    F487N_cp1_mcr_data, F487N_cp2_mcr_data, F487N_date_list, F487N_Filter = collect_txt_data(txt)
    F487N_file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in F487N_date_list]

    txt = 'F502N_medratio_eps.txt'
    F502N_cp1_mcr_data, F502N_cp2_mcr_data, F502N_date_list, F502N_Filter = collect_txt_data(txt)
    F502N_file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in F502N_date_list]

    txt = 'F631N_medratio_eps.txt'
    F631N_cp1_mcr_data, F631N_cp2_mcr_data, F631N_date_list, F631N_Filter = collect_txt_data(txt)
    F631N_file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in F631N_date_list]

    txt = 'F645N_medratio_eps.txt'
    F645N_cp1_mcr_data, F645N_cp2_mcr_data, F645N_date_list, F645N_Filter = collect_txt_data(txt)
    F645N_file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in F645N_date_list]

    txt = 'F656N_medratio_eps.txt'
    F656N_cp1_mcr_data, F656N_cp2_mcr_data, F656N_date_list, F656N_Filter = collect_txt_data(txt)
    F656N_file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in F656N_date_list]

    txt = 'F657N_medratio_eps.txt'
    F657N_cp1_mcr_data, F657N_cp2_mcr_data, F657N_date_list, F657N_Filter = collect_txt_data(txt)
    F657N_file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in F657N_date_list]

    txt = 'F658N_medratio_eps.txt'
    F658N_cp1_mcr_data, F658N_cp2_mcr_data, F658N_date_list, F658N_Filter = collect_txt_data(txt)
    F658N_file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in F658N_date_list]

    txt = 'F665N_medratio_eps.txt'
    F665N_cp1_mcr_data, F665N_cp2_mcr_data, F665N_date_list, F665N_Filter = collect_txt_data(txt)
    F665N_file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in F665N_date_list]

    txt = 'F673N_medratio_eps.txt'
    F673N_cp1_mcr_data, F673N_cp2_mcr_data, F673N_date_list, F673N_Filter = collect_txt_data(txt)
    F673N_file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in F673N_date_list]

    txt = 'F680N_medratio_eps.txt'
    F680N_cp1_mcr_data, F680N_cp2_mcr_data, F680N_date_list, F680N_Filter = collect_txt_data(txt)
    F680N_file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in F680N_date_list]

    txt = 'F953N_medratio_eps.txt'
    F953N_cp1_mcr_data, F953N_cp2_mcr_data, F953N_date_list, F953N_Filter = collect_txt_data(txt)
    F953N_file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in F953N_date_list]

    #---------------------------
    #X band filters
    #---------------------------

    txt = 'F475X_medratio_eps.txt'
    F475X_cp1_mcr_data, F475X_cp2_mcr_data, F475X_date_list, F475X_Filter = collect_txt_data(txt)
    F475X_file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in F475X_date_list]

    #---------------------------
    #Quad filters
    #---------------------------

    txt = 'FQ508N_medratio_eps.txt'
    FQ508N_cp1_mcr_data, FQ508N_cp2_mcr_data, FQ508N_date_list, FQ508N_Filter = collect_txt_data(txt)
    FQ508N_file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in FQ508N_date_list]

    txt = 'FQ619N_medratio_eps.txt'
    FQ619N_cp1_mcr_data, FQ619N_cp2_mcr_data, FQ619N_date_list, FQ619N_Filter = collect_txt_data(txt)
    FQ619N_file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in FQ619N_date_list]

    txt = 'FQ889N_medratio_eps.txt'
    FQ889N_cp1_mcr_data, FQ889N_cp2_mcr_data, FQ889N_date_list, FQ889N_Filter = collect_txt_data(txt)
    FQ889N_file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in FQ889N_date_list]


    print('--------------------------------------')
    print('Chip1')
    print('--------------------------------------')

#chip1 data and date for linear fit
    all_c1_mcr_data = []
    lp_c1_mcr_data  = []
    wide_c1_mcr_data = []
    med_c1_mcr_data = []
    narrow_c1_mcr_data = []
    quad_c1_mcr_data = []
    x_c1_mcr_data = []

    #LP filters
    lp_c1_mcr_data = np.append(lp_c1_mcr_data, F200LP_cp1_mcr_data)
    lp_c1_mcr_data = np.append(lp_c1_mcr_data, F350LP_cp1_mcr_data)
    lp_c1_mcr_data = np.append(lp_c1_mcr_data, F850LP_cp1_mcr_data)
    lp_c1_mcr_data = np.append(lp_c1_mcr_data, F600LP_cp1_mcr_data)

    print('# of LP files',len(lp_c1_mcr_data))

    #Wide filters
    wide_c1_mcr_data = np.append(wide_c1_mcr_data, F336W_cp1_mcr_data) 
    wide_c1_mcr_data = np.append(wide_c1_mcr_data, F390W_cp1_mcr_data)
    wide_c1_mcr_data = np.append(wide_c1_mcr_data, F438W_cp1_mcr_data)
    wide_c1_mcr_data = np.append(wide_c1_mcr_data, F606W_cp1_mcr_data)
    wide_c1_mcr_data = np.append(wide_c1_mcr_data, F475W_cp1_mcr_data)
    wide_c1_mcr_data = np.append(wide_c1_mcr_data, F555W_cp1_mcr_data)
    wide_c1_mcr_data = np.append(wide_c1_mcr_data, F625W_cp1_mcr_data)
    wide_c1_mcr_data = np.append(wide_c1_mcr_data, F775W_cp1_mcr_data)
    wide_c1_mcr_data = np.append(wide_c1_mcr_data, F814W_cp1_mcr_data)

    print('# of wide files',len(wide_c1_mcr_data))

    #Medium filters
    med_c1_mcr_data = np.append(med_c1_mcr_data, F390M_cp1_mcr_data) 
    med_c1_mcr_data = np.append(med_c1_mcr_data, F410M_cp1_mcr_data)
    med_c1_mcr_data = np.append(med_c1_mcr_data, F467M_cp1_mcr_data)
    med_c1_mcr_data = np.append(med_c1_mcr_data, F547M_cp1_mcr_data)
    med_c1_mcr_data = np.append(med_c1_mcr_data, F621M_cp1_mcr_data)
    med_c1_mcr_data = np.append(med_c1_mcr_data, F689M_cp1_mcr_data)
    med_c1_mcr_data = np.append(med_c1_mcr_data, F763M_cp1_mcr_data)
    med_c1_mcr_data = np.append(med_c1_mcr_data, F845M_cp1_mcr_data)

    print('# of medium' ,len(med_c1_mcr_data))

    #Narrow filters
    narrow_c1_mcr_data = np.append(narrow_c1_mcr_data, F343N_cp1_mcr_data)     
    narrow_c1_mcr_data = np.append(narrow_c1_mcr_data, F469N_cp1_mcr_data)
    narrow_c1_mcr_data = np.append(narrow_c1_mcr_data, F487N_cp1_mcr_data)
    narrow_c1_mcr_data = np.append(narrow_c1_mcr_data, F502N_cp1_mcr_data)
    narrow_c1_mcr_data = np.append(narrow_c1_mcr_data, F631N_cp1_mcr_data)
    narrow_c1_mcr_data = np.append(narrow_c1_mcr_data, F645N_cp1_mcr_data)
    narrow_c1_mcr_data = np.append(narrow_c1_mcr_data, F656N_cp1_mcr_data)
    narrow_c1_mcr_data = np.append(narrow_c1_mcr_data, F657N_cp1_mcr_data)
    narrow_c1_mcr_data = np.append(narrow_c1_mcr_data, F658N_cp1_mcr_data)
    narrow_c1_mcr_data = np.append(narrow_c1_mcr_data, F665N_cp1_mcr_data)
    narrow_c1_mcr_data = np.append(narrow_c1_mcr_data, F673N_cp1_mcr_data)
    narrow_c1_mcr_data = np.append(narrow_c1_mcr_data, F680N_cp1_mcr_data)
    narrow_c1_mcr_data = np.append(narrow_c1_mcr_data, F953N_cp1_mcr_data)

    print('# of narrow files',len(narrow_c1_mcr_data))

    #Quad filters
    quad_c1_mcr_data = np.append(quad_c1_mcr_data, FQ508N_cp1_mcr_data) 
    quad_c1_mcr_data = np.append(quad_c1_mcr_data, FQ619N_cp1_mcr_data)
    quad_c1_mcr_data = np.append(quad_c1_mcr_data, FQ889N_cp1_mcr_data)

    print('# of quad',len(quad_c1_mcr_data))

    #X filters
    x_c1_mcr_data = np.append(x_c1_mcr_data, F475X_cp1_mcr_data)

    print('# of x ',len(x_c1_mcr_data))

    #all_filter_data_c1_data
    all_c1_mcr_data = np.append(all_c1_mcr_data, lp_c1_mcr_data)
    all_c1_mcr_data = np.append(all_c1_mcr_data, wide_c1_mcr_data)
    all_c1_mcr_data = np.append(all_c1_mcr_data, med_c1_mcr_data)
    all_c1_mcr_data = np.append(all_c1_mcr_data, narrow_c1_mcr_data)
    all_c1_mcr_data = np.append(all_c1_mcr_data, x_c1_mcr_data)
    all_c1_mcr_data = np.append(all_c1_mcr_data, quad_c1_mcr_data)

    print('# of all files',len(all_c1_mcr_data))


    print('--------------------------------------')
    print('Chip2')
    print('--------------------------------------')


#chip2 data and date for linear fit
    all_c2_mcr_data = []
    lp_c2_mcr_data  = []
    wide_c2_mcr_data = []
    med_c2_mcr_data = []
    narrow_c2_mcr_data = []
    quad_c2_mcr_data = []
    x_c2_mcr_data = []

    #LP filters
    lp_c2_mcr_data = np.append(lp_c2_mcr_data, F200LP_cp2_mcr_data)
    lp_c2_mcr_data = np.append(lp_c2_mcr_data, F350LP_cp2_mcr_data)
    lp_c2_mcr_data = np.append(lp_c2_mcr_data, F850LP_cp2_mcr_data)
    lp_c2_mcr_data = np.append(lp_c2_mcr_data, F600LP_cp2_mcr_data)

    print('# of LP files',len(lp_c2_mcr_data))

    #Wide filters
    wide_c2_mcr_data = np.append(wide_c2_mcr_data, F336W_cp2_mcr_data) 
    wide_c2_mcr_data = np.append(wide_c2_mcr_data, F390W_cp2_mcr_data)
    wide_c2_mcr_data = np.append(wide_c2_mcr_data, F438W_cp2_mcr_data)
    wide_c2_mcr_data = np.append(wide_c2_mcr_data, F606W_cp2_mcr_data)
    wide_c2_mcr_data = np.append(wide_c2_mcr_data, F475W_cp2_mcr_data)
    wide_c2_mcr_data = np.append(wide_c2_mcr_data, F555W_cp2_mcr_data)
    wide_c2_mcr_data = np.append(wide_c2_mcr_data, F625W_cp2_mcr_data)
    wide_c2_mcr_data = np.append(wide_c2_mcr_data, F775W_cp2_mcr_data)
    wide_c2_mcr_data = np.append(wide_c2_mcr_data, F814W_cp2_mcr_data)

    print('# of wide files',len(wide_c2_mcr_data))

    #Medium filters
    med_c2_mcr_data = np.append(med_c2_mcr_data, F390M_cp2_mcr_data) 
    med_c2_mcr_data = np.append(med_c2_mcr_data, F410M_cp2_mcr_data)
    med_c2_mcr_data = np.append(med_c2_mcr_data, F467M_cp2_mcr_data)
    med_c2_mcr_data = np.append(med_c2_mcr_data, F547M_cp2_mcr_data)
    med_c2_mcr_data = np.append(med_c2_mcr_data, F621M_cp2_mcr_data)
    med_c2_mcr_data = np.append(med_c2_mcr_data, F689M_cp2_mcr_data)
    med_c2_mcr_data = np.append(med_c2_mcr_data, F763M_cp2_mcr_data)
    med_c2_mcr_data = np.append(med_c2_mcr_data, F845M_cp2_mcr_data)

    print('# of medium' ,len(med_c2_mcr_data))

    #Narrow filters
    narrow_c2_mcr_data = np.append(narrow_c2_mcr_data, F343N_cp2_mcr_data)     
    narrow_c2_mcr_data = np.append(narrow_c2_mcr_data, F469N_cp2_mcr_data)
    narrow_c2_mcr_data = np.append(narrow_c2_mcr_data, F487N_cp2_mcr_data)
    narrow_c2_mcr_data = np.append(narrow_c2_mcr_data, F502N_cp2_mcr_data)
    narrow_c2_mcr_data = np.append(narrow_c2_mcr_data, F631N_cp2_mcr_data)
    narrow_c2_mcr_data = np.append(narrow_c2_mcr_data, F645N_cp2_mcr_data)
    narrow_c2_mcr_data = np.append(narrow_c2_mcr_data, F656N_cp2_mcr_data)
    narrow_c2_mcr_data = np.append(narrow_c2_mcr_data, F657N_cp2_mcr_data)
    narrow_c2_mcr_data = np.append(narrow_c2_mcr_data, F658N_cp2_mcr_data)
    narrow_c2_mcr_data = np.append(narrow_c2_mcr_data, F665N_cp2_mcr_data)
    narrow_c2_mcr_data = np.append(narrow_c2_mcr_data, F673N_cp2_mcr_data)
    narrow_c2_mcr_data = np.append(narrow_c2_mcr_data, F680N_cp2_mcr_data)
    narrow_c2_mcr_data = np.append(narrow_c2_mcr_data, F953N_cp2_mcr_data)

    print('# of narrow files',len(narrow_c2_mcr_data))

    #Quad filters
    quad_c2_mcr_data = np.append(quad_c2_mcr_data, FQ508N_cp2_mcr_data) 
    quad_c2_mcr_data = np.append(quad_c2_mcr_data, FQ619N_cp2_mcr_data)
    quad_c2_mcr_data = np.append(quad_c2_mcr_data, FQ889N_cp2_mcr_data)

    print('# of quad',len(quad_c2_mcr_data))

    #X filters
    x_c2_mcr_data = np.append(x_c2_mcr_data, F475X_cp2_mcr_data)

    print('# of x ',len(x_c2_mcr_data))

    #all_filter_data_c1_data
    all_c2_mcr_data = np.append(all_c2_mcr_data, lp_c2_mcr_data)
    all_c2_mcr_data = np.append(all_c2_mcr_data, wide_c2_mcr_data)
    all_c2_mcr_data = np.append(all_c2_mcr_data, med_c2_mcr_data)
    all_c2_mcr_data = np.append(all_c2_mcr_data, narrow_c2_mcr_data)
    all_c2_mcr_data = np.append(all_c2_mcr_data, x_c2_mcr_data)
    all_c2_mcr_data = np.append(all_c2_mcr_data, quad_c2_mcr_data)

    print('# of all files',len(all_c2_mcr_data))
#--------- Date list ------------------
    all_dates = []
    lp_dates = []
    wide_dates = []
    med_dates = []
    narrow_dates = []
    x_dates = []
    quad_dates = []



    lp_dates = np.append(lp_dates, F200LP_date_list) #LP filters
    lp_dates = np.append(lp_dates, F350LP_date_list)
    lp_dates = np.append(lp_dates, F850LP_date_list)
    lp_dates = np.append(lp_dates, F600LP_date_list)

    wide_dates = np.append(wide_dates, F336W_date_list) #Wide filters
    wide_dates = np.append(wide_dates, F390W_date_list)
    wide_dates = np.append(wide_dates, F438W_date_list)
    wide_dates = np.append(wide_dates, F606W_date_list)
    wide_dates = np.append(wide_dates, F475W_date_list)
    wide_dates = np.append(wide_dates, F555W_date_list)
    wide_dates = np.append(wide_dates, F625W_date_list)
    wide_dates = np.append(wide_dates, F775W_date_list)
    wide_dates = np.append(wide_dates, F814W_date_list)

    med_dates = np.append(med_dates, F390M_date_list) #Medium filters
    med_dates = np.append(med_dates, F410M_date_list)
    med_dates = np.append(med_dates, F467M_date_list)
    med_dates = np.append(med_dates, F547M_date_list)
    med_dates = np.append(med_dates, F621M_date_list)
    med_dates = np.append(med_dates, F689M_date_list)
    med_dates = np.append(med_dates, F763M_date_list)
    med_dates = np.append(med_dates, F845M_date_list)

    narrow_dates = np.append(narrow_dates, F343N_date_list) #Narrow filters    
    narrow_dates = np.append(narrow_dates, F469N_date_list)
    narrow_dates = np.append(narrow_dates, F487N_date_list)
    narrow_dates = np.append(narrow_dates, F502N_date_list)
    narrow_dates = np.append(narrow_dates, F631N_date_list)
    narrow_dates = np.append(narrow_dates, F645N_date_list)
    narrow_dates = np.append(narrow_dates, F656N_date_list)
    narrow_dates = np.append(narrow_dates, F657N_date_list)
    narrow_dates = np.append(narrow_dates, F658N_date_list)
    narrow_dates = np.append(narrow_dates, F665N_date_list)
    narrow_dates = np.append(narrow_dates, F673N_date_list)
    narrow_dates = np.append(narrow_dates, F680N_date_list)
    narrow_dates = np.append(narrow_dates, F953N_date_list)

    quad_dates = np.append(quad_dates, FQ508N_date_list) #Quad filters
    quad_dates = np.append(quad_dates, FQ619N_date_list)
    quad_dates = np.append(quad_dates, FQ889N_date_list)

    x_dates = np.append(x_dates, F475X_date_list) #X filters

    all_dates = np.append(all_dates, lp_dates)
    all_dates = np.append(all_dates, wide_dates)
    all_dates = np.append(all_dates, med_dates)
    all_dates = np.append(all_dates, narrow_dates)
    all_dates = np.append(all_dates, x_dates)
    all_dates = np.append(all_dates, quad_dates)


#----------------------Chip 1 plot ---------------------------
    plt.clf()
    fig, [(ax0, ax1), (ax2, ax3), (ax4, ax5)] = plt.subplots(3, 2, figsize=(16, 8),sharex='col',sharey='row')
    #legend position
    x0, y0 = 0.98, 1.025


    #LP filters
    lp_dates = [pd.to_datetime(d,format='%Y-%m-%d') for d in lp_dates]
    x = mdates.date2num(lp_dates)
    polyfit1=np.polyfit(x, lp_c1_mcr_data, 1)
    polyfit1_data=((polyfit1[0]*x + polyfit1[1]))

    ax0.scatter(F200LP_file_date, F200LP_cp1_mcr_data, marker='o', color='red',   label='F200LP')
    ax0.scatter(F350LP_file_date, F350LP_cp1_mcr_data, marker='o', color='blue',  label='F350LP')
    ax0.scatter(F850LP_file_date, F850LP_cp1_mcr_data, marker='o', color='green', label='F850LP')
    ax0.scatter(F600LP_file_date, F600LP_cp1_mcr_data, marker='o', color='orange',label='F600LP')
    fit = ax0.plot(x, polyfit1_data, color='red')
    ax0.set_ylabel('Ratio')
    ax0.grid()
    ax0.legend(loc='best', bbox_to_anchor=(x0, y0), ncol=1, borderaxespad=0, frameon=False)

    #Wide band filter
    wide_dates = [pd.to_datetime(d,format='%Y-%m-%d') for d in wide_dates]
    x = mdates.date2num(wide_dates)
    polyfit1=np.polyfit(x, wide_c1_mcr_data, 1)
    polyfit1_data=((polyfit1[0]*x + polyfit1[1]))

    ax1.scatter(F336W_file_date, F336W_cp1_mcr_data, marker='+', color='red',    label='F336W')
    ax1.scatter(F390W_file_date, F390W_cp1_mcr_data, marker='+', color='blue',   label='F390W')
    ax1.scatter(F438W_file_date, F438W_cp1_mcr_data, marker='+', color='green',  label='F438W')
    ax1.scatter(F606W_file_date, F606W_cp1_mcr_data, marker='+', color='orange', label='F606W')
    ax1.scatter(F475W_file_date, F475W_cp1_mcr_data, marker='+', color='purple', label='F475W')
    ax1.scatter(F555W_file_date, F555W_cp1_mcr_data, marker='+', color='black',  label='F555W')
    ax1.scatter(F625W_file_date, F625W_cp1_mcr_data, marker='+', color='pink',   label='F625W')
    ax1.scatter(F775W_file_date, F775W_cp1_mcr_data, marker='+', color='grey',   label='F775W')
    ax1.scatter(F814W_file_date, F814W_cp1_mcr_data, marker='+', color='skyblue',label='F814W')
    ax1.plot(x, polyfit1_data, color='red')    
    ax1.grid()
    ax1.legend(loc='best', bbox_to_anchor=(x0, y0), ncol=1, borderaxespad=0, frameon=False)

    #Medium band filters
    med_dates = [pd.to_datetime(d,format='%Y-%m-%d') for d in med_dates]
    x = mdates.date2num(med_dates)
    polyfit1=np.polyfit(x, med_c1_mcr_data, 1)
    polyfit1_data=((polyfit1[0]*x + polyfit1[1]))

    ax2.scatter(F390M_file_date, F390M_cp1_mcr_data, marker='h', color='red',    label='F390M')
    ax2.scatter(F410M_file_date, F410M_cp1_mcr_data, marker='h', color='blue',   label='F410M')
    ax2.scatter(F467M_file_date, F467M_cp1_mcr_data, marker='h', color='green',  label='F467M')
    ax2.scatter(F547M_file_date, F547M_cp1_mcr_data, marker='h', color='orange', label='F547M')
    ax2.scatter(F621M_file_date, F621M_cp1_mcr_data, marker='h', color='purple', label='F621M')
    ax2.scatter(F689M_file_date, F689M_cp1_mcr_data, marker='h', color='black',  label='F689M')
    ax2.scatter(F763M_file_date, F763M_cp1_mcr_data, marker='h', color='pink',   label='F763M')
    ax2.scatter(F845M_file_date, F845M_cp1_mcr_data, marker='h', color='grey',   label='F845M')
    ax2.plot(x, polyfit1_data, color='red')
    ax2.grid()
    ax2.set_ylabel('Ratio')
    ax2.legend(loc='best', bbox_to_anchor=(x0, y0), ncol=1, borderaxespad=0, frameon=False)
    
    #Narrow band filter
    narrow_dates = [pd.to_datetime(d,format='%Y-%m-%d') for d in narrow_dates]
    x = mdates.date2num(narrow_dates)
    polyfit1=np.polyfit(x, narrow_c1_mcr_data, 1)
    polyfit1_data=((polyfit1[0]*x + polyfit1[1]))

    #plt.scatter(F343N_file_date, F343N_cp1_mcr_data, marker='*', color='red',    label='F343N')
    ax3.scatter(F469N_file_date, F469N_cp1_mcr_data, marker='*', color='blue',    label='F469N')
    ax3.scatter(F487N_file_date, F487N_cp1_mcr_data, marker='*', color='green',   label='F487N')
    ax3.scatter(F502N_file_date, F502N_cp1_mcr_data, marker='*', color='orange',  label='F502N')
    ax3.scatter(F631N_file_date, F631N_cp1_mcr_data, marker='*', color='purple',  label='F631N')
    ax3.scatter(F645N_file_date, F645N_cp1_mcr_data, marker='*', color='black',   label='F645N')
    ax3.scatter(F656N_file_date, F656N_cp1_mcr_data, marker='*', color='pink',    label='F656N')
    ax3.scatter(F657N_file_date, F657N_cp1_mcr_data, marker='*', color='grey',    label='F657N')
    ax3.scatter(F658N_file_date, F658N_cp1_mcr_data, marker='*', color='yellow',  label='F658N')
    ax3.scatter(F665N_file_date, F665N_cp1_mcr_data, marker='*', color='gold',    label='F665N')
    ax3.scatter(F673N_file_date, F673N_cp1_mcr_data, marker='*', color='navy',    label='F673N')
    ax3.scatter(F680N_file_date, F680N_cp1_mcr_data, marker='*', color='deeppink',label='F680N')
    ax3.scatter(F953N_file_date, F953N_cp1_mcr_data, marker='*', color='skyblue', label='F953N')
    ax3.plot(x, polyfit1_data, color='red')
    ax3.grid()
    ax3.legend(loc='best', bbox_to_anchor=(x0, y0), ncol=1, borderaxespad=0, frameon=False)


    #Quad filter
    quad_dates = [pd.to_datetime(d,format='%Y-%m-%d') for d in quad_dates]
    x = mdates.date2num(quad_dates)
    polyfit1=np.polyfit(x, quad_c1_mcr_data, 1)
    polyfit1_data=((polyfit1[0]*x + polyfit1[1]))

    ax4.scatter(FQ508N_file_date, FQ508N_cp1_mcr_data, marker='D', color='red',  label='FQ508N')
    ax4.scatter(FQ619N_file_date, FQ619N_cp1_mcr_data, marker='D', color='blue', label='FQ619N')
    ax4.scatter(FQ889N_file_date, FQ889N_cp1_mcr_data, marker='D', color='green',label='FQ889N')
    ax4.plot(x, polyfit1_data, color='red')
    ax4.grid()
    ax4.set_ylabel('Ratio')
    ax4.set_xlabel('Date of Oberservation')
    ax4.legend(loc='best', bbox_to_anchor=(x0, y0), ncol=1, borderaxespad=0, frameon=False)
    
    #X filter
    x_dates = [pd.to_datetime(d,format='%Y-%m-%d') for d in x_dates]
    x = mdates.date2num(x_dates)
    polyfit1=np.polyfit(x, x_c1_mcr_data, 1)
    polyfit1_data=((polyfit1[0]*x + polyfit1[1]))

    ax5.scatter(F475X_file_date, F475X_cp1_mcr_data, marker='^', color='red', label='F475X')
    ax5.plot(x, polyfit1_data, color='red')
    ax5.grid()
    ax5.set_xlabel('Date of Oberservation')
    ax5.legend(loc='best', bbox_to_anchor=(0.98, 1.0), ncol=1, borderaxespad=0, frameon=False)

    #plt.plot(x,polyfit1_data,'r')
    plt.savefig('c1_filter_med_countrate_subplot.png')
    #plt.show()
    #plt.clf()

#------------------------Chip1 All filters plot--------------------------------------------

    all_dates = [pd.to_datetime(d,format='%Y-%m-%d') for d in all_dates]
    x = mdates.date2num(all_dates)
    polyfit1=np.polyfit(x, all_c1_mcr_data, 1)
    polyfit1_data=((polyfit1[0]*x + polyfit1[1]))

    plt.figure(figsize=(20, 8))
    plt.scatter(F200LP_file_date, F200LP_cp1_mcr_data, marker='o', color='red',   label='F200LP')
    plt.scatter(F350LP_file_date, F350LP_cp1_mcr_data, marker='o', color='blue',  label='F350LP')
    plt.scatter(F850LP_file_date, F850LP_cp1_mcr_data, marker='o', color='green', label='F850LP')
    plt.scatter(F600LP_file_date, F600LP_cp1_mcr_data, marker='o', color='orange',label='F600LP')
   
    plt.scatter(F336W_file_date, F336W_cp1_mcr_data, marker='+', color='red',    label='F336W')
    plt.scatter(F390W_file_date, F390W_cp1_mcr_data, marker='+', color='blue',   label='F390W')
    plt.scatter(F438W_file_date, F438W_cp1_mcr_data, marker='+', color='green',  label='F438W')
    plt.scatter(F606W_file_date, F606W_cp1_mcr_data, marker='+', color='orange', label='F606W')
    plt.scatter(F475W_file_date, F475W_cp1_mcr_data, marker='+', color='purple', label='F475W')
    plt.scatter(F555W_file_date, F555W_cp1_mcr_data, marker='+', color='black',  label='F555W')
    plt.scatter(F625W_file_date, F625W_cp1_mcr_data, marker='+', color='pink',   label='F625W')
    plt.scatter(F775W_file_date, F775W_cp1_mcr_data, marker='+', color='grey',   label='F775W')
    plt.scatter(F814W_file_date, F814W_cp1_mcr_data, marker='+', color='skyblue',label='F814W')

    plt.scatter(F390M_file_date, F390M_cp1_mcr_data, marker='h', color='red',    label='F390M')
    plt.scatter(F410M_file_date, F410M_cp1_mcr_data, marker='h', color='blue',   label='F410M')
    plt.scatter(F467M_file_date, F467M_cp1_mcr_data, marker='h', color='green',  label='F467M')
    plt.scatter(F547M_file_date, F547M_cp1_mcr_data, marker='h', color='orange', label='F547M')
    plt.scatter(F621M_file_date, F621M_cp1_mcr_data, marker='h', color='purple', label='F621M')
    plt.scatter(F689M_file_date, F689M_cp1_mcr_data, marker='h', color='black',  label='F689M')
    plt.scatter(F763M_file_date, F763M_cp1_mcr_data, marker='h', color='pink',   label='F763M')
    plt.scatter(F845M_file_date, F845M_cp1_mcr_data, marker='h', color='grey',   label='F845M')
 
    plt.scatter(F469N_file_date, F469N_cp1_mcr_data, marker='*', color='blue',    label='F469N')
    plt.scatter(F487N_file_date, F487N_cp1_mcr_data, marker='*', color='green',   label='F487N')
    plt.scatter(F502N_file_date, F502N_cp1_mcr_data, marker='*', color='orange',  label='F502N')
    plt.scatter(F631N_file_date, F631N_cp1_mcr_data, marker='*', color='purple',  label='F631N')
    plt.scatter(F645N_file_date, F645N_cp1_mcr_data, marker='*', color='black',   label='F645N')
    plt.scatter(F656N_file_date, F656N_cp1_mcr_data, marker='*', color='pink',    label='F656N')
    plt.scatter(F657N_file_date, F657N_cp1_mcr_data, marker='*', color='grey',    label='F657N')
    plt.scatter(F658N_file_date, F658N_cp1_mcr_data, marker='*', color='yellow',  label='F658N')
    plt.scatter(F665N_file_date, F665N_cp1_mcr_data, marker='*', color='gold',    label='F665N')
    plt.scatter(F673N_file_date, F673N_cp1_mcr_data, marker='*', color='navy',    label='F673N')
    plt.scatter(F680N_file_date, F680N_cp1_mcr_data, marker='*', color='deeppink',label='F680N')
    plt.scatter(F953N_file_date, F953N_cp1_mcr_data, marker='*', color='skyblue', label='F953N')

    plt.scatter(FQ508N_file_date, FQ508N_cp1_mcr_data, marker='D', color='red',  label='FQ508N')
    plt.scatter(FQ619N_file_date, FQ619N_cp1_mcr_data, marker='D', color='blue', label='FQ619N')
    plt.scatter(FQ889N_file_date, FQ889N_cp1_mcr_data, marker='D', color='green',label='FQ889N')
 
    plt.scatter(F475X_file_date, F475X_cp1_mcr_data, marker='^', color='red', label='F475X')
    
    plt.plot(x,polyfit1_data,'r')

    plt.ylabel('Ratio')
    plt.xlabel('Date of Observation')
    plt.grid()
    plt.ylim()
    plt.legend(loc='best', ncol=9)
    #plt.legend(loc=2, bbox_to_anchor=(0.5, 1.025), ncol=5, borderaxespad=1, 
    #            frameon=False )
    plt.savefig('all_c1_filter_med_countrate.png')
    plt.savefig('all_c1_filter_med_countrate.pdf')
    #plt.show()
    #plt.clf()

#----------------------Chip 2 plot ---------------------------
    fig, [(ax0, ax1), (ax2, ax3), (ax4, ax5)] = plt.subplots(3, 2, figsize=(16, 8),sharex='col',sharey='row')
    #legend position
    x0, y0 = 0.98, 1.025


    #LP filters
    lp_dates = [pd.to_datetime(d,format='%Y-%m-%d') for d in lp_dates]
    x = mdates.date2num(lp_dates)
    polyfit1=np.polyfit(x, lp_c2_mcr_data, 1)
    polyfit1_data=((polyfit1[0]*x + polyfit1[1]))

    ax0.scatter(F200LP_file_date, F200LP_cp2_mcr_data, marker='o', color='red',   label='F200LP')
    ax0.scatter(F350LP_file_date, F350LP_cp2_mcr_data, marker='o', color='blue',  label='F350LP')
    ax0.scatter(F850LP_file_date, F850LP_cp2_mcr_data, marker='o', color='green', label='F850LP')
    ax0.scatter(F600LP_file_date, F600LP_cp2_mcr_data, marker='o', color='orange',label='F600LP')
    fit = ax0.plot(x, polyfit1_data, color='red')
    ax0.set_ylabel('Ratio')
    ax0.grid()
    ax0.legend(loc='best', bbox_to_anchor=(x0, y0), ncol=1, borderaxespad=0, frameon=False)

    #Wide band filter
    wide_dates = [pd.to_datetime(d,format='%Y-%m-%d') for d in wide_dates]
    x = mdates.date2num(wide_dates)
    polyfit1=np.polyfit(x, wide_c2_mcr_data, 1)
    polyfit1_data=((polyfit1[0]*x + polyfit1[1]))

    ax1.scatter(F336W_file_date, F336W_cp2_mcr_data, marker='+', color='red',    label='F336W')
    ax1.scatter(F390W_file_date, F390W_cp2_mcr_data, marker='+', color='blue',   label='F390W')
    ax1.scatter(F438W_file_date, F438W_cp2_mcr_data, marker='+', color='green',  label='F438W')
    ax1.scatter(F606W_file_date, F606W_cp2_mcr_data, marker='+', color='orange', label='F606W')
    ax1.scatter(F475W_file_date, F475W_cp2_mcr_data, marker='+', color='purple', label='F475W')
    ax1.scatter(F555W_file_date, F555W_cp2_mcr_data, marker='+', color='black',  label='F555W')
    ax1.scatter(F625W_file_date, F625W_cp2_mcr_data, marker='+', color='pink',   label='F625W')
    ax1.scatter(F775W_file_date, F775W_cp2_mcr_data, marker='+', color='grey',   label='F775W')
    ax1.scatter(F814W_file_date, F814W_cp2_mcr_data, marker='+', color='skyblue',label='F814W')
    ax1.plot(x, polyfit1_data, color='red')    
    ax1.grid()
    ax1.legend(loc='best', bbox_to_anchor=(x0, y0), ncol=1, borderaxespad=0, frameon=False)

    #Medium band filters
    med_dates = [pd.to_datetime(d,format='%Y-%m-%d') for d in med_dates]
    x = mdates.date2num(med_dates)
    polyfit1=np.polyfit(x, med_c2_mcr_data, 1)
    polyfit1_data=((polyfit1[0]*x + polyfit1[1]))

    ax2.scatter(F390M_file_date, F390M_cp2_mcr_data, marker='h', color='red',    label='F390M')
    ax2.scatter(F410M_file_date, F410M_cp2_mcr_data, marker='h', color='blue',   label='F410M')
    ax2.scatter(F467M_file_date, F467M_cp2_mcr_data, marker='h', color='green',  label='F467M')
    ax2.scatter(F547M_file_date, F547M_cp2_mcr_data, marker='h', color='orange', label='F547M')
    ax2.scatter(F621M_file_date, F621M_cp2_mcr_data, marker='h', color='purple', label='F621M')
    ax2.scatter(F689M_file_date, F689M_cp2_mcr_data, marker='h', color='black',  label='F689M')
    ax2.scatter(F763M_file_date, F763M_cp2_mcr_data, marker='h', color='pink',   label='F763M')
    ax2.scatter(F845M_file_date, F845M_cp2_mcr_data, marker='h', color='grey',   label='F845M')
    ax2.plot(x, polyfit1_data, color='red')
    ax2.grid()
    ax2.set_ylabel('Ratio')
    ax2.legend(loc='best', bbox_to_anchor=(x0, y0), ncol=1, borderaxespad=0, frameon=False)
    
    #Narrow band filter
    narrow_dates = [pd.to_datetime(d,format='%Y-%m-%d') for d in narrow_dates]
    x = mdates.date2num(narrow_dates)
    polyfit1=np.polyfit(x, narrow_c2_mcr_data, 1)
    polyfit1_data=((polyfit1[0]*x + polyfit1[1]))

    #plt.scatter(F343N_file_date, F343N_cp1_mcr_data, marker='*', color='red',    label='F343N')
    ax3.scatter(F469N_file_date, F469N_cp2_mcr_data, marker='*', color='blue',    label='F469N')
    ax3.scatter(F487N_file_date, F487N_cp2_mcr_data, marker='*', color='green',   label='F487N')
    ax3.scatter(F502N_file_date, F502N_cp2_mcr_data, marker='*', color='orange',  label='F502N')
    ax3.scatter(F631N_file_date, F631N_cp2_mcr_data, marker='*', color='purple',  label='F631N')
    ax3.scatter(F645N_file_date, F645N_cp2_mcr_data, marker='*', color='black',   label='F645N')
    ax3.scatter(F656N_file_date, F656N_cp2_mcr_data, marker='*', color='pink',    label='F656N')
    ax3.scatter(F657N_file_date, F657N_cp2_mcr_data, marker='*', color='grey',    label='F657N')
    ax3.scatter(F658N_file_date, F658N_cp2_mcr_data, marker='*', color='yellow',  label='F658N')
    ax3.scatter(F665N_file_date, F665N_cp2_mcr_data, marker='*', color='gold',    label='F665N')
    ax3.scatter(F673N_file_date, F673N_cp2_mcr_data, marker='*', color='navy',    label='F673N')
    ax3.scatter(F680N_file_date, F680N_cp2_mcr_data, marker='*', color='deeppink',label='F680N')
    ax3.scatter(F953N_file_date, F953N_cp2_mcr_data, marker='*', color='skyblue', label='F953N')
    ax3.plot(x, polyfit1_data, color='red')
    ax3.grid()
    ax3.legend(loc='best', bbox_to_anchor=(x0, y0), ncol=1, borderaxespad=0, frameon=False)


    #Quad filter
    quad_dates = [pd.to_datetime(d,format='%Y-%m-%d') for d in quad_dates]
    x = mdates.date2num(quad_dates)
    polyfit1=np.polyfit(x, quad_c2_mcr_data, 1)
    polyfit1_data=((polyfit1[0]*x + polyfit1[1]))

    ax4.scatter(FQ508N_file_date, FQ508N_cp2_mcr_data, marker='D', color='red',  label='FQ508N')
    ax4.scatter(FQ619N_file_date, FQ619N_cp2_mcr_data, marker='D', color='blue', label='FQ619N')
    ax4.scatter(FQ889N_file_date, FQ889N_cp2_mcr_data, marker='D', color='green',label='FQ889N')
    ax4.plot(x, polyfit1_data, color='red')
    ax4.grid()
    ax4.set_ylabel('Ratio')
    ax4.set_xlabel('Date of Oberservation')
    ax4.legend(loc='best', bbox_to_anchor=(x0, y0), ncol=1, borderaxespad=0, frameon=False)
    
    #X filter
    x_dates = [pd.to_datetime(d,format='%Y-%m-%d') for d in x_dates]
    x = mdates.date2num(x_dates)
    polyfit1=np.polyfit(x, x_c2_mcr_data, 1)
    polyfit1_data=((polyfit1[0]*x + polyfit1[1]))

    ax5.scatter(F475X_file_date, F475X_cp2_mcr_data, marker='^', color='red', label='F475X')
    ax5.plot(x, polyfit1_data, color='red')
    ax5.grid()
    ax5.set_xlabel('Date of Oberservation')
    ax5.legend(loc='best', bbox_to_anchor=(0.98, 1.0), ncol=1, borderaxespad=0, frameon=False)

    #plt.plot(x,polyfit1_data,'r')
    plt.savefig('c2_filter_med_countrate_subplot.png')
    #plt.show()
    #plt.clf()

#------------------------Chip2 All filters plot--------------------------------------------

    all_dates = [pd.to_datetime(d,format='%Y-%m-%d') for d in all_dates]
    x = mdates.date2num(all_dates)
    polyfit1=np.polyfit(x, all_c2_mcr_data, 1)
    polyfit1_data=((polyfit1[0]*x + polyfit1[1]))

    plt.figure(figsize=(20, 8))
    plt.scatter(F200LP_file_date, F200LP_cp2_mcr_data, marker='o', color='red',   label='F200LP')
    plt.scatter(F350LP_file_date, F350LP_cp2_mcr_data, marker='o', color='blue',  label='F350LP')
    plt.scatter(F850LP_file_date, F850LP_cp2_mcr_data, marker='o', color='green', label='F850LP')
    plt.scatter(F600LP_file_date, F600LP_cp2_mcr_data, marker='o', color='orange',label='F600LP')
   
    plt.scatter(F336W_file_date, F336W_cp2_mcr_data, marker='+', color='red',    label='F336W')
    plt.scatter(F390W_file_date, F390W_cp2_mcr_data, marker='+', color='blue',   label='F390W')
    plt.scatter(F438W_file_date, F438W_cp2_mcr_data, marker='+', color='green',  label='F438W')
    plt.scatter(F606W_file_date, F606W_cp2_mcr_data, marker='+', color='orange', label='F606W')
    plt.scatter(F475W_file_date, F475W_cp2_mcr_data, marker='+', color='purple', label='F475W')
    plt.scatter(F555W_file_date, F555W_cp2_mcr_data, marker='+', color='black',  label='F555W')
    plt.scatter(F625W_file_date, F625W_cp2_mcr_data, marker='+', color='pink',   label='F625W')
    plt.scatter(F775W_file_date, F775W_cp2_mcr_data, marker='+', color='grey',   label='F775W')
    plt.scatter(F814W_file_date, F814W_cp2_mcr_data, marker='+', color='skyblue',label='F814W')

    plt.scatter(F390M_file_date, F390M_cp2_mcr_data, marker='h', color='red',    label='F390M')
    plt.scatter(F410M_file_date, F410M_cp2_mcr_data, marker='h', color='blue',   label='F410M')
    plt.scatter(F467M_file_date, F467M_cp2_mcr_data, marker='h', color='green',  label='F467M')
    plt.scatter(F547M_file_date, F547M_cp2_mcr_data, marker='h', color='orange', label='F547M')
    plt.scatter(F621M_file_date, F621M_cp2_mcr_data, marker='h', color='purple', label='F621M')
    plt.scatter(F689M_file_date, F689M_cp2_mcr_data, marker='h', color='black',  label='F689M')
    plt.scatter(F763M_file_date, F763M_cp2_mcr_data, marker='h', color='pink',   label='F763M')
    plt.scatter(F845M_file_date, F845M_cp2_mcr_data, marker='h', color='grey',   label='F845M')
 
    plt.scatter(F469N_file_date, F469N_cp2_mcr_data, marker='*', color='blue',    label='F469N')
    plt.scatter(F487N_file_date, F487N_cp2_mcr_data, marker='*', color='green',   label='F487N')
    plt.scatter(F502N_file_date, F502N_cp2_mcr_data, marker='*', color='orange',  label='F502N')
    plt.scatter(F631N_file_date, F631N_cp2_mcr_data, marker='*', color='purple',  label='F631N')
    plt.scatter(F645N_file_date, F645N_cp2_mcr_data, marker='*', color='black',   label='F645N')
    plt.scatter(F656N_file_date, F656N_cp2_mcr_data, marker='*', color='pink',    label='F656N')
    plt.scatter(F657N_file_date, F657N_cp2_mcr_data, marker='*', color='grey',    label='F657N')
    plt.scatter(F658N_file_date, F658N_cp2_mcr_data, marker='*', color='yellow',  label='F658N')
    plt.scatter(F665N_file_date, F665N_cp2_mcr_data, marker='*', color='gold',    label='F665N')
    plt.scatter(F673N_file_date, F673N_cp2_mcr_data, marker='*', color='navy',    label='F673N')
    plt.scatter(F680N_file_date, F680N_cp2_mcr_data, marker='*', color='deeppink',label='F680N')
    plt.scatter(F953N_file_date, F953N_cp2_mcr_data, marker='*', color='skyblue', label='F953N')

    plt.scatter(FQ508N_file_date, FQ508N_cp2_mcr_data, marker='D', color='red',  label='FQ508N')
    plt.scatter(FQ619N_file_date, FQ619N_cp2_mcr_data, marker='D', color='blue', label='FQ619N')
    plt.scatter(FQ889N_file_date, FQ889N_cp2_mcr_data, marker='D', color='green',label='FQ889N')
 
    plt.scatter(F475X_file_date, F475X_cp2_mcr_data, marker='^', color='red', label='F475X')
    
    plt.plot(x,polyfit1_data,'r')

    plt.ylabel('Ratio')
    plt.xlabel('Date of Observation')
    plt.grid()
    plt.ylim()
    plt.legend(loc='best', ncol=9)
    plt.savefig('all_c2_filter_med_countrate.png')
    plt.savefig('all_c2_filter_med_countrate.pdf')
    plt.show()
    plt.clf()

#cycle 26 phase 1

#    plt.scatter(F845M_file_date, F845M_cp2_mcr_data, marker='+', color='blue',  label='F845M')
#    plt.scatter(F410M_file_date, F410M_cp2_mcr_data, marker='+', color='red',   label='F410M')
    plt.scatter(F606W_file_date, F606W_cp2_mcr_data, marker='+', color='green', label='F606W')
    #plt.scatter(F438W_file_date, F438W_cp2_mcr_data, marker='+', color='green',  label='F438W')
    #F438W_file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in F438W_file_date]
    #x = mdates.date2num(F438W_file_date)
    #polyfit1=np.polyfit(x, F438W_cp2_mcr_data, 1)
    #polyfit1_data=((polyfit1[0]*x + polyfit1[1]))
    #plt.plot(x,polyfit1_data,'r')
    plt.grid()
    plt.ylabel('Ratio')
    plt.xlabel('Date of Observation')
    plt.title('Median count rate ratio vs. Date-obs')
    plt.legend()
    plt.savefig('F438W_mcr_plot.png')
    plt.show()
    plt.clf()


