from astropy.io import fits
import numpy as np
import glob
import os
import argparse
from multiprocessing import Pool
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import skimage.morphology as morph 
from astropy.table import Table, Column       



'''
Author:
Myles McKay

About:
This script calculates the median count rate (electrons/second) ratio for calibrated WFC3/UVIS internal 
flats using the first file (Earliest observation) as the fiducial.

Requirements:
1. The internal flats files must be the same filter and lamp
2. Internal flats must be calculated
3. Files must be in the same directory

Parameters:
--path:
    The path to directory with internal flat files
    required

Output:
    .txt file with 5 columns:
    columns:
    1. Filename
    2. Date-obs(Date of observation)
    3. Filter Name
    4. Median count-rate ratio for chip1
    5. Median count-rate ratio for chip2 

Calling Example: python med_countrate.py --path='/grp/hst/wfc3v/mmckay/internal_flats/calibrated_files/tung3_F200LP/'

'''

def med_count_rate(files):


    #Reading in the first image in the list of files (The earliest observation)
    first_file=fits.open(files[0])
    first_rootname= first_file[0].header['rootname']
    first_exptime= first_file[0].header['exptime']
    first_ccdgain= first_file[0].header['ccdgain']
    first_sci_chip1 = first_file[4].data
    first_dq_chip1  = first_file[6].data
    first_sci_chip2 = first_file[1].data
    first_dq_chip2  = first_file[3].data
    
    # Calculate the median count-rate for chip and 2
    first_sci_chip1 = (first_sci_chip1 * first_ccdgain) / first_exptime
    first_sci_chip2 = (first_sci_chip2 * first_ccdgain) / first_exptime

    #Masking science data with Data Quality (DQ) data(sets all bad pixels to np.nan)
    first_sci_chip1[first_dq_chip1 !=0]=np.nan
    first_sci_chip2[first_dq_chip2 !=0]=np.nan
    

    #reading in all other images in directory including the first observation
    list_of_files = glob.glob('*flt.fits')
    file_med1=[]
    file_med2=[]
    file_date=[]
    file_date_list = []
    file_pid =[]
    file_filter = []
    file_gain = []
    filename_list = []


    for i in list_of_files:
        hdu=fits.open(i)
        date=hdu[0].header['date-obs']
        filename=hdu[0].header['filename']
        filter_name=hdu[0].header['filter']
        ccdgain = hdu[0].header['ccdgain']
        exp_time=hdu[0].header['exptime']
        rootname=hdu[0].header['rootname']
        proposal = hdu[0].header['PROPOSID']

        sci_chip1=hdu[4].data
        dq_chip1 =hdu[6].data
        sci_chip2=hdu[1].data
        dq_chip2 =hdu[3].data

        #Calculates the median count-rate for chip1 and chip2
        sci_chip1= (sci_chip1 * ccdgain) / exp_time
        sci_chip2= (sci_chip2 * ccdgain) / exp_time

        #Masking science data with Data Quality (DQ) data(sets all bad pixels to np.nan)
        sci_chip1[dq_chip1 !=0]= np.nan
        sci_chip2[dq_chip2 !=0]= np.nan
    
        #Listing the date-obs header keywords
        file_date_list = np.append(file_date_list,date)
        
        #Listing the date-obs header keyword 
        #file_date=np.append(file_date,date)

        #Formating the date-obs ketyword and inputting them in Pandas data-frame for plotting
        #file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in file_date]
        
        #Calculates the median count-rate ratio (observation / first observation for chip1 and chip2)
        data_median1=np.nanmedian(sci_chip1)/np.nanmedian(first_sci_chip1)
        data_median2=np.nanmedian(sci_chip2)/np.nanmedian(first_sci_chip2)
        print(rootname, exp_time,data_median1,data_median2)
        
        #Adds the ratio result to a list
        file_med1=np.append(file_med1,data_median1)
        file_med2=np.append(file_med2,data_median2)
        
        #Adds the filter name and filenames to a list
        file_filter = np.append(file_filter, filter_name)
        filename_list = np.append(filename,filename_list)
        hdu.close()
        
    #print(file_date_list)
    #Creates an ascii table as a .txt file
    t1 = Table()
    t1['Filename'] = filename_list
    t1['Date-Obs'] = file_date_list
    t1['filter'] = filter_name 
    t1['c1 median ratio (e-/sec)'] = file_med1
    t1['c2 median ratio (e-/sec)'] = file_med2
    t1.write('{}_medratio_eps.txt'.format(filter_name), format='ascii.fast_commented_header', overwrite=True)




    #Plot the median count-rate data for chip1 and chip2 (Does not read from the txt file)

    #print(file_med1)
    #print(file_med2)
    #x = mdates.date2num(file_date)
    #print(x)
    #Mean_c1=np.nanmean(file_med1)
    #STD_c1 =np.nanstd(file_med1)
    #
    #Mean_c2=np.nanmean(file_med2)
    #STD_c2 =np.nanstd(file_med2)
    #
    #
    #polyfit1=np.polyfit(x,file_med1,1)
    #polyfit1_data=((polyfit1[0]*x + polyfit1[1]))
    #
    #plt.scatter(file_date,file_med1)
    #plt.plot(x, polyfit1_data,'r')
    #plt.xlabel('Date-Obs')
    #plt.ylabel('Ratio')
    #plt.title(' {} Date Observation vs. Median Count Rate (ext4)\n Mean Value: {}  Std.Dev Value: {}'.format(filter_name,"%.3f" %Mean_c1, "%.3f" %STD_c1))
    #plt.xticks(rotation=30)
    #plt.grid()
    #plt.savefig('Date Observation vs. Median Count Rate (ext4).png')
    #plt.show()
    #plt.clf()
    #
    #polyfit2=np.polyfit(x,file_med2,1)
    #polyfit2_data=((polyfit2[0]*x+polyfit2[1]))
    #
    #plt.scatter(file_date,file_med2)
    #plt.plot(x, polyfit2_data,'r')
    #plt.xlabel('Date-Obs')
    #plt.ylabel('Ratio')
    #plt.title(' {} Date Observation vs. Median Count Rate (ext1)\n Mean Values: {}  Std.Dev Value: {}'.format(filter_name,"%.3f" %Mean_c2, "%.3f" %STD_c2))
    #plt.xticks(rotation=30)
    #plt.grid()
    #plt.savefig('Date Observation vs. Median Count Rate (ext1).png')
    #plt.show()
    #plt.clf() 
    #



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
    os.chdir(path) #Move to specifed directory
    files = sorted(glob.glob('*flt.fits')) #List all the files in alphabetical order
    med_count_rate(files) #Run function
    #os.system('cp *medratio_eps.txt /grp/hst/wfc3v/mmckay/internal_flats/med_countrate/') #Moved data to specfied directory







