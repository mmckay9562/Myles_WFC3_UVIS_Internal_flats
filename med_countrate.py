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


def med_count_rate(files):
    '''
    Calling: python med_countrate.py --path='/grp/hst/wfc3v/mmckay/internal_flats/calibrated_files/tung3_F200LP/'
    '''

    first_file=fits.open(files[0])
    first_rootname= first_file[0].header['rootname']
    first_exptime= first_file[0].header['exptime']
    first_ccdgain= first_file[0].header['ccdgain']
    first_sci_chip1 = first_file[4].data
    first_dq_chip1  = first_file[6].data
    first_sci_chip2 = first_file[1].data
    first_dq_chip2  = first_file[3].data
    
    first_sci_chip1 = (first_sci_chip1 * first_ccdgain) / first_exptime
    first_sci_chip2 = (first_sci_chip2 * first_ccdgain) / first_exptime
    first_sci_chip1[first_dq_chip1 !=0]=np.nan
    first_sci_chip2[first_dq_chip2 !=0]=np.nan
    
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
        #date = date[:10]
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
        
        sci_chip1= (sci_chip1 * ccdgain) / exp_time
        sci_chip2= (sci_chip2 * ccdgain) / exp_time
        sci_chip1[dq_chip1 !=0]= np.nan
        sci_chip2[dq_chip2 !=0]= np.nan
    
        file_date_list = np.append(file_date_list,date)
        
        file_date=np.append(file_date,date)
        file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in file_date]
        
        data_median1=np.nanmedian(sci_chip1)/np.nanmedian(first_sci_chip1)
        data_median2=np.nanmedian(sci_chip2)/np.nanmedian(first_sci_chip2)
        print(rootname, exp_time,data_median1,data_median2)
        
        file_med1=np.append(file_med1,data_median1)
        file_med2=np.append(file_med2,data_median2)
        
        file_filter = np.append(file_filter, filter_name)
        filename_list = np.append(filename,filename_list)
        hdu.close()
        
    print(file_date_list)
    t1 = Table()
    t1['Filename'] = filename_list
    t1['Date-Obs'] = file_date_list
    t1['filter'] = filter_name 
    t1['c1 median ratio (e-/sec)'] = file_med1
    t1['c2 median ratio (e-/sec)'] = file_med2
    t1.write('{}_medratio_eps.txt'.format(filter_name), format='ascii.fast_commented_header', overwrite=True)





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
    os.chdir(path)
    files = sorted(glob.glob('*flt.fits'))
    med_count_rate(files)
    os.system('cp *medratio_eps.txt /grp/hst/wfc3v/mmckay/internal_flats/med_countrate/')







