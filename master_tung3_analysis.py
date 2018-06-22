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
from skimage.morphology import disk
from wfc3tools import calwf3
from astropy.stats import sigma_clip


def calibrate_raws(files):
    for im in files:
        hdu = fits.open(im, mode='update')
        hdu[0].header['PCTECORR'] = 'OMIT'
        hdu.close()
        calwf3(im)

    os.system('mkdir raw_crj_tra_csv_files')
    os.system('mv *raw.fits raw_crj_tra_csv_files')
    os.system('mv *.tra raw_crj_tra_csv_files')
    os.system('mv *crj.fits raw_crj_tra_csv_files')
    os.system('mv *.csv raw_crj_tra_csv_files')
    os.system('mv *asn.fits raw_crj_tra_csv_files')

def sigma_clipping(files):
    for im in files:
        hdu =  fits.open(im)
        rootname=hdu[0].header['rootname']
        sci_chip1 = hdu[4].data
        sci_chip2 = hdu[1].data
        dq_chip1=hdu[6].data
        dq_chip2=hdu[3].data
        sci_chip1[dq_chip1 != 0] = np.nan
        sci_chip2[dq_chip2 != 0] = np.nan
    
        
#        bins=np.linspace(-15,15,50) 
#        n,bins,patches=plt.hist(sci_chip1[~np.isnan(sci_chip1)], 50, facecolor='green', alpha=0.50) 
#        #Titles for the histogram    
#        plt.title("{} Bias File Chip1 SCI Histogram \n Mean {} Sigma {}".format(rootname, "%.3f" % np.nanmean(sci_chip1),"%.3f" %np.nanstd(sci_chip1)))   
#        plt.xlabel("Pixel Value Max={} Min={}".format("%.3f" % np.nanmax(sci_chip1),"%.3f" % np.nanmin(sci_chip1)))
#        plt.ylabel("Frequency")
#        plt.yscale('log')
#        plt.savefig('{}_mean_stacked_chip1_sci_hist.png'.format(rootname))
#        plt.clf()
#    
#        bins=np.linspace(-15,15,50) 
#        n,bins,patches=plt.hist(sci_chip2[~np.isnan(sci_chip2)], 50, facecolor='pink', alpha=0.50) 
#        #Titles for the histogram    
#        plt.title("{} Bias File Chip2 SCI Histogram \n Mean {} Sigma {}".format(rootname, "%.3f" % np.nanmean(sci_chip2),"%.3f" %np.nanstd(sci_chip2)))   
#        plt.xlabel("Pixel Value Max={} Min={}".format("%.3f" % np.nanmax(sci_chip2),"%.3f" % np.nanmin(sci_chip2)))
#        plt.ylabel("Frequency")
#        plt.yscale('log')
#        plt.savefig('{}_mean_stacked_chip2_sci_hist.png'.format(rootname))
#        plt.clf()
    
        clipped1 = sigma_clip(sci_chip1, sigma=3, iters = 3)
        data1 = clipped1.data
        mask1 = clipped1.mask
    #    print(clipped1.shape, data1.shape, mask1.shape)
        data1[~mask1 != True] = np.nan
        clipped1=data1
    #    print(clipped1.shape)
    #    print(~mask1[0,0])
    
        clipped2 = sigma_clip(sci_chip2, sigma=3, iters = 3)
        data2 = clipped2.data
        mask2 = clipped2.mask
    #    print(clipped2.shape, data.shape, mask.shape)
        data2[~mask2 != True] = np.nan
        clipped2=data2
        
        
         
#        n,bins,patches=plt.hist(clipped1[~np.isnan(clipped1)], 50, facecolor='blue', alpha=0.50) 
#        #Titles for the histogram
#        plt.title("{} Bias File Chip1 SCI Histogram \n Mean {} Sigma {}".format(rootname, "%.3f" % np.nanmean(clipped1),"%.3f" %np.nanstd(clipped1)))
#        plt.xlabel("Pixel Value Max={} Min={}".format("%.3f" % np.nanmax(clipped1),"%.3f" % np.nanmin(clipped1)))
#        plt.ylabel("Frequency")
#        plt.yscale('log')
#        plt.savefig('{}_clipped_stacked_chip1_sci_hist.png'.format(rootname))
#        plt.clf()
#    
#     
#        n,bins,patches=plt.hist(clipped2[~np.isnan(clipped2)], 50, facecolor='red', alpha=0.50) 
#        #Titles for the histogram
#        plt.title("{} Bias File Chip1 SCI Histogram \n Mean {} Sigma {}".format(rootname, "%.3f" % np.nanmean(clipped2),"%.3f" %np.nanstd(clipped2)))
#        plt.xlabel("Pixel Value Max={} Min={}".format("%.3f" % np.nanmax(clipped2),"%.3f" % np.nanmin(clipped2)))
#        plt.ylabel("Frequency")
#        plt.yscale('log')
#        plt.savefig('{}_clipped_stacked_chip2_sci_hist.png'.format(rootname))
#        plt.clf()
        
        where_are_nans1 = np.isnan(clipped1)
        where_are_nans2 = np.isnan(clipped2)
        clipped1[where_are_nans1] = 0
        clipped2[where_are_nans2] = 0
        hdu.close()
        
        hdulist = fits.open(im)
        hdulist[1].data = clipped2
        hdulist[4].data = clipped1
        hdulist[3].header['EXTNAME'] = 'DQ'
        hdulist[6].header['EXTNAME'] = 'DQ'
        hdulist[3].header['EXTVER'] = '1'
        hdulist[6].header['EXTVER'] = '2'
        hdulist.writeto('{}_clipped_flt.fits'.format(rootname), overwrite = True)
        hdulist.close()
        


def Normalize(files):
    file_number=[]
    file_date=[]
    file_mean1=[]
    file_mean2=[]
    
    
    for i in files:
        hdu = fits.open(i)
        date=hdu[0].header['date-obs']
        filter_name1=hdu[0].header['filter']
        print(hdu[0].header['date-obs'])
        sci_chip1 = hdu[4].data
        sci_chip2 = hdu[1].data
        avg_chip1 = np.mean(sci_chip1)
        avg_chip2 = np.mean(sci_chip2)
    
        normalized_chip1 = sci_chip1 / avg_chip1
        normalized_chip2 = sci_chip2 / avg_chip2

        file_date=np.append(file_date,date)
        file_mean1=np.append(file_mean1,np.mean(normalized_chip1))
        file_mean2=np.append(file_mean2,np.mean(normalized_chip2))
        
        hdu.close()
    
       
    
        hdulist = fits.open(i)
        hdulist[1].data = normalized_chip2
        hdulist[4].data = normalized_chip1
        hdulist[3].header['EXTNAME'] = 'DQ'
        hdulist[6].header['EXTNAME'] = 'DQ'
        hdulist[3].header['EXTVER'] = '1'
        hdulist[6].header['EXTVER'] = '2'
        hdulist.writeto('norm_{}_flt.fits'.format(hdu[0].header['rootname'],overwrite=True))
        hdulist.close()
    
#    print(file_mean1)
#    print(file_mean2)
#    file_date = [pd.to_datetime(d,format='%Y-%m-%d') for d in file_date]
#    Mean_c1=np.mean(file_mean1)
#    STD_c1 =np.std(file_mean1)
#    
#    Mean_c2=np.mean(file_mean2)
#    STD_c2 =np.std(file_mean2)
#    
#    upper_sigma_c1=Mean_c1 + 3.0*STD_c1
#    lower_sigma_c1=Mean_c1 - 3.0*STD_c1
#    
#    upper_sigma_c2=Mean_c2 + 3.0*STD_c2
#    lower_sigma_c2=Mean_c2 - 3.0*STD_c2
    
#    plt.scatter(file_date,file_mean1)
#    plt.xlabel('Date-Obs')
#    plt.ylabel('Mean Values')
#    plt.title('{} flats chip1 Statistics\n Average Value: {}  Std.Dev Value: {}'.format(filter_name1,"%.3f" %Mean_c1, "%.3f" %STD_c1))
#    plt.xticks(rotation=30)
#    plt.axhline(y=upper_sigma_c1, xmin=-100,xmax=100,linewidth=2, color='red')
#    plt.axhline(y=lower_sigma_c1, xmin=-100,xmax=100,linewidth=2, color='red')
#    plt.axhline(y=Mean_c1, xmin=-100,xmax=100,linewidth=1, color='blue')
#    plt.savefig('{} Normalized_Statistics_chip1_data_plot.png'.format(filter_name1),clobber=True)
#    plt.clf()
#    
#    plt.scatter(file_date,file_mean2)
#    plt.xlabel('Date-Obs')
#    plt.ylabel('Mean Values')
#    plt.title('{} flats chip2 Statistics\n Average Value: {}  Std.Dev Value: {}'.format(filter_name1,"%.3f" %Mean_c2, "%.3f" %STD_c2))
#    plt.xticks(rotation=30)
#    plt.axhline(y=upper_sigma_c2, xmin=-100,xmax=100,linewidth=2, color='red')
#    plt.axhline(y=lower_sigma_c2, xmin=-100,xmax=100,linewidth=2, color='red')
#    plt.axhline(y=Mean_c2, xmin=-100,xmax=100,linewidth=1, color='blue')
#    plt.savefig('{} Normalized_Statistics_chip2_data_plot.png'.format(filter_name1),clobber=True)
#    plt.xlabel('Date-Obs')
#    plt.ylabel('Mean Values')
#    plt.clf()  

def CR_grow_main(fitsName1):
    hdu=fits.open(fitsName1, mode='update')

    dq3 = fits.getdata(fitsName1, ext=3)
    dq3_orig=fits.getdata(fitsName1, ext=3)
    dq6 = fits.getdata(fitsName1, ext=6)
    dq6_orig = fits.getdata(fitsName1, ext=6)

    dq3[np.where(dq3 != 8192)] = 0
    dq6[np.where(dq6 != 8192)] = 0


    dq3_grown = morph.dilation(dq3.byteswap().newbyteorder('='), disk(5))
    dq6_grown = morph.dilation(dq6.byteswap().newbyteorder('='), disk(5))

    for i in np.arange(0,14):
        value=[2**i]
        dq3_grown[np.where(dq3_orig & value)] += value
        dq6_grown[np.where(dq6_orig & value)] += value


    hdu.close()
    hdulist = fits.open(fitsName1)
    hdulist[3].data = dq3_grown
    hdulist[6].data = dq6_grown
    hdulist[3].header['EXTNAME'] = 'DQ'
    hdulist[6].header['EXTNAME'] = 'DQ'
    hdulist[3].header['EXTVER'] = '1'
    hdulist[6].header['EXTVER'] = '2'
    hdulist.writeto('crrg_norm_{}_{}_flt.fits'.format(hdu[0].header['date-obs'],hdu[0].header['rootname'],overwrite=True))
    hdulist.close()



def mean_stack1(j):
    """This is a function to create a mean stack of a data cube through the z axis. It will print the column it is currently working on.

    mean_stack1(j)
    Parameters:
        j: int
            The number of columns to mean stack together at a time


    """
    col_mean1=np.nanmean(data_cube1[:,:,j],axis=0)
    #print(j)
    return col_mean1

def mean_stack2(j):
    """This is a function to create a mean stack of a data cube through the z axis. It will print the column it is currently working on.

    mean_stack1(j)
    Parameters:
        j: int
            The number of columns to mean stack together at a time


    """
    col_mean2=np.nanmean(data_cube2[:,:,j],axis=0)
    #print(j)
    return col_mean2

def sum_stacked1(j):
    """This is a function to create a sum stack of a data cube through the z axis. It will print the column it is currently working on.

    sum_stacked1(j)
    Parameters:
        j: int
            The number of columns to sum stack together at a time


    """
    col_err1=np.nansum(err_data_cube1[:,:,j]**2,axis=0)
#   nf1=np.count_nonzero(~np.isnan(err_data_cube1[:,:,j]))
    #print(j)
    return col_err1


def sum_stacked2(j):
    """This is a function to create a mean stack of a data cube through the z axis. It will print the column it is currently working on.

    mean_stack1(j)
    Parameters:
        j: int
            The number of columns to mean stack together at a time


    """
    col_err2=np.nansum(err_data_cube2[:,:,j]**2,axis=0)
    #print(j)
    return col_err2     


def stacking(files):
    #This is to masked the data with the DQ array 
    i=0
    full_frame = np.zeros((4112,4096))
    New_dq = np.zeros((2051,4096))
    for fitsname in files:
        h = fits.open(fitsname)
        sci_chip1=h[4].data
        err_chip1=h[5].data
        dq_chip1=h[6].data
        sci_chip2=h[1].data
        err_chip2=h[2].data
        dq_chip2=h[3].data
        date = h[0].header['date-obs']
    #    print(err_chip2[1703:1708,2693:2698])
    
        h.close()
        sci_chip1[dq_chip1 != 0] = np.nan
        sci_chip2[dq_chip2 != 0] = np.nan
        err_chip1[dq_chip1 != 0] = np.nan
        err_chip2[dq_chip2 != 0] = np.nan
     
#        print(err_chip2[1703:1708,2693:2698])
    
    
    #Inputs the masked data in to the data cube
        data_cube1[i] = sci_chip1
        data_cube2[i] = sci_chip2
        err_data_cube1[i]=err_chip1
        err_data_cube2[i]=err_chip2
        i+=1
        print(i)
    
    
    nf1=np.count_nonzero(~np.isnan(err_data_cube1[:,:,range(4096)]),axis=0)
    nf2=np.count_nonzero(~np.isnan(err_data_cube2[:,:,range(4096)]),axis=0)
    #Using parellel computing to median stack the columns for faster results (Do not use 8 when running the locally)    
    p=Pool(9)
    
    #result1 = p.map(median_stack1,range(4096))
    #result2 = p.map(median_stack2,range(4096))
    result1 = p.map(mean_stack1,range(4096))
    result2 = p.map(mean_stack2,range(4096))
    err_result1=p.map(sum_stacked1,range(4096))
    err_result2=p.map(sum_stacked2,range(4096))
    
    
    #Puts the final 2d list into a numpy array
    result1=np.array(result1)
    result2=np.array(result2)
    err_result1=np.array(err_result1)
    err_result2=np.array(err_result2)
    
    #print(err_result1[1344:1350,1303:1309])
    
    #print(err_result1[500,500])
    #print(err_result2[500,500])
    #The data deminsions changes when the median stack occures so we transpose the data to re shape the array
    result1 =np.transpose(result1)
    result2 =np.transpose(result2)
    err_result1=np.transpose(err_result1)
    err_result2=np.transpose(err_result2)
    full_frame[:2051,:4096]=result2[:,:]
    full_frame[2061:,:4096]=result1[:,:]
    #print(err_result2[1703:1708,2693:2698])
    #nf1=np.count_nonzero(~np.isnan(err_result1))
    #nf2=np.count_nonzero(~np.isnan(err_result2))
    #print(nf1[1703:1708,2693:2698])
    #print(nf2[1703:1708,2693:2698])
    err_result1=(1/nf1)*np.sqrt(err_result1)
    err_result2=(1/nf2)*np.sqrt(err_result2)
    
    #print(err_result1[1344:1350,1303:1309])
    #print(err_result2[1703:1708,2693:2698])
    new_hdul = fits.HDUList()
    new_hdul.append(fits.ImageHDU(full_frame))
    new_hdul.append(fits.ImageHDU(result1))
    new_hdul.append(fits.ImageHDU(err_result1))
    new_hdul.append(fits.ImageHDU(New_dq))
    new_hdul.append(fits.ImageHDU(result2))
    new_hdul.append(fits.ImageHDU(err_result2))
    new_hdul.append(fits.ImageHDU(New_dq))
    new_hdul.writeto('{}_Flats_crr_err_stacked_files.fits'.format(date[0:4]),overwrite=True)
    new_hdul.close()
    
    
    where_are_nans1 = np.isnan(result1)
    where_are_nans2 = np.isnan(result2)
    where_are_nans_full = np.isnan(full_frame)
    result1[where_are_nans1] = 0
    result2[where_are_nans2] = 0
    full_frame[where_are_nans_full] = 0
    where_are_nans_err1 =np.isnan(err_result1)
    where_are_nans_err2 =np.isnan(err_result2)
    err_result1[where_are_nans_err1] = 0
    err_result2[where_are_nans_err2] = 0
    
    Master_new_hdul = fits.HDUList()
    Master_new_hdul.append(fits.ImageHDU(full_frame))
    Master_new_hdul.append(fits.ImageHDU(result1))
    Master_new_hdul.append(fits.ImageHDU(err_result1))
    Master_new_hdul.append(fits.ImageHDU(New_dq))
    Master_new_hdul.append(fits.ImageHDU(result2))
    Master_new_hdul.append(fits.ImageHDU(err_result2))
    Master_new_hdul.append(fits.ImageHDU(New_dq))
    Master_new_hdul.writeto('Master_flats_crr_err_stacked_files.fits',overwrite=True)
    Master_new_hdul.close()


def stacked_ratio(files):
    ratio_full_frame = np.zeros((4112,4096))
    ratio_new_dq = np.zeros((2051,4096))
    first=(files[0])
    fiducial_stacked_file = fits.open(files[0])
    fiducial_stacked_chip1=fiducial_stacked_file[1].data
    fiducial_stacked_chip2=fiducial_stacked_file[4].data
    fiducial_stacked_file.close()


    
    for im in files:
        hdu=fits.open(im)
#        first=fits.open(files[0])
#        print(first)
        sci_chip1 = hdu[1].data
        sci_chip2 = hdu[4].data

        sci_chip1[sci_chip1 == 0] = 1
        sci_chip2[sci_chip2 == 0] = 1
    
        ratio1 = fiducial_stacked_chip1 / sci_chip1 
        ratio2 = fiducial_stacked_chip2 / sci_chip2 
#        print(ratio_full_frame.shape)
#        print(ratio2.shape, ratio1.shape)
        ratio_full_frame[:2051,:4096]=ratio2[:,:]
        ratio_full_frame[2061:,:4096]=ratio1[:,:]
        
    
    
        where_are_nans1 = np.isnan(ratio1)
        where_are_nans2 = np.isnan(ratio2)
        where_are_nans_ratio =  np.isnan(ratio_full_frame)

        ratio1[where_are_nans1] = 0
        ratio2[where_are_nans2] = 0
        ratio_full_frame[where_are_nans_ratio] = 0
        hdu.close()
    
        Ratio_new_hdul = fits.HDUList()
        Ratio_new_hdul.append(fits.ImageHDU(ratio_full_frame))
        Ratio_new_hdul.append(fits.ImageHDU(ratio2))
        Ratio_new_hdul.append(fits.ImageHDU(ratio_new_dq))
        Ratio_new_hdul.append(fits.ImageHDU(ratio_new_dq))
        Ratio_new_hdul.append(fits.ImageHDU(ratio1))
        Ratio_new_hdul.writeto('{}_vs_{}_ratios.fits'.format(first[0:4],im[0:4]),overwrite=True)
        Ratio_new_hdul.close()
    



    

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
    #os.system('mkdir zzzzzzzz_dummy_dir')

    #calibrate _raw.fits files
    print('-----------------------Calibrating raw files-------------------------------')
    raw_files = glob.glob('*raw.fits')
    calibrate_raws(raw_files)

    #Sigma clip flt files
    print('-------------------------------Sigma clipping calibrated files-------------------------------')
    cal_files = glob.glob('*flt.fits')
    sigma_clipping(cal_files)

    # Make directories to store data
    print('-------------------------------Making needed directories-------------------------------')
    os.system('mkdir Normilized_flts clipped_flts flt_files clipped_unclipped_histograms')
    os.system('mv i*q_flt.fits flt_files') #Stores original _flt.fits files in directory
    
    #Normilize clipped flt.fits files
    print('-------------------------------Normalizing clipped flt.fits files-------------------------------')
    clipped_files = glob.glob('*clipped_flt.fits')
    Normalize(clipped_files)
    os.system('mv *clipped_flt.fits clipped_flts') #Stores clipped_flt.fit files in directory 
    os.system('mv *hist.png clipped_unclipped_histograms') #Stores clipped histograms files in directory
#    CR_files=glob.glob('norm*flt.fits')
#    p=Pool(6)
#    p.map(CR_grow_main,CR_files)
    os.system('mv *plot.png Normilized_flts')
    list_of_files= glob.glob('norm_*_flt.fits')   
    for i in list_of_files:
        hdu=fits.open(i)
        date=hdu[0].header['date-obs']
        rootname=hdu[0].header['rootname']
        filter_name=hdu[0].header['filter']
        hdu.close()
        #print(date[0:4])
        if date[0:4] == '2009':
            os.system('mkdir 2009_{}_data'.format(filter_name))
            os.system('cp *{}*.fits 2009_*_data'.format(rootname))
    
        if date[0:4] == '2010':
            os.system('mkdir 2010_{}_data'.format(filter_name))
            os.system('cp *{}*.fits 2010_*_data'.format(rootname))
    
        if date[0:4] == '2011':
            os.system('mkdir 2011_{}_data'.format(filter_name))
            os.system('cp *{}*.fits 2011_*_data'.format(rootname))
    
        if date[0:4] == '2012':
            os.system('mkdir 2012_{}_data'.format(filter_name))
            os.system('cp *{}*.fits 2012_*_data'.format(rootname))
    
        if date[0:4] == '2013':
            os.system('mkdir 2013_{}_data'.format(filter_name))
            os.system('cp *{}*.fits 2013_*_data'.format(rootname))
    
        if date[0:4] == '2014':
            os.system('mkdir 2014_{}_data'.format(filter_name))
            os.system('cp *{}*.fits 2014_*_data'.format(rootname))
    
        if date[0:4] == '2015':
            os.system('mkdir 2015_{}_data'.format(filter_name))
            os.system('cp *{}*.fits 2015_*_data'.format(rootname))
    
        if date[0:4] == '2016':
            os.system('mkdir 2016_{}_data'.format(filter_name))
            os.system('cp *{}*.fits 2016_*_data'.format(rootname))

        if date[0:4] == '2017':
            os.system('mkdir 2017_{}_data'.format(filter_name))
            os.system('cp *{}*.fits 2017_*_data'.format(rootname))
    base_path = path
    #print(path)
    os.system('mkdir stacking_split')
    os.system('mv 20*_data stacking_split')
    os.system('pwd')
    path1 = '{}/stacking_split'.format(base_path)
    #print(path1)
    os.chdir(path1)
    for subdir, dirs, files in os.walk(path1):
        for dir in dirs:
            path1=os.path.join(subdir,dir)
            print(path1)
            os.chdir(path1)
            files = glob.glob('norm_*_flt.fits')
            #print(files)
            hdr = fits.getheader(files[0], 1)
            nx = hdr['NAXIS1']
            ny = hdr['NAXIS2']
            nf = len(files)
            data_cube1 = np.zeros((nf, ny, nx), dtype=float)
            data_cube2 = np.zeros((nf, ny, nx), dtype=float)
            err_data_cube1 = np.zeros((nf, ny, nx), dtype=float)
            err_data_cube2 = np.zeros((nf, ny, nx), dtype=float)
            os.system('pwd')
            stacking(files)
            os.system('cp  *_Flats_crr_err_stacked_files.fits ..')
    
    os.chdir('{}/stacking_split'.format(base_path))
    os.system('pwd')
    Stacked_files = sorted(glob.glob('*files.fits'))
    stacked_ratio(Stacked_files)
    os.system('mkdir ratios')
    os.system('mv *vs*.fits ratios')
