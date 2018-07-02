#Filename: sep_filters.py
#Description: This reads the header of a file and sort
#Author: Myles McKay

from astropy.io import fits
import numpy as np
import os
import glob
import pdb
import shutil

current = os.getcwd()

base_path = '/grp/hst/wfc3v/mmckay/internal_flats/clean_deuterium_raw_files'
os.chdir(base_path)

F645N=os.path.join(base_path, 'D2_F645N')
F656N=os.path.join(base_path, 'D2_F656N')
F657N=os.path.join(base_path, 'D2_F657N')
F658N=os.path.join(base_path, 'D2_F658N')
F200LP=os.path.join(base_path,'D2_F200LP')
F350LP=os.path.join(base_path,'D2_F350LP')
F390W=os.path.join(base_path, 'D2_F390W')
F410M=os.path.join(base_path, 'D2_F410M')
F438W=os.path.join(base_path, 'D2_F438W')
F467M=os.path.join(base_path, 'D2_F467M')
F469N=os.path.join(base_path, 'D2_F469N')
F475W=os.path.join(base_path, 'D2_F475W')
F475X=os.path.join(base_path, 'D2_F475X')
F487N=os.path.join(base_path, 'D2_F487N')
F502N=os.path.join(base_path, 'D2_F502N')
F547M=os.path.join(base_path, 'D2_F547M')
F555W=os.path.join(base_path, 'D2_F555W')
F600LP=os.path.join(base_path,'D2_F600LP')
F606W=os.path.join(base_path, 'D2_F606W')
F621M=os.path.join(base_path, 'D2_F621M')
F625W=os.path.join(base_path, 'D2_F625W')
F631N=os.path.join(base_path, 'D2_F631N')
F665N=os.path.join(base_path, 'D2_F665N')
F673N=os.path.join(base_path, 'D2_F673N')
F680N=os.path.join(base_path, 'D2_F680N')
F689M=os.path.join(base_path, 'D2_F689M')
F763M=os.path.join(base_path, 'D2_F763M')
F775W=os.path.join(base_path, 'D2_F775W')
F814W=os.path.join(base_path, 'D2_F814W')
F845M=os.path.join(base_path, 'D2_F845M')
F850LP=os.path.join(base_path,'D2_F850LP')
F953N=os.path.join(base_path, 'D2_F953N')
FQ508N=os.path.join(base_path,'D2_FQ508N')
FQ619N=os.path.join(base_path,'D2_FQ619N')
FQ889N=os.path.join(base_path,'D2_FQ889N')
F343N =os.path.join(base_path,'D2_F343N')
F275W =os.path.join(base_path,'D2_F275W')
FQ387N=os.path.join(base_path,'D2_FQ387N')
F225W=os.path.join(base_path, 'D2_F225W')
F395N=os.path.join(base_path, 'D2_F395N')
FQ437N=os.path.join(base_path,'D2_FQ437N')
F373N=os.path.join(base_path, 'D2_F373N')
F336W=os.path.join(base_path, 'D2_F336W')
F300X=os.path.join(base_path, 'D2_F300X')
F390M=os.path.join(base_path, 'D2_F390M')
F218W=os.path.join(base_path, 'D2_F218W')
F280N=os.path.join(base_path, 'D2_F280N')  

for file in glob.glob('*.fits'):
    value = fits.getheader(file)['filter']
    print(value)
    os.system('mkdir D2_{}'.format(value))
	
    if 'F645N' in value:
        fileName= os.path.join(F645N, file)
        curentName= os.path.join(base_path, file)
        os.rename(curentName, fileName)
        print('Moving {} to F645N'.format(file))

    elif 'F200LP' in value:
       fileName= os.path.join(F200LP, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to F200LP'.format(file))
       
    elif 'F657N' in value:
       fileName= os.path.join(F657N, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to F657N'.format(file))  
      
    elif 'F658N' in value:
       fileName= os.path.join(F658N, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to F658N'.format(file))  
      
    elif 'F656N' in value:
       fileName= os.path.join(F656N, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to F656N'.format(file))  
      
    elif 'F350LP' in value:
       fileName= os.path.join(F350LP, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to F350LP'.format(file))  
      
    elif 'F390W' in value:
       fileName= os.path.join(F390W, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to F390W'.format(file))  
      
    elif 'F410M' in value:
       fileName= os.path.join(F410M, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to F410M'.format(file))  
      
    elif 'F438W' in value:
       fileName= os.path.join(F438W, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to F438W'.format(file))  
      
    elif 'F467M' in value:
       fileName= os.path.join(F467M, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to F467M'.format(file))  
      
    elif 'F469N' in value:
       fileName= os.path.join(F469N, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to F469N'.format(file))  
      
    elif 'F475W' in value:
       fileName= os.path.join(F475W, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to F475W'.format(file))  
      
    elif 'F475X' in value:
       fileName= os.path.join(F475X, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to F475X'.format(file))  
      
      
    elif 'F487N' in value:
       fileName= os.path.join(F487N, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to F487N'.format(file))  
      
      
    elif 'F502N' in value:
       fileName= os.path.join(F502N, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to F502N'.format(file))  
      
      
    elif 'F547M' in value:
       fileName= os.path.join(F547M, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to F547M'.format(file))  
      
      
    elif 'F555W' in value:
       fileName= os.path.join(F555W, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to F555W'.format(file))  
      
      
    elif 'F600LP' in value:
       fileName= os.path.join(F600LP, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to F600LP'.format(file))  
      
      
    elif 'F606W' in value:
       fileName= os.path.join(F606W, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to F606W'.format(file))  
      
      
    elif 'F621M' in value:
       fileName= os.path.join(F621M, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to F621M'.format(file))  
      
      
    elif 'F625W' in value:
       fileName= os.path.join(F625W, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to F625W'.format(file))  
      
      
    elif 'F631N' in value:
       fileName= os.path.join(F631N, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to F631N'.format(file))  
      
      
    elif 'F665N' in value:
       fileName= os.path.join(F665N, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to F665N'.format(file))  
      
      
    elif 'F673N' in value:
       fileName= os.path.join(F673N, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to F673N'.format(file))  
      
      
    elif 'F680N' in value:
       fileName= os.path.join(F680N, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to F680N'.format(file))  
      
      
    elif 'F689M' in value:
       fileName= os.path.join(F689M, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to F689M'.format(file))  
      
        
    elif 'F763M' in value:
       fileName= os.path.join(F763M, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to F763M'.format(file))  
            
    elif 'F775W' in value:
       fileName= os.path.join(F775W, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to F775W'.format(file))  
            
    elif 'F814W' in value:
       fileName= os.path.join(F814W, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to F814W'.format(file))  
            
    elif 'F845M' in value:
       fileName= os.path.join(F845M, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to F845M'.format(file))  
            
    elif 'F850LP' in value:
       fileName= os.path.join(F850LP, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to F850LP'.format(file))  
            
    elif 'F953N' in value:
       fileName= os.path.join(F953N, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to F953N'.format(file))  
            
    elif 'FQ508N' in value:
       fileName= os.path.join(FQ508N, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to FQ508N'.format(file))  
                  
    elif 'FQ619N' in value:
       fileName= os.path.join(FQ619N, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to FQ619N'.format(file))  
                  
    elif 'FQ889N' in value:
       fileName= os.path.join(FQ889N, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to FQ889N'.format(file)) 
                  
    elif 'F343N' in value:
       fileName= os.path.join(F343N, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to F343N'.format(file))  
                  
    elif 'F275W' in value:
       fileName= os.path.join(F275W, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to F275W'.format(file))  
                  
    elif 'FQ387N' in value:
       fileName= os.path.join(FQ387N, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to FQ387N'.format(file))  
 
                  
    elif 'FQ437N' in value:
       fileName= os.path.join(FQ437N, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to FQ437N'.format(file))  
                  
    elif 'F373N' in value:
       fileName= os.path.join(F373N, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to F373N'.format(file))  
                  
    elif 'F336W' in value:
       fileName= os.path.join(F336W, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to F336W'.format(file))  
                  
    elif 'F300X' in value:
       fileName= os.path.join(F300X, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to F300X'.format(file))  
                  
    elif 'F390M' in value:
       fileName= os.path.join(F390M, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to F390M'.format(file))  
                  
    elif 'F218W' in value:
       fileName= os.path.join(F218W, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to F218W'.format(file))  
                  
    elif 'F280N' in value:
       fileName= os.path.join(F280N, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to F280N'.format(file))  
                  
    elif 'F225W' in value:
       fileName= os.path.join(F225W, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to F225W'.format(file))  
                         
    elif 'F395N' in value:
       fileName= os.path.join(F395N, file)
       curentName= os.path.join(base_path, file)
       os.rename(curentName, fileName)
       print('Moving {} to F395N'.format(file))  


    else:
        print(value, 'Filter is not specified for {}'.format(file))


os.chdir(current)        