"""
DELTA CHANGE BIAS CORRECTION, GENERATION OF TIME SERIES


Consider that the french case has 5x5 grid cells. it has to be run alone

IMPORTANT: THIS USES MUCH DISC SPACE, THUS RUN IT PER SITE BY CHANGING THE RANGE IN n_site
located aprox. in line 46



"""
    
import pandas as pd
import numpy as np
import netCDF4 as nc 
import math
import datetime
import os
import matplotlib.pyplot as plt
import sys

#Set the list of models

dates = pd.read_excel('C:/Users/epz/Desktop/UEF/PET/Python/dates1971.xlsx')
month = dates.Month
year = dates.Year
julian_days = dates. Julian
julian_days = np.array(julian_days)
dates_no_leap = pd.read_excel('C:/Users/epz/Desktop/UEF/PET/Python/dates1971_noleap.xlsx')
month_noleap = dates_no_leap.Month 
year_noleap = dates_no_leap.Year
julian_days_no_leap = dates_no_leap.Julian

julian_days_n = np.arange(0,len(julian_days))
grid_index = pd.read_excel('C:/Users/epz/Desktop/UEF/PET/Python/gridcell_index.xlsx')
grid_index_fr = pd.read_excel('C:/Users/epz/Desktop/UEF/PET/Python/gridcell_index_fr.xlsx')

sites_list = ['at','dk','dl','fi','fr','gr','it','ro','sp','uk']

# the following conditional avoids warning to pop up when running the script
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

for n_site in np.arange(0,10): # change here based on the site to be corrected
    
    if n_site !=4:
        y_grid = np.arange(0,6)
        x_grid = np.arange(0,6)
        print('Processing data for',sites_list[n_site])
        print('===============================')
        print('===============================')
        #Imports correction factor for all variables
        correction_tas_list = [filename for filename in os.listdir('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/') if filename.startswith('tas_')]
        correction_tasmax_list = [filename for filename in os.listdir('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/') if filename.startswith('tasmax_')]
        correction_tasmin_list = [filename for filename in os.listdir('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/') if filename.startswith('tasmin_')]
        correction_sfcWind_list = [filename for filename in os.listdir('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/') if filename.startswith('sfcWind_')]
        correction_sund_list = [filename for filename in os.listdir('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/') if filename.startswith('sund_')]
        correction_hurs_list = [filename for filename in os.listdir('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/') if filename.startswith('hurs_')]
        
        # Import list from the RCM, setting up the models
        tas_list = [filename for filename in os.listdir('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/') if filename.startswith('tas_EUR') and filename.endswith('all.nc')]
        tasmax_list = [filename for filename in os.listdir('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/') if filename.startswith('tasmax_EUR') and filename.endswith('all.nc')]
        tasmin_list = [filename for filename in os.listdir('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/') if filename.startswith('tasmin_EUR') and filename.endswith('all.nc')]
        sfcWind_list = [filename for filename in os.listdir('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/') if filename.startswith('sfcWind_EUR') and filename.endswith('all.nc')]
        orog_list = [filename for filename in os.listdir('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/') if filename.startswith('orog_EUR')]
        sund_list = [filename for filename in os.listdir('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/') if filename.startswith('sund_EUR') and filename.endswith('all.nc')]
        hurs_list = [filename for filename in os.listdir('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/') if filename.startswith('hurs_EUR') and filename.endswith('all.nc')]
        
        #loop for the climate models, one as ref and the others are corrected 
        
        annual_mean_tas_site_ref=np.zeros([1,10])#top row is the uncorrected (reference) models and the other rows are the models corrected to the top-row model
        annual_mean_tas_site_cor=np.zeros([10,10])
        annual_mean_tasmax_site_ref=np.zeros([1,10])#top row is the uncorrected (reference) models and the other rows are the models corrected to the top-row model
        annual_mean_tasmax_site_cor=np.zeros([10,10])
        annual_mean_tasmin_site_ref=np.zeros([1,10])#top row is the uncorrected (reference) models and the other rows are the models corrected to the top-row model
        annual_mean_tasmin_site_cor=np.zeros([10,10])
        annual_mean_sfcWind_site_ref=np.zeros([1,10])#top row is the uncorrected (reference) models and the other rows are the models corrected to the top-row model
        annual_mean_sfcWind_site_cor=np.zeros([10,10])
        annual_mean_hurs_site_ref=np.zeros([1,10])#top row is the uncorrected (reference) models and the other rows are the models corrected to the top-row model
        annual_mean_hurs_site_cor=np.zeros([10,10])
        annual_mean_sund_site_ref=np.zeros([1,10])#top row is the uncorrected (reference) models and the other rows are the models corrected to the top-row model
        annual_mean_sund_site_cor=np.zeros([10,10])

        
        for i in np.arange(0,10):#Gets the data for all variables of the climate model acting as reference
            print('Using climate model '+str(i+1)+' as reference...')
            print('===============================')
            file_tas = tas_list[i]
            tas_model = nc.Dataset('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/'+file_tas)
            ref_rcm_tas = tas_model.variables['tas'][:,:,:]
            tas_model.close()        
            file_tasmax = tasmax_list[i]
            tasmax_model = nc.Dataset('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/'+file_tasmax)
            ref_rcm_tasmax = tasmax_model.variables['tasmax'][:,:,:]
            file_tasmin = tasmin_list[i]
            tasmin_model = nc.Dataset('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/'+file_tasmin)
            ref_rcm_tasmin = tasmin_model.variables['tasmin'][:,:,:]
            file_sfcWind = sfcWind_list[i]
            sfcWind_model = nc.Dataset('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/'+file_sfcWind)
            ref_rcm_sfcWind = sfcWind_model.variables['sfcWind'][:,:,:]
            file_sund = sund_list[i]
            sund_model = nc.Dataset('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/'+file_sund)
            ref_rcm_sund = sund_model.variables['sund'][:,:,:]
            file_hurs = hurs_list[i]
            hurs_model = nc.Dataset('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/'+file_hurs)
            ref_rcm_hurs = hurs_model.variables['hurs'][:,:,:]
            
            if i<=6:
                annual_mean_tas_site_ref[0,i]=np.mean(ref_rcm_tas[(year >= 1976) & (year <= 2005)])-273.15
                annual_mean_tasmax_site_ref[0,i]=np.mean(ref_rcm_tasmax[(year >= 1976) & (year <= 2005)])-273.15
                annual_mean_tasmin_site_ref[0,i]=np.mean(ref_rcm_tasmin[(year >= 1976) & (year <= 2005)])-273.15
                annual_mean_sfcWind_site_ref[0,i]=np.mean(ref_rcm_sfcWind[(year >= 1976) & (year <= 2005)])
                annual_mean_sund_site_ref[0,i]=np.mean(ref_rcm_sund[(year >= 1976) & (year <= 2005)])
                annual_mean_hurs_site_ref[0,i]=np.mean(ref_rcm_hurs[(year >= 1976) & (year <= 2005)])
            else:
                annual_mean_tas_site_ref[0,i]=np.mean(ref_rcm_tas[(year_noleap >= 1976) & (year_noleap <= 2005)])-273.15
                annual_mean_tasmax_site_ref[0,i]=np.mean(ref_rcm_tasmax[(year_noleap >= 1976) & (year_noleap <= 2005)])-273.15
                annual_mean_tasmin_site_ref[0,i]=np.mean(ref_rcm_tasmin[(year_noleap >= 1976) & (year_noleap <= 2005)])-273.15
                annual_mean_sfcWind_site_ref[0,i]=np.mean(ref_rcm_sfcWind[(year_noleap >= 1976) & (year_noleap <= 2005)])
                annual_mean_hurs_site_ref[0,i]=np.mean(ref_rcm_hurs[(year_noleap >= 1976) & (year_noleap <= 2005)])
                annual_mean_sund_site_ref[0,i]=np.mean(ref_rcm_sund[(year_noleap >= 1976) & (year_noleap <= 2005)])
            for ii in np.arange(0,7):#climate model
                if i != ii:
                    file_tas_bc = tas_list[ii]
                    tas_model_bc = nc.Dataset('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/'+file_tas_bc)
                    bc_rcm_tas = tas_model_bc.variables['tas'][:,:,:]-273.15
                    tas_model_bc.close
                    raw_rcm_tas = bc_rcm_tas
                    file_tasmax_bc = tasmax_list[ii]
                    tasmax_model_bc = nc.Dataset('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/'+file_tasmax_bc)
                    bc_rcm_tasmax = tasmax_model_bc.variables['tasmax'][:,:,:]-273.15
                    tasmax_model_bc.close
                    raw_rcm_tasmax = bc_rcm_tasmax
                    file_tasmin_bc = tasmin_list[ii]
                    tasmin_model_bc = nc.Dataset('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/'+file_tasmin_bc)
                    bc_rcm_tasmin = tasmin_model_bc.variables['tasmin'][:,:,:]-273.15
                    tasmin_model_bc.close
                    raw_rcm_tasmin = bc_rcm_tasmin
                    file_hurs_bc = hurs_list[ii]
                    hurs_model_bc = nc.Dataset('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/'+file_hurs_bc)
                    bc_rcm_hurs = hurs_model_bc.variables['hurs'][:,:,:]
                    hurs_model_bc.close
                    raw_rcm_hurs = bc_rcm_hurs
                    file_sund_bc = sund_list[ii]
                    sund_model_bc = nc.Dataset('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/'+file_sund_bc)
                    bc_rcm_sund = sund_model_bc.variables['sund'][:,:,:]
                    sund_model_bc.close
                    raw_rcm_sund = bc_rcm_sund
                    file_sfcWind_bc = sfcWind_list[ii]
                    sfcWind_model_bc = nc.Dataset('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/'+file_sfcWind_bc)
                    bc_rcm_sfcWind = sfcWind_model_bc.variables['sfcWind'][:,:,:]
                    sfcWind_model_bc.close
                    raw_rcm_sfcWind = bc_rcm_sfcWind
                    
                    
                    print ('Reading the correction factors for climate model '+str(1+ii)+'...')
                    #print('===============================')
                    
                    #Next line reads and stored the monthly correction factor for each cell
                    # The correction factor was estimated as model-reference
                    #Thus, the corrected projection is equal to BCmod = RAWmod-correction
                    for cell_index in np.arange (0,36):# This line has to be changed for the french case as it has a 5x5 domain
                        for n_month in np.arange(1,13):
                            correction_ref=np.load('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/'+correction_tas_list[i])
                            correction_factor_tas = correction_ref[cell_index,ii,n_month-1]
                            bc_rcm_tas[:,grid_index.lat_cell[cell_index],grid_index.lon_cell[cell_index]][month==n_month]= raw_rcm_tas[:,grid_index.lat_cell[cell_index],grid_index.lon_cell[cell_index]][month==n_month]-correction_factor_tas
                            correction_ref=np.load('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/'+correction_tasmax_list[i])
                            correction_factor_tasmax = correction_ref[cell_index,ii,n_month-1]
                            bc_rcm_tasmax[:,grid_index.lat_cell[cell_index],grid_index.lon_cell[cell_index]][month==n_month]= raw_rcm_tasmax[:,grid_index.lat_cell[cell_index],grid_index.lon_cell[cell_index]][month==n_month]-correction_factor_tasmax
                            correction_ref=np.load('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/'+correction_tasmin_list[i])
                            correction_factor_tasmin = correction_ref[cell_index,ii,n_month-1]
                            bc_rcm_tasmin[:,grid_index.lat_cell[cell_index],grid_index.lon_cell[cell_index]][month==n_month]= raw_rcm_tasmin[:,grid_index.lat_cell[cell_index],grid_index.lon_cell[cell_index]][month==n_month]-correction_factor_tasmin
                            correction_ref=np.load('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/'+correction_hurs_list[i])
                            correction_factor_hurs = correction_ref[cell_index,ii,n_month-1]
                            bc_rcm_hurs[:,grid_index.lat_cell[cell_index],grid_index.lon_cell[cell_index]][month==n_month]= raw_rcm_hurs[:,grid_index.lat_cell[cell_index],grid_index.lon_cell[cell_index]][month==n_month]-correction_factor_hurs
                            correction_ref=np.load('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/'+correction_sund_list[i])
                            correction_factor_sund = correction_ref[cell_index,ii,n_month-1]
                            bc_rcm_sund[:,grid_index.lat_cell[cell_index],grid_index.lon_cell[cell_index]][month==n_month]= raw_rcm_sund[:,grid_index.lat_cell[cell_index],grid_index.lon_cell[cell_index]][month==n_month]-correction_factor_sund
                            correction_ref=np.load('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/'+correction_sfcWind_list[i])
                            correction_factor_sfcWind = correction_ref[cell_index,ii,n_month-1]
                            bc_rcm_sfcWind[:,grid_index.lat_cell[cell_index],grid_index.lon_cell[cell_index]][month==n_month]= raw_rcm_sfcWind[:,grid_index.lat_cell[cell_index],grid_index.lon_cell[cell_index]][month==n_month]-correction_factor_sfcWind
                            
                    #np.save
                    #np.save('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/corrected/tas_bc_delta_ref'+str(i)+'_corr'+str(ii)+'.npy',bc_rcm_tas)
                    bc_rcm_tas.dump('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/corrected/tas_bc_delta_ref'+str(i)+'_corr'+str(ii)+'.npy')
                    annual_mean_tas_site_cor[ii,i]=np.mean(bc_rcm_tas[(year >= 1976) & (year <= 2005)])
                    bc_rcm_tasmax.dump('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/corrected/tasmax_bc_delta_ref'+str(i)+'_corr'+str(ii)+'.npy')
                    annual_mean_tasmax_site_cor[ii,i]=np.mean(bc_rcm_tasmax[(year >= 1976) & (year <= 2005)])
                    bc_rcm_tasmin.dump('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/corrected/tasmin_bc_delta_ref'+str(i)+'_corr'+str(ii)+'.npy')
                    annual_mean_tasmin_site_cor[ii,i]=np.mean(bc_rcm_tasmin[(year >= 1976) & (year <= 2005)])
                    bc_rcm_sfcWind.dump('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/corrected/sfcWind_bc_delta_ref'+str(i)+'_corr'+str(ii)+'.npy')
                    annual_mean_sfcWind_site_cor[ii,i]=np.mean(bc_rcm_sfcWind[(year >= 1976) & (year <= 2005)])
                    bc_rcm_hurs.dump('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/corrected/hurs_bc_delta_ref'+str(i)+'_corr'+str(ii)+'.npy')
                    annual_mean_hurs_site_cor[ii,i]=np.mean(bc_rcm_hurs[(year >= 1976) & (year <= 2005)])
                    bc_rcm_sund.dump('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/corrected/sund_bc_delta_ref'+str(i)+'_corr'+str(ii)+'.npy')
                    annual_mean_sund_site_cor[ii,i]=np.mean(bc_rcm_sund[(year >= 1976) & (year <= 2005)])
                    print('Done')
                    
                    
            for iii in np.arange(7,10):#climate model
                if i != iii:
                    file_tas_bc = tas_list[iii]
                    tas_model_bc = nc.Dataset('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/'+file_tas_bc)
                    bc_rcm_tas = tas_model_bc.variables['tas'][:,:,:]-273.15
                    #print('Tas netcdf shape: ', bc_rcm_tas.shape)
                    tas_model_bc.close
                    raw_rcm_tas = bc_rcm_tas
                    file_tasmax_bc = tasmax_list[iii]
                    tasmax_model_bc = nc.Dataset('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/'+file_tasmax_bc)
                    bc_rcm_tasmax = tasmax_model_bc.variables['tasmax'][:,:,:]-273.15
                    #print('Tasmax netcdf shape: ', bc_rcm_tasmax.shape)
                    tasmax_model_bc.close
                    raw_rcm_tasmax = bc_rcm_tasmax
                    file_tasmin_bc = tasmin_list[iii]
                    tasmin_model_bc = nc.Dataset('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/'+file_tasmin_bc)
                    bc_rcm_tasmin = tasmin_model_bc.variables['tasmin'][:,:,:]-273.15
                    #print('Tasmin netcdf shape: ', bc_rcm_tasmin.shape)
                    tasmin_model_bc.close
                    raw_rcm_tasmin = bc_rcm_tasmin
                    file_hurs_bc = hurs_list[iii]
                    hurs_model_bc = nc.Dataset('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/'+file_hurs_bc)
                    bc_rcm_hurs = hurs_model_bc.variables['hurs'][:,:,:]
                    #print('Hurs netcdf shape: ', bc_rcm_hurs.shape)
                    hurs_model_bc.close
                    raw_rcm_hurs = bc_rcm_hurs
                    file_sund_bc = sund_list[iii]
                    sund_model_bc = nc.Dataset('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/'+file_sund_bc)
                    bc_rcm_sund = sund_model_bc.variables['sund'][:,:,:]
                    sund_model_bc.close
                    raw_rcm_sund = bc_rcm_sund
                    file_sfcWind_bc = sfcWind_list[iii]
                    sfcWind_model_bc = nc.Dataset('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/'+file_sfcWind_bc)
                    bc_rcm_sfcWind = sfcWind_model_bc.variables['sfcWind'][:,:,:]
                    sfcWind_model_bc.close
                    raw_rcm_sfcWind = bc_rcm_sfcWind
                    
                    print ('Reading the correction factors for climate model '+str(1+iii)+'...')
                    #print('===============================')
                    
                    #Next line reads and stored the monthly correction factor for each cell
                    # The correction factor was estimated as model-reference
                    #Thus, the corrected projection is equal to BCmod = RAWmod-correction
                    for cell_index in np.arange (0,36):# This line has to be changed for the french case as it has a 5x5 domain
                        for n_month in np.arange(1,13):
                            correction_ref=np.load('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/'+correction_tas_list[i])
                            correction_factor_tas = correction_ref[cell_index,iii,n_month-1]
                            bc_rcm_tas[:,grid_index.lat_cell[cell_index],grid_index.lon_cell[cell_index]][month_noleap==n_month]= raw_rcm_tas[:,grid_index.lat_cell[cell_index],grid_index.lon_cell[cell_index]][month_noleap==n_month]-correction_factor_tas
                            correction_ref=np.load('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/'+correction_tasmax_list[i])
                            correction_factor_tasmax = correction_ref[cell_index,iii,n_month-1]
                            bc_rcm_tasmax[:,grid_index.lat_cell[cell_index],grid_index.lon_cell[cell_index]][month_noleap==n_month]= raw_rcm_tasmax[:,grid_index.lat_cell[cell_index],grid_index.lon_cell[cell_index]][month_noleap==n_month]-correction_factor_tasmax
                            correction_ref=np.load('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/'+correction_tasmin_list[i])
                            correction_factor_tasmin = correction_ref[cell_index,iii,n_month-1]
                            bc_rcm_tasmin[:,grid_index.lat_cell[cell_index],grid_index.lon_cell[cell_index]][month_noleap==n_month]= raw_rcm_tasmin[:,grid_index.lat_cell[cell_index],grid_index.lon_cell[cell_index]][month_noleap==n_month]-correction_factor_tasmin
                            correction_ref=np.load('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/'+correction_hurs_list[i])
                            correction_factor_hurs = correction_ref[cell_index,iii,n_month-1]
                            bc_rcm_hurs[:,grid_index.lat_cell[cell_index],grid_index.lon_cell[cell_index]][month_noleap==n_month]= raw_rcm_hurs[:,grid_index.lat_cell[cell_index],grid_index.lon_cell[cell_index]][month_noleap==n_month]-correction_factor_hurs
                            correction_ref=np.load('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/'+correction_sund_list[i])
                            correction_factor_sund = correction_ref[cell_index,iii,n_month-1]
                            bc_rcm_sund[:,grid_index.lat_cell[cell_index],grid_index.lon_cell[cell_index]][month_noleap==n_month]= raw_rcm_sund[:,grid_index.lat_cell[cell_index],grid_index.lon_cell[cell_index]][month_noleap==n_month]-correction_factor_sund
                            correction_ref=np.load('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/'+correction_sfcWind_list[i])
                            correction_factor_sfcWind = correction_ref[cell_index,iii,n_month-1]
                            bc_rcm_sfcWind[:,grid_index.lat_cell[cell_index],grid_index.lon_cell[cell_index]][month_noleap==n_month]= raw_rcm_sfcWind[:,grid_index.lat_cell[cell_index],grid_index.lon_cell[cell_index]][month_noleap==n_month]-correction_factor_sfcWind
                            
                    #np.save('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/corrected/tas_bc_delta_ref'+str(i)+'_corr'+str(ii)+'.npy',bc_rcm_tas)
                    bc_rcm_tas.dump('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/corrected/tas_bc_delta_ref'+str(i)+'_corr'+str(iii)+'.npy')
                    annual_mean_tas_site_cor[iii,i]=np.mean(bc_rcm_tas[(year_noleap >= 1976) & (year_noleap <= 2005)])
                    bc_rcm_tasmax.dump('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/corrected/tasmax_bc_delta_ref'+str(i)+'_corr'+str(iii)+'.npy')
                    annual_mean_tasmax_site_cor[iii,i]=np.mean(bc_rcm_tasmax[(year_noleap >= 1976) & (year_noleap <= 2005)])
                    bc_rcm_tasmin.dump('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/corrected/tasmin_bc_delta_ref'+str(i)+'_corr'+str(iii)+'.npy')
                    annual_mean_tasmin_site_cor[iii,i]=np.mean(bc_rcm_tasmin[(year_noleap >= 1976) & (year_noleap <= 2005)])
                    bc_rcm_sfcWind.dump('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/corrected/sfcWind_bc_delta_ref'+str(i)+'_corr'+str(iii)+'.npy')
                    annual_mean_sfcWind_site_cor[iii,i]=np.mean(bc_rcm_sfcWind[(year_noleap >= 1976) & (year_noleap <= 2005)])
                    bc_rcm_hurs.dump('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/corrected/hurs_bc_delta_ref'+str(i)+'_corr'+str(iii)+'.npy')
                    annual_mean_hurs_site_cor[iii,i]=np.mean(bc_rcm_hurs[(year_noleap >= 1976) & (year_noleap <= 2005)])
                    bc_rcm_sund.dump('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/corrected/sund_bc_delta_ref'+str(i)+'_corr'+str(iii)+'.npy')
                    annual_mean_sund_site_cor[iii,i]=np.mean(bc_rcm_sund[(year_noleap >= 1976) & (year_noleap <= 2005)])
                    
                    print('Done')
        
    else:
        print('Processing data for',sites_list[n_site])
        print('===============================')
        print('===============================')
        y_grid=np.arange(0,5)
        x_grid=np.arange(0,5)
        #Imports correction factor for all variables
        correction_tas_list = [filename for filename in os.listdir('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/') if filename.startswith('tas_')]
        correction_tasmax_list = [filename for filename in os.listdir('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/') if filename.startswith('tasmax_')]
        correction_tasmin_list = [filename for filename in os.listdir('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/') if filename.startswith('tasmin_')]
        correction_sfcWind_list = [filename for filename in os.listdir('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/') if filename.startswith('sfcWind_')]
        correction_sund_list = [filename for filename in os.listdir('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/') if filename.startswith('sund_')]
        correction_hurs_list = [filename for filename in os.listdir('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/') if filename.startswith('hurs_')]
        
        # Import list from the RCM, setting up the models
        tas_list = [filename for filename in os.listdir('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/') if filename.startswith('tas_EUR') and filename.endswith('all.nc')]
        tasmax_list = [filename for filename in os.listdir('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/') if filename.startswith('tasmax_EUR') and filename.endswith('all.nc')]
        tasmin_list = [filename for filename in os.listdir('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/') if filename.startswith('tasmin_EUR') and filename.endswith('all.nc')]
        sfcWind_list = [filename for filename in os.listdir('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/') if filename.startswith('sfcWind_EUR') and filename.endswith('all.nc')]
        orog_list = [filename for filename in os.listdir('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/') if filename.startswith('orog_EUR')]
        sund_list = [filename for filename in os.listdir('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/') if filename.startswith('sund_EUR') and filename.endswith('all.nc')]
        hurs_list = [filename for filename in os.listdir('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/') if filename.startswith('hurs_EUR') and filename.endswith('all.nc')]
        
        #loop for the climate models, one as ref and the others are corrected 
        
        annual_mean_tas_site_ref=np.zeros([1,10])#top row is the uncorrected (reference) models and the other rows are the models corrected to the top-row model
        annual_mean_tas_site_cor=np.zeros([10,10])
        annual_mean_tas_site_raw=np.zeros([10,10])
        for i in np.arange(0,10):#Gets the data for all variables of the climate model acting as reference
            print('Using climate model '+str(i+1)+' as reference...')
            print('===============================')
            file_tas = tas_list[i]
            tas_model = nc.Dataset('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/'+file_tas)
            ref_rcm_tas = tas_model.variables['tas'][:,:,:]
            tas_model.close()        
            file_tasmax = tasmax_list[i]
            tasmax_model = nc.Dataset('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/'+file_tasmax)
            rcm_tasmax = tasmax_model.variables['tasmax'][:,:,:]
            file_tasmin = tasmin_list[i]
            tasmin_model = nc.Dataset('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/'+file_tasmin)
            rcm_tasmin = tasmin_model.variables['tasmin'][:,:,:]
            file_sfcWind = sfcWind_list[i]
            sfcWind_model = nc.Dataset('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/'+file_sfcWind)
            rcm_sfcWind = sfcWind_model.variables['sfcWind'][:,:,:]
            file_sund = sund_list[i]
            sund_model = nc.Dataset('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/'+file_sund)
            rcm_sund = sund_model.variables['sund'][:,:,:]
            file_hurs = hurs_list[i]
            hurs_model = nc.Dataset('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/'+file_hurs)
            rcm_hurs = hurs_model.variables['hurs'][:,:,:]
            
            if i<=6:
                annual_mean_tas_site_ref[0,i]=np.mean(ref_rcm_tas[(year >= 1976) & (year <= 2005)])-273.15
            else:
                annual_mean_tas_site_ref[0,i]=np.mean(ref_rcm_tas[(year_noleap >= 1976) & (year_noleap <= 2005)])-273.15
            for ii in np.arange(0,7):#climate model
                if i != ii:
                    file_tas_bc = tas_list[ii]
                    tas_model_bc = nc.Dataset('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/'+file_tas_bc)
                    bc_rcm_tas = tas_model_bc.variables['tas'][:,:,:]-273.15
                    tas_model_bc.close
                    raw_rcm_tas = bc_rcm_tas
                    print ('Reading the correction factors for climate model '+str(1+ii)+'...')
                    #print('===============================')
                    
                    #Next line reads and stored the monthly correction factor for each cell
                    # The correction factor was estimated as model-reference
                    #Thus, the corrected projection is equal to BCmod = RAWmod-correction
                    for cell_index in np.arange (0,25):# This line has to be changed for the french case as it has a 5x5 domain
                        for n_month in np.arange(1,13):
                            correction_ref=np.load('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/'+correction_tas_list[i])
                            correction_factor_tas = correction_ref[cell_index,ii,n_month-1]
                            bc_rcm_tas[:,grid_index_fr.lat_cell[cell_index],grid_index_fr.lon_cell[cell_index]][month==n_month]= raw_rcm_tas[:,grid_index_fr.lat_cell[cell_index],grid_index_fr.lon_cell[cell_index]][month==n_month]-correction_factor_tas
                    #np.save('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/corrected/tas_bc_delta_ref'+str(i)+'_corr'+str(ii)+'.npy',bc_rcm_tas)
                    bc_rcm_tas.dump('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/corrected/tas_bc_delta_ref'+str(i)+'_corr'+str(ii)+'.npy')
                    annual_mean_tas_site_cor[ii,i]=np.mean(bc_rcm_tas[(year >= 1976) & (year <= 2005)])
                    print('Done')
                    
                    
            for iii in np.arange(7,10):#climate model
                if i != iii:
                    file_tas_bc = tas_list[iii]
                    tas_model_bc = nc.Dataset('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/'+file_tas_bc)
                    bc_rcm_tas = tas_model_bc.variables['tas'][:,:,:]-273.15
                    tas_model_bc.close
                    raw_rcm_tas = bc_rcm_tas
                    print ('Reading the correction factors for climate model '+str(1+iii)+'...')
                    #print('===============================')
                    
                    #Next line reads and stored the monthly correction factor for each cell
                    # The correction factor was estimated as model-reference
                    #Thus, the corrected projection is equal to BCmod = RAWmod-correction
                    for cell_index in np.arange (0,25):# This line has to be changed for the french case as it has a 5x5 domain
                        for n_month in np.arange(1,13):
                            correction_ref=np.load('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/'+correction_tas_list[i])
                            correction_factor_tas = correction_ref[cell_index,iii,n_month-1]
                            bc_rcm_tas[:,grid_index_fr.lat_cell[cell_index],grid_index_fr.lon_cell[cell_index]][month_noleap==n_month]= raw_rcm_tas[:,grid_index_fr.lat_cell[cell_index],grid_index_fr.lon_cell[cell_index]][month_noleap==n_month]-correction_factor_tas
                    #np.save('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/corrected/tas_bc_delta_ref'+str(i)+'_corr'+str(ii)+'.npy',bc_rcm_tas)
                    bc_rcm_tas.dump('C:/Users/epz/Desktop/UEF/PET/CMs/'+sites_list[n_site]+'/pet/delta/corrected/tas_bc_delta_ref'+str(i)+'_corr'+str(iii)+'.npy')
                    annual_mean_tas_site_cor[iii,i]=np.mean(bc_rcm_tas[(year_noleap >= 1976) & (year_noleap <= 2005)])
                    print('Done')
            

    # Set the plots to check how it is being corrected
    font = {'family' : 'normal',
    'weight' : 'normal',
    'size'   : 22}
    
    fig, ax = plt.subplots(2,3,figsize=(40,40))
    fontsize=20
    plt.rc('font', **font)
    plt.rcParams['axes.grid'] = True
    colour_plot = ['b','k','r','g','c','y','m','aqua','orange','crimson']
    site_x_axis = np.arange(0,10)
    my_xticks = ['CM1','CM2','CM3','CM4','CM5','CM6','CM7','CM8','CM9','CM10']
    fig.suptitle('Mean annual values for climate variables after delta change correction - ' +sites_list[n_site])
    for nn in np.arange(0,10):
        ax[0,0].scatter(my_xticks[nn],annual_mean_tas_site_ref[0,nn],facecolors=colour_plot[nn],edgecolors=colour_plot[nn], marker='s', s=140)#,label=pet_list[nn],markevery=5)          
        ax[0,1].scatter(my_xticks[nn],annual_mean_tasmax_site_ref[0,nn],facecolors=colour_plot[nn],edgecolors=colour_plot[nn], marker='s', s=140)#,label=pet_list[nn],markevery=5)          
        ax[0,2].scatter(my_xticks[nn],annual_mean_tasmin_site_ref[0,nn],facecolors=colour_plot[nn],edgecolors=colour_plot[nn], marker='s', s=140)#,label=pet_list[nn],markevery=5)          
        ax[1,0].scatter(my_xticks[nn],annual_mean_hurs_site_ref[0,nn],facecolors=colour_plot[nn],edgecolors=colour_plot[nn], marker='s', s=140)#,label=pet_list[nn],markevery=5)          
        ax[1,1].scatter(my_xticks[nn],annual_mean_sund_site_ref[0,nn],facecolors=colour_plot[nn],edgecolors=colour_plot[nn], marker='s', s=140)#,label=pet_list[nn],markevery=5)          
        ax[1,2].scatter(my_xticks[nn],annual_mean_sfcWind_site_ref[0,nn],facecolors=colour_plot[nn],edgecolors=colour_plot[nn], marker='s', s=140)#,label=pet_list[nn],markevery=5)          
        
        for nnn in np.arange(0,10):
            if annual_mean_tas_site_cor[nnn,nn]!=0:
                ax[0,0].scatter(my_xticks[nn],annual_mean_tas_site_cor[nnn,nn],facecolors='none',edgecolors='k', marker='o', s=140)#,label=pet_list[nn],markevery=5)          
                ax[0,0].set(ylabel='Annual mean temperature (Celsius)', xlabel = 'Reference climate model')
                ax[0,1].scatter(my_xticks[nn],annual_mean_tasmax_site_cor[nnn,nn],facecolors='none',edgecolors='k', marker='o', s=140)#,label=pet_list[nn],markevery=5)          
                ax[0,1].set(ylabel='Annual maximum temperature (Celsius)', xlabel = 'Reference climate model')
                ax[0,2].scatter(my_xticks[nn],annual_mean_tasmin_site_cor[nnn,nn],facecolors='none',edgecolors='k', marker='o', s=140)#,label=pet_list[nn],markevery=5)          
                ax[0,2].set(ylabel='Annual minimum temperature (Celsius)', xlabel = 'Reference climate model')
                ax[1,0].scatter(my_xticks[nn],annual_mean_hurs_site_cor[nnn,nn],facecolors='none',edgecolors='k', marker='o', s=140)#,label=pet_list[nn],markevery=5)          
                ax[1,0].set(ylabel='Annual mean near-surface relative humidity (%)', xlabel = 'Reference climate model')
                ax[1,1].scatter(my_xticks[nn],annual_mean_sund_site_cor[nnn,nn],facecolors='none',edgecolors='k', marker='o', s=140)#,label=pet_list[nn],markevery=5)          
                ax[1,1].set(ylabel='Annual mean duration of sushine (seconds)', xlabel = 'Reference climate model')
                ax[1,2].scatter(my_xticks[nn],annual_mean_sfcWind_site_cor[nnn,nn],facecolors='none',edgecolors='k', marker='o', s=140)#,label=pet_list[nn],markevery=5)          
                ax[1,2].set(ylabel='Annual mean surface wind speed (m/s)', xlabel = 'Reference climate model')
    plt.savefig('C:/Users/epz/Desktop/UEF/PET/CMs/figures/delta/'+sites_list[n_site]+'_all_corrected_ref_period.png')