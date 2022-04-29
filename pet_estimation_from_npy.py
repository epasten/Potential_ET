"""PET ESTIMATION FROM NPY FILES, SUCH AS THE BIAS CORRECTED FILES

This script is run only for one site once, thus change before runnning always 
change the previous site, eg '/fi/', with the new one, eg '/uk/'.

Definition of the variables

Climate model inputs
tas = mean temperature (C)
tasmax = maximum temperature (C)
tasmin = minimum temperature (C)
sfcWind = mean surface wind speed (m/s)
z = altitude (meters)
lat_dd = latitude (decimal degrees)
RH = ralive humidity (%)
julian = julian day
n = bright sunshine hours

Constants
den_wat = water density (Kg/L)
Cp = Air specific heat capacity (MJ kg-1 C-1)
E = ratio of the molecular weigth of water vapor to dry air 
lat_heat = latent heat of vaporization 
alpha = a Priestley-Taylor coefficient 
As = regression constant for Ra, recommended value of 0.25 (Allen et al. 1998)
Bs = constant used to estimate the solar radiation, recommended value of 0.5 (Allen et al. 1998)


Common estimations
patm = athmospheric pressure 
N = maximum possible daylight length (hours) (Allen et al. 1998)

sol_decl = solar declination (radians; Allen et al. 1998) 
ws = sunset hour angle (radians) 
dr = inverse relative distance Earth-Sun (Allen et al. 1998)
psyco = Psychometric constant (KPa C-1)

es = saturation vapor pressute (kPa)
slope = slope of the relation between saturation vapor pressure and air temperature (kPa C-1)
G = soil heat flux (MJ m-2 day-1) => small at daily scales, thus can be ignored
ea = actual vapor pressure (kPa)
es-ea = saturation vapor pressure deficit (kPa)
eT =  saturation vapor pressure at temperature T (kPa)
Rs = solar radiation (MJ m2 d-1) (Allen et al. 1998)
Ra = extraterrestrial radiation for daily inputs (MJ m-2 d-1; Allen et al. 1998)
Rn = Net radiation over canopy (MJ m-2 day-1; Allen et al. 1998)
Rso = clear-sky solar radiation '(MJ m-2 d-1; Allen et al. 1998)
Rns = Net shortwave radiation (MJ m-2 d-1; Allen et al. 1998)
alb = reference grass albeldo (Allen et al. 1998)
Rnl = Net longwave radiation (MJ m-2 d-1; Allen et al. 1998)


    
Pet equations

Penman Equation (as seen in Zheng et al. 2017, from Penman (1948)

Priestley-Taylor (as seen in Zheng et al. 2017, from Priestley-Taylor 1972

Hargreaves (as seen in Zheng et al. 2017, from Hargeaves 1975; Xu and Singh, 2000)

Hamon (as seen in Zheng et al. 2017; Seiller and Anctil, 2016;  Prudhomme and Williamson, 2013, Ousin et al. 2005)

Oudin (from Oudin et al. 2005)

Hargreaves and Samani (as seen in Xu and Singh 2002)

Makkink (as seen in zheng et al 2017; Xu and Singh 2002)

Turc (as seen in Prudhomme and Williamson, 2013, from Turc, 1961)

Jensen Haise (as seen in Rosenbery et al. 2004)

Penman Monteith (As seen in Guo et bal. 2017 and Chang et al. 2016)
"""
import pandas as pd
import numpy as np
import netCDF4 as nc 
import math
import datetime
import os
import matplotlib as plt

dates = pd.read_excel('C:/Users/epz/Desktop/UEF/PET/Python/dates1971.xlsx')
month = dates.Month
year = dates.Year
julian_days = dates. Julian
julian_days = np.array(julian_days)
dates_no_leap = pd.read_excel('C:/Users/epz/Desktop/UEF/PET/Python/dates1971_noleap.xlsx')
orog_order =pd.read_excel('C:/Users/epz/Desktop/UEF/PET/Python/orog_model_read.xlsx')
orog_i = orog_order.order
orog_model = orog_order.orog
month_noleap = dates_no_leap.Month 
year_noleap = dates_no_leap.Year
julian_days_no_leap = dates_no_leap.Julian
y_grid = np.arange(0,6)
x_grid = np.arange(0,6)
julian_days_n = np.arange(0,len(julian_days))

# Import data from the RCM, setting up the models
tas_list = [filename for filename in os.listdir('C:/Users/epz/Desktop/UEF/PET/CMs/fi/pet/delta/corrected/') if filename.startswith('tas_') and filename.endswith('.npy')]
tasmax_list = [filename for filename in os.listdir('C:/Users/epz/Desktop/UEF/PET/CMs/fi/pet/delta/corrected/') if filename.startswith('tasmax_') and filename.endswith('.npy')]
tasmin_list = [filename for filename in os.listdir('C:/Users/epz/Desktop/UEF/PET/CMs/fi/pet/delta/corrected/') if filename.startswith('tasmin_') and filename.endswith('.npy')]
sfcWnd_list = [filename for filename in os.listdir('C:/Users/epz/Desktop/UEF/PET/CMs/fi/pet/delta/corrected/') if filename.startswith('sfcWind_') and filename.endswith('.npy')]
orog_list = [filename for filename in os.listdir('C:/Users/epz/Desktop/UEF/PET/CMs/fi/') if filename.startswith('orog_EUR')]
sund_list = [filename for filename in os.listdir('C:/Users/epz/Desktop/UEF/PET/CMs/fi/pet/delta/corrected/') if filename.startswith('sund_') and filename.endswith('.npy')]
hurs_list = [filename for filename in os.listdir('C:/Users/epz/Desktop/UEF/PET/CMs/fi/pet/delta/corrected/') if filename.startswith('hurs_') and filename.endswith('.npy')]

print('tas models:',len(tas_list))
print('tasmax models:',len(tasmax_list))
print('tasmin models:',len(tasmin_list))
print('sfcWnd models:',len(sfcWnd_list))
print('orog models:',len(orog_list))
print('sund models:',len(sund_list))
print('hurs models:',len(hurs_list))


n_models = np.arange(9,90)
#n_models = [0]#######################################################CHANGE THIS
for i in n_models:
    pet_penman=np.zeros([len(julian_days_n),6,6])
    pet_pt = np.zeros([len(julian_days_n),6,6])
    pet_har = np.zeros([len(julian_days_n),6,6])
    pet_ham = np.zeros([len(julian_days_n),6,6])
    pet_harsam = np.zeros([len(julian_days_n),6,6])
    pet_mak = np.zeros([len(julian_days_n),6,6])
    pet_jh = np.zeros([len(julian_days_n),6,6])
    pet_turc = np.zeros([len(julian_days_n),6,6])
    pet_bla_cri = np.zeros([len(julian_days_n),6,6])
    pet_oudin = np.zeros([len(julian_days_n),6,6])
    pet_penman_mon = np.zeros([len(julian_days_n),6,6])

    xx = orog_model[i]
    print(tas_list[i])
    print(tasmax_list[i])
    print(tasmin_list[i])
    print(sfcWnd_list[i])
    print(orog_list[xx])
    print(sund_list[i])
    print(hurs_list[i])
    model_name = tas_list[i]
    model_name = model_name[3:]
    file_tas =  tas_list[i]
    file_tasmin = tasmin_list[i]
    file_tasmax = tasmax_list[i]
    file_hurs = hurs_list[i]
    file_sund = sund_list[i]
    file_sfcWind = sfcWnd_list[i]
    
    rcm_tas = np.load('C:/Users/epz/Desktop/UEF/PET/CMs/fi/pet/delta/corrected/'+file_tas,allow_pickle=True)
    len_tas = len(rcm_tas)
    if len_tas == 47482:
        
        len_data = np.arange(0,len_tas)
        file_orog = orog_list[xx]
        rcm_orog_file = nc.Dataset('C:/Users/epz/Desktop/UEF/PET/CMs/fi/'+file_orog)
        rcm_orog = rcm_orog_file.variables['orog'][:,:]
        lat_rcm = rcm_orog_file.variables['lat'][:]
        
        rcm_tasmin = np.load('C:/Users/epz/Desktop/UEF/PET/CMs/fi/pet/delta/corrected/'+file_tasmin,allow_pickle=True)
        rcm_tasmax = np.load('C:/Users/epz/Desktop/UEF/PET/CMs/fi/pet/delta/corrected/'+file_tasmax,allow_pickle=True)
        rcm_tas=rcm_tas+273.15
        rcm_tasmax=rcm_tasmax+273.15
        rcm_tasmin=rcm_tasmin+273.15
        rcm_hurs = np.load('C:/Users/epz/Desktop/UEF/PET/CMs/fi/pet/delta/corrected/'+file_hurs,allow_pickle=True)
        rcm_sund = np.load('C:/Users/epz/Desktop/UEF/PET/CMs/fi/pet/delta/corrected/'+file_sund,allow_pickle=True)
        rcm_sfcWind = np.load('C:/Users/epz/Desktop/UEF/PET/CMs/fi/pet/delta/corrected/'+file_sfcWind,allow_pickle=True)
        
        #### FOR EACH GRIDCELL IN THE RCM SUBDOMAIN
        for ii in [0,1,2,3,4]:#,5]:
            for iii in [0,1,2,3,4,5]:
                print(ii,',',iii)
                lat_dd = lat_rcm[ii,iii]
                lat_rad = lat_dd * math.pi/180
                z = rcm_orog[ii,iii]
                
                
                
                
                for nn in julian_days_n:
                    
                    julian=julian_days[nn]
                    ### EXTRACTING THE DAILY CLIMATE DATA
                    print('Processing day ',nn+1,' out of ', len(julian_days),' for gridcell [',ii,',',iii,'] for model', i+1, 'out of 90')
                    RH = rcm_hurs[nn,ii,iii]
                    #print('RH:',RH)
                    tas =  rcm_tas[nn,ii,iii]-273.15
                    #print('tas:',tas)
                    tasmin = rcm_tasmin[nn,ii,iii]-273.15
                    #print('tasmin:',tasmin)
                    tasmax = rcm_tasmax[nn,ii,iii]-273.15
                    #print('tasmax:',tasmax)
                    sfcWind_pre = rcm_sfcWind[nn,ii,iii]
                    ##Wind correction
                    sfcWind = sfcWind_pre * 4.87 / math.log(678.7-5.42)
                    #print('wind speed:',sfcWind)
                    n_sun = rcm_sund[nn,ii,iii]/3600#####CHECK IF UNITS ARE OK FOR THIS ONE !!!!!!!!!!NPO
                    #print('sund:',n_sun)
                    
                    # Constants 
                    den_wat = 1000
                    Cp = 1.013*10**(-3)
                    E = 0.622
                    lat_heat = 2.45
                    alpha = 1.26
                    As = 0.25
                    Bs = 0.5
                    alb = 0.23 # from Guo et al 2017
                    stef_boltz = 4.903*10**-9
                    rs = 69 #m/s, from Oudin et al 2015
    
                    # Common estimations
                    patm = 101.3*((293-0.0065*z)/293)**5.26
                    sol_decl = 0.409*math.sin(((2*math.pi*julian)/365)-1.39)
                    ws = math.acos(-math.tan(lat_rad)*math.tan(sol_decl))
                    dr = 1 + 0.033*math.cos((math.pi*2*julian)/365)
                    psyco = (Cp*patm)/(E*lat_heat)
                    N = 24*ws/math.pi 
                    slope = 4098*(0.6108*math.exp((17.27*tas)/(tas+237.3)))/((tas+237.3)**2)
                    eTmin = 0.6108*math.exp((17.27*tasmin/(237.3+tasmin)))
                    eTmax = 0.6108*math.exp((17.27*tasmax/(237.3+tasmax)))
                    es = (eTmax + eTmin)/2
                    ea = (RH/100)*((eTmax+eTmin)/2)
                    Ra = ((24*60)/math.pi)*0.082*dr*(ws*math.sin(lat_rad)*math.sin(sol_decl)+math.cos(lat_rad)*math.cos(sol_decl)*math.sin(ws))
                    Rs = Ra *(As+Bs*(n_sun/N))
                    Rso = (0.75+2*10**(-5)*z)*Ra
                    Rns = (1-alb) * Rs
                    Rnl =  ((((stef_boltz *(tasmax+273.16)**4))+(stef_boltz *(tasmin+273.16)**4))/2)*(0.34-0.14*math.sqrt(ea))*(1.35*(Rs/Rso)-0.35)
                    Rn = Rns-Rnl               
            
    # PET Equations 
    
    #Penman
                    
                    pet_penman[nn,ii,iii] = (slope/(slope+psyco))*(Rn/lat_heat)+(psyco/(slope+psyco))*((6.43*(1+0.536*sfcWind)*(es-ea))/lat_heat)
                    
    
    # Priestley-Taylor
                    
                    pet_pt[nn,ii,iii] = alpha*(slope/(slope+psyco))*(Rn/lat_heat)                
    
    
    #Hargreaves
                    
                    pet_har[nn,ii,iii] = 0.0135*(tas+17.8)*Rs/lat_heat
    
    # Hamon
                    
                    pet_ham[nn,ii,iii] = ((N/12)**2)*math.exp(tas/16)
    
    # Hargreaves and Samani
                    
                    pet_harsam[nn,ii,iii] = 0.0023*math.sqrt(abs(tasmax-tasmin))*0.408*Ra*(tas+17.8) #mm/day
    
    # Makkink
                    
                    pet_mak[nn,ii,iii] = (0.61*(slope/(slope+psyco))*(Rs/lat_heat))-0.12
    
    # Jensen Haise
                    pet_jh[nn,ii,iii] = 25.4 * (0.016*tas + 0.186)*(0.000673*(Rs/0.041868))
                    #pet_jh[nn,ii,iii] = (0.025*(tas+3)*Ra)/lat_heat
            
    # Turc
                    if tas>-15:
                    
                        if RH < 50:
                            pet_turc[nn,ii,iii] = 0.31*(tas/(tas+15))*(Rns+2.09)*(1+(50-RH)/70)
                        else:
                            pet_turc[nn,ii,iii] =  0.31*(tas/(tas+15))*(Rns+2.09)
                    else:
                        pet_turc[nn,ii,iii] =  0
    # Oudin
                    if tas>-5:
                        pet_oudin[nn,ii,iii] = 1000*(Ra/(den_wat*lat_heat))*((tas+5)/100)
                    else:
                        pet_oudin[nn,ii,iii] = 0
    
    
    
    # Penman-Monteith
                    
                    #FAO56 Penman-Monteith from Liu et al. 2017
                    pet_penman_mon[nn,ii,iii] = (0.408*slope*Rn+psyco*(900/(tas+273))*sfcWind*(es-ea))/(slope+psyco*(1+0.34*sfcWind))
                    #pet_penman_mon[nn,ii,iii] = (slope*Rn+1013*(ea-es)/(208/sfcWind))/(lat_heat*1000*(slope+psyco*(1+(rs/(208/sfcWind)))))
                    
        pet_penman[pet_penman<0]=0
        pet_pt[pet_pt<0]=0
        pet_har[pet_har<0]=0
        pet_ham[pet_ham<0]=0
        pet_harsam[pet_harsam<0]=0
        pet_mak[pet_mak<0]=0
        pet_jh[pet_jh<0]=0
        pet_turc[pet_turc<0]=0
        pet_penman_mon[pet_penman_mon<0]=0
        
        
        np.save('C:/Users/epz/Desktop/UEF/PET/CMs/fi/pet/delta/corrected/pet/delta_penman'+model_name+'.npy',pet_penman)
        np.save('C:/Users/epz/Desktop/UEF/PET/CMs/fi/pet/delta/corrected/pet/delta_priest_taylor'+model_name+'.npy',pet_pt)
        np.save('C:/Users/epz/Desktop/UEF/PET/CMs/fi/pet/delta/corrected/pet/delta_hargreaves'+model_name+'.npy',pet_har)
        np.save('C:/Users/epz/Desktop/UEF/PET/CMs/fi/pet/delta/corrected/pet/delta_hamon'+model_name+'.npy',pet_ham)
        np.save('C:/Users/epz/Desktop/UEF/PET/CMs/fi/pet/delta/corrected/pet/delta_har_sam'+model_name+'.npy',pet_harsam)
        np.save('C:/Users/epz/Desktop/UEF/PET/CMs/fi/pet/delta/corrected/pet/delta_makkink'+model_name+'.npy',pet_mak)
        np.save('C:/Users/epz/Desktop/UEF/PET/CMs/fi/pet/delta/corrected/pet/delta_jensen_haise'+model_name+'.npy',pet_jh)
        np.save('C:/Users/epz/Desktop/UEF/PET/CMs/fi/pet/delta/corrected/pet/delta_turc'+model_name+'.npy',pet_turc)
        #np.save('C:/Users/epz/Desktop/UEF/PET/CMs/fi/pet/delta/delta_blanney_criddle'+model_name+'.npy',pet_bla_cri)
        np.save('C:/Users/epz/Desktop/UEF/PET/CMs/fi/pet/delta/corrected/pet/delta_oudin'+model_name+'.npy',pet_oudin)
        np.save('C:/Users/epz/Desktop/UEF/PET/CMs/fi/pet/delta/corrected/pet/delta_penman_mon'+model_name+'.npy',pet_penman_mon)
        
        
    else:
            
        
        julian_days_noleap_n = np.arange(0,len(julian_days_no_leap))
        #Output arrays:
        pet_penman=np.zeros([len(julian_days_no_leap),6,6])
        pet_pt = np.zeros([len(julian_days_no_leap),6,6])
        pet_har = np.zeros([len(julian_days_no_leap),6,6])
        pet_ham = np.zeros([len(julian_days_no_leap),6,6])
        pet_harsam = np.zeros([len(julian_days_no_leap),6,6])
        pet_mak = np.zeros([len(julian_days_no_leap),6,6])
        pet_jh = np.zeros([len(julian_days_no_leap),6,6])
        pet_turc = np.zeros([len(julian_days_no_leap),6,6])
        pet_bla_cri = np.zeros([len(julian_days_no_leap),6,6])
        pet_oudin = np.zeros([len(julian_days_no_leap),6,6])
        pet_penman_mon = np.zeros([len(julian_days_no_leap),6,6])
        
        
        print(tas_list[i])
        print(tasmax_list[i])
        print(tasmin_list[i])
        print(sfcWnd_list[i])
        print(orog_list[xx])
        print(sund_list[i])
        print(hurs_list[i])
        model_name = tas_list[i]
        model_name = model_name[3:]
        file_tas =  tas_list[i]
        len_data = np.arange(0,len_tas)
        file_orog = orog_list[xx]
        rcm_orog_file = nc.Dataset('C:/Users/epz/Desktop/UEF/PET/CMs/fi/'+file_orog)
        rcm_orog = rcm_orog_file.variables['orog'][:,:]
        lat_rcm = rcm_orog_file.variables['lat'][:]
        
        rcm_tasmin = np.load('C:/Users/epz/Desktop/UEF/PET/CMs/fi/pet/delta/corrected/'+file_tasmin,allow_pickle=True)
        rcm_tasmax = np.load('C:/Users/epz/Desktop/UEF/PET/CMs/fi/pet/delta/corrected/'+file_tasmax,allow_pickle=True)
        rcm_tas=rcm_tas+273.15
        rcm_tasmax=rcm_tasmax+273.15
        rcm_tasmin=rcm_tasmin+273.15
        rcm_hurs = np.load('C:/Users/epz/Desktop/UEF/PET/CMs/fi/pet/delta/corrected/'+file_hurs,allow_pickle=True)
        rcm_sund = np.load('C:/Users/epz/Desktop/UEF/PET/CMs/fi/pet/delta/corrected/'+file_sund,allow_pickle=True)
        rcm_sfcWind = np.load('C:/Users/epz/Desktop/UEF/PET/CMs/fi/pet/delta/corrected/'+file_sfcWind,allow_pickle=True)
        
        #### FOR EACH GRIDCELL IN THE RCM SUBDOMAIN
        for ii in [0,1,2,3,4]:#,5]:
            for iii in [0,1,2,3,4,5]:
                print(ii,',',iii)
                lat_dd = lat_rcm[ii,iii]
                lat_rad = lat_dd * math.pi/180
                z = rcm_orog[ii,iii]
                
                for nn in julian_days_noleap_n:
                    julian=julian_days_no_leap[nn]
                    ### EXTRACTING THE DAILY CLIMATE DATA
                    print('Processing day ',nn+1,' out of ', len(julian_days_no_leap),' for gridcell [',ii,',',iii,'] for model', i+1, 'out of 90')
                    RH = rcm_hurs[nn,ii,iii]
                    #print('RH:',RH)
                    tas =  rcm_tas[nn,ii,iii]-273.15
                    #print('tas:',tas)
                    tasmin = rcm_tasmin[nn,ii,iii]-273.15
                    #print('tasmin:',tasmin)
                    tasmax = rcm_tasmax[nn,ii,iii]-273.15
                    #print('tasmax:',tasmax)
                    sfcWind_pre = rcm_sfcWind[nn,ii,iii]
                    ##Wind correction
                    sfcWind = sfcWind_pre * 4.87 / math.log(678.7-5.42)
                    #print('wind speed:',sfcWind)
                    n_sun = rcm_sund[nn,ii,iii]/3600#changing from seconds to hours
                    #print('sund:',n_sun)
                    
                    # Constants 
                    den_wat = 1000
                    Cp = 1.013*10**(-3)
                    E = 0.622
                    lat_heat = 2.45
                    alpha = 1.26
                    As = 0.25
                    Bs = 0.5
                    alb = 0.23 # from Guo et al 2017
                    stef_boltz = 4.903*10**-9
                    rs = 69 #m/s, from Oudin et al 2015
    
                    # Common estimations
                    patm = 101.3*((293-0.0065*z)/293)**5.26
                    sol_decl = 0.409*math.sin(((2*math.pi*julian)/365)-1.39)
                    ws = math.acos(-math.tan(lat_rad)*math.tan(sol_decl))
                    dr = 1 + 0.033*math.cos((math.pi*2*julian)/365)
                    psyco = (Cp*patm)/(E*lat_heat)
                    N = 24*ws/math.pi 
                    slope = 4098*(0.6108*math.exp((17.27*tas)/(tas+237.3)))/((tas+237.3)**2)
                    eTmin = 0.6108*math.exp((17.27*tasmin/(237.3+tasmin)))
                    eTmax = 0.6108*math.exp((17.27*tasmax/(237.3+tasmax)))
                    es = (eTmax + eTmin)/2
                    ea = (RH/100)*((eTmax+eTmin)/2)
                    Ra = ((24*60)/math.pi)*0.082*dr*(ws*math.sin(lat_rad)*math.sin(sol_decl)+math.cos(lat_rad)*math.cos(sol_decl)*math.sin(ws))
                    Rs = Ra *(As+Bs*(n_sun/N))
                    Rso = (0.75+2*10**(-5)*z)*Ra
                    Rns = (1-alb) * Rs
                    Rnl =  ((((stef_boltz *(tasmax+273.16)**4))+(stef_boltz *(tasmin+273.16)**4))/2)*(0.34-0.14*math.sqrt(ea))*(1.35*(Rs/Rso)-0.35)
                    Rn = Rns-Rnl
                    
     
    # PET Equations 
    
    #1) Penman
                   
                    pet_penman[nn,ii,iii] = (slope/(slope+psyco))*(Rn/lat_heat)+(psyco/(slope+psyco))*((6.43*(1+0.536*sfcWind)*(es-ea))/lat_heat)
                    
    
    #2) Priestley-Taylor
                  
                    pet_pt[nn,ii,iii] = alpha*(slope/(slope+psyco))*(Rn/lat_heat)                
    
    
    #3) Hargreaves
                   
                    pet_har[nn,ii,iii] = 0.0135*(tas+17.8)*Rs/lat_heat
    
    #4) Hamon
                    
                    pet_ham[nn,ii,iii] = ((N/12)**2)*math.exp(tas/16)
    
    #5) Hargreaves and Samani
                   
                    pet_harsam[nn,ii,iii] = 0.0023*math.sqrt(abs(tasmax-tasmin))*0.408*Ra*(tas+17.8) #mm/day
    
    #6) Makkink
                    
                    pet_mak[nn,ii,iii] = (0.61*(slope/(slope+psyco))*(Rs/lat_heat))-0.12
    
    #7) Jensen Haise
                    pet_jh[nn,ii,iii] = 25.4 * (0.016*tas + 0.186)*(0.000673*(Rs/0.041868))
                    #pet_jh[nn,ii,iii] = (0.025*(tas+3)*Ra)/lat_heat
    
            
    #8) Turc
                    if tas>-15:
                    
                        if RH < 50:
                            pet_turc[nn,ii,iii] = 0.31*(tas/(tas+15))*(Rns+2.09)*(1+(50-RH)/70)
                        else:
                            pet_turc[nn,ii,iii] =  0.31*(tas/(tas+15))*(Rns+2.09)
                    else:
                        pet_turc[nn,ii,iii] =  0
    #9) Oudin
                    
                    if tas>-5:
                        pet_oudin[nn,ii,iii] = 1000*(Ra/(den_wat*lat_heat))*((tas+5)/100)
                    else:
                        pet_oudin[nn,ii,iii] = 0
    
    
    
    #10) Penman-Monteith
                    #FAO56 Penman-Monteith from Liu et al. 2017
                    pet_penman_mon[nn,ii,iii] = (0.408*slope*Rn+psyco*(900/(tas+273))*sfcWind*(es-ea))/(slope+psyco*(1+0.34*sfcWind))
                    #pet_penman_mon[nn,ii,iii] = (slope*Rn+1013*(ea-es)/(208/sfcWind))/(lat_heat*1000*(slope+psyco*(1+(rs/(208/sfcWind)))))
                    
        pet_penman[pet_penman<0]=0
        pet_pt[pet_pt<0]=0
        pet_har[pet_har<0]=0
        pet_ham[pet_ham<0]=0
        pet_harsam[pet_harsam<0]=0
        pet_mak[pet_mak<0]=0
        pet_jh[pet_jh<0]=0
        pet_turc[pet_turc<0]=0
        pet_penman_mon[pet_penman_mon<0]=0
        np.save('C:/Users/epz/Desktop/UEF/PET/CMs/fi/pet/delta/corrected/pet/delta_penman'+model_name+'.npy',pet_penman)
        np.save('C:/Users/epz/Desktop/UEF/PET/CMs/fi/pet/delta/corrected/pet/delta_priest_taylor'+model_name+'.npy',pet_pt)
        np.save('C:/Users/epz/Desktop/UEF/PET/CMs/fi/pet/delta/corrected/pet/delta_hargreaves'+model_name+'.npy',pet_har)
        np.save('C:/Users/epz/Desktop/UEF/PET/CMs/fi/pet/delta/corrected/pet/delta_hamon'+model_name+'.npy',pet_ham)
        np.save('C:/Users/epz/Desktop/UEF/PET/CMs/fi/pet/delta/corrected/pet/delta_har_sam'+model_name+'.npy',pet_harsam)
        np.save('C:/Users/epz/Desktop/UEF/PET/CMs/fi/pet/delta/corrected/pet/delta_makkink'+model_name+'.npy',pet_mak)
        np.save('C:/Users/epz/Desktop/UEF/PET/CMs/fi/pet/delta/corrected/pet/delta_jensen_haise'+model_name+'.npy',pet_jh)
        np.save('C:/Users/epz/Desktop/UEF/PET/CMs/fi/pet/delta/corrected/pet/delta_turc'+model_name+'.npy',pet_turc)
        #np.save('C:/Users/epz/Desktop/UEF/PET/CMs/fi/pet/delta/corrected/pet/delta_blanney_criddle'+model_name+'.npy',pet_bla_cri)
        np.save('C:/Users/epz/Desktop/UEF/PET/CMs/fi/pet/delta/corrected/pet/delta_oudin'+model_name+'.npy',pet_oudin)
        np.save('C:/Users/epz/Desktop/UEF/PET/CMs/fi/pet/delta/corrected/pet/delta_penman_mon'+model_name+'.npy',pet_penman_mon)