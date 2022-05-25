import numpy as np
from scipy import optimize
from sympy.solvers import solve
from scipy.optimize import fsolve
from scipy.optimize import root
from sympy import Symbol
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import newton
from scipy.optimize import bisect
import six 
from six.moves import zip
import xlwt
import csv
from numpy import genfromtxt
from matplotlib.ticker import FuncFormatter
import pandas as pd
from scipy.optimize import minimize
# ========================================== parameters ========================================== #
global coeffTmax
global tsN, tuN, N, Aa, Am, Da, Dm, Dr, Ar
global scale_um, scale_uf, scale_sm, scale_sf
# ========================================== parameters ========================================== #
Tstart = 2000
Tend = 2100
Tstep = 20
T = int((Tend - Tstart)/Tstep + 1)                  # Time horizonreg
nrcp = 5                                            # number of RCP scenarios + 1 scenario for no-climate chaneg
theta = 0.55/2                                      # Share of agri good in consumption function
#za                                                 # Share of male labor in agri production function
#zb                                                 # Share of male labor in non-agri production function
eps = 0.5                                           # Elasticity of substitution between agri and non-agri goods in consumption function
eta = 2.9                                           # Elasticity of substitution between male and female labor in production function
coeffTmax = [12.4, 0.8798]                          # Coefficients of the linear approximation of Tmax from Tmean

# ========================================== Damages =========================================== #
# D = g0 + g1 * T + g2 * T^2

# Agricultural parameters
g0a = -2.24
g1a = 0.308
g2a = -0.0073

# Manufacturing parameters
g0b = 0.3
g1b = 0.08
g2b = -0.0023
# ========================================== Variables =========================================== #                    

# == temperature == #
Temp = np.zeros((T, nrcp))                     # Mean Temperature
MaxTemp = np.zeros((T, nrcp))                  # Max Temperature

# == child-rearing time == #
gamma = 0.4                                    # Share of children's welbeing in Utility function of parents in 1980

# == Age matrix == #
nu = np.zeros((T, nrcp))                       # number of low-skilled children
num = np.zeros((T, nrcp))                      # number of male low-skilled children
nuf = np.zeros((T, nrcp))                      # number of female low-skilled children

ns = np.zeros((T, nrcp))                       # number of high-skilled children
nsm = np.zeros((T, nrcp))                      # number of male high-skilled children
nsf = np.zeros((T, nrcp))                      # number of female high-skilled children

L = np.zeros((T, nrcp))                        # Number of low-skilled parents
Lm = np.zeros((T, nrcp))                       # Number of male low-skilled parents
Lf = np.zeros((T, nrcp))                       # Number of female low-skilled parents

Ma = np.zeros((T, nrcp))
Mb = np.zeros((T, nrcp))

H = np.zeros((T, nrcp))                        # Number of high-skilled parents
Hm = np.zeros((T, nrcp))                       # Number of male high-skilled parents
Hf = np.zeros((T, nrcp))                       # Number of female high-skilled parents

h = np.zeros((T, nrcp))                        # Ratio of high-skilled to low-skilled labor h=H/L
hm = np.zeros((T, nrcp))                       # Ratio of male high-skilled to low-skilled labor h=H/L
hf = np.zeros((T, nrcp))                       # Ratio of male high-skilled to low-skilled labor h=H/L

hn = np.zeros((T, nrcp))                       # approximate ratio of high-skilled to low-skilled children h=ns/nu

N = np.zeros((T, nrcp))                        # Adult population
Nm = np.zeros((T, nrcp))                       # male adult population
Nf = np.zeros((T, nrcp))                       # female adult population

Pop = np.zeros((T, nrcp))                      # total population
Pgr = np.zeros((T, nrcp))                      # population growth rate

# == Prices == #
pa = np.zeros((T, nrcp))                       # Pice of AgricuLtural good
pb = np.zeros((T, nrcp))                       # Pice of Manufacturing good
pr = np.zeros((T, nrcp))                       # Relative pice of Manufacturing to Agricultural goods

# == Cliamte Damages == #
Da = np.zeros((T, nrcp))                       # AgricuLtural damage
Db = np.zeros((T, nrcp))                       # Manufacturing damage
Dr = np.zeros((T, nrcp))                       # Ratio of Manufacturing damages to Agricultural damages
 
# == Wages == #
wum = np.zeros((T, nrcp))                      # Wage of male low-skilled labor
wuf = np.zeros((T, nrcp))                      # Wage of female low-skilled labor

wsm = np.zeros((T, nrcp))                      # Wage of male high-skilled labor
wsf = np.zeros((T, nrcp))                      # Wage of female high-skilled labor

wrm = np.zeros((T, nrcp))                      # Wage ratio of male high-skilled to low-skilled labor
wrf = np.zeros((T, nrcp))                      # Wage ratio of female high-skilled to low-skilled labor

wgaps = np.zeros((T, nrcp))                    # Wage ratio of high-skilled male to female
wgapu = np.zeros((T, nrcp))                    # Wage ratio of low-skilled male to female

# == Technology == #
Aa = np.zeros((T, nrcp))                       # Technological growth function for Agriculture
Ab = np.zeros((T, nrcp))                       # Technological growth function for Manufacurng
Ar = np.zeros((T, nrcp))                       # ratio of Technology in Manufacurng to Agriculture
Aag = nrcp * [0]                               # growth rate of Agricultural productivity
Abg = nrcp * [0]                               # growth rate of Manufacturing productivity
Abgr = 0.01                                    # annual growth rate of Manufacturing productivity
 
# == Output == #
Y = np.zeros((T, nrcp))                        # Total output
Ya = np.zeros((T, nrcp))                       # AgricuLtural output
Yb = np.zeros((T, nrcp))                       # Manufacturing output
Yr = np.zeros((T, nrcp))                       # Ratio of Manufacturing output to Agricultural output
Yp = np.zeros((T, nrcp))                       # Output per capita

# == Availability == #
Suf = 1 + np.zeros((T, nrcp))                  # Availability of female low-skilled labor
Ssf = 1 + np.zeros((T, nrcp))                  # Availability of female high-skilled labor
Sum = 1 + np.zeros((T, nrcp))                  # Availability of male low-skilled labor
Ssm = 1 + np.zeros((T, nrcp))                  # Availability of male high-skilled labor

# == Consumption == #
caum = np.zeros((T, nrcp))                     # consumption of agricultural good male low-skilled
cauf = np.zeros((T, nrcp))                     # consumption of agricultural good female low-skilled

casm = np.zeros((T, nrcp))                     # consumption of agricultural good male high-skilled
casf = np.zeros((T, nrcp))                     # consumption of agricultural good female high-skilled

cbum = np.zeros((T, nrcp))                     # consumption of manufacturing good male low-skilled
cbuf = np.zeros((T, nrcp))                     # consumption of manufacturing good female low-skilled

cbsm = np.zeros((T, nrcp))                     # consumption of manufacturing good male high-skilled
cbsf = np.zeros((T, nrcp))                     # consumption of manufacturing good female high-skilled

cum = np.zeros((T, nrcp))                      # consumption of all goods male low-skilled
cuf = np.zeros((T, nrcp))                      # consumption of all goods female low-skilled

csm = np.zeros((T, nrcp))                      # consumption of all goods male high-skilled
csf = np.zeros((T, nrcp))                      # consumption of all goods female high-skilled

# ============================================== Country Calibration ============================================== #
Con_data = genfromtxt('Input/climate_ssp.csv', dtype=[('SSP', "|S10"), ('Model', "|S10"), ('Scenario', "|S10"), ('Region', "|S10"), ('Education', "|S10"), ('Unit', "|S10"), ('2000', float), ('2005', float), ('2010', float), ('2015', float), ('2020', float), ('2025', float), ('2030', float), ('2035', float), ('2040', float), ('2045', float), ('2050', float), ('2055', float), ('2060', float), ('2065', float), ('2070', float), ('2075', float), ('2080', float), ('2085', float), ('2090', float), ('2095', float), ('2100', float)], skip_header =1, delimiter=',')
Pro_data = pd.read_csv('Input/productivity.csv')
Pop_data = pd.read_csv('Input/pop.csv') #, header=None
Gdp_data = pd.read_csv('Input/gdp.csv') #, header=None
Temp_data = pd.read_csv('Input/Temp-SA.csv')


RCPName = ['RCP26', 'RCP45', 'RCP60', 'RCP85', 'Baseline']
Yearname = [i for i in range(Tstart, Tend + 1, Tstep)]

Ndata = np.zeros((T, nrcp))
Nmdata = np.zeros((T, nrcp))
Nfdata = np.zeros((T, nrcp))

Pdata = np.zeros((T, nrcp))
Cdata = np.zeros((T, nrcp))

Lmdata = np.zeros((T, nrcp))
Lfdata = np.zeros((T, nrcp))

Hmdata = np.zeros((T, nrcp))
Hfdata = np.zeros((T, nrcp))

hdata = np.zeros((T, nrcp))
hmdata = np.zeros((T, nrcp))
hfdata = np.zeros((T, nrcp))

Ydata = np.zeros((T, nrcp))
Ypdata = np.zeros((T, nrcp))

Condata = np.zeros((T, nrcp))
Tempdata = np.zeros((T, nrcp))

for i in range(T):
    for j in range(nrcp):
        childm = Pop_data.loc[(Pop_data['Scenario'] == 'SSP2')&(Pop_data['Education'] == 'Under 15')&(Pop_data['Sex'] == 'Male')&(Pop_data['Year'] == Yearname[i])]['Population']
        nonedm = Pop_data.loc[(Pop_data['Scenario'] == 'SSP2')&(Pop_data['Education'] == 'No Education')&(Pop_data['Sex'] == 'Male')&(Pop_data['Year'] == Yearname[i])]['Population']
        incomm = Pop_data.loc[(Pop_data['Scenario'] == 'SSP2')&(Pop_data['Education'] == 'Incomplete Primary')&(Pop_data['Sex'] == 'Male')&(Pop_data['Year'] == Yearname[i])]['Population']
        primam = Pop_data.loc[(Pop_data['Scenario'] == 'SSP2')&(Pop_data['Education'] == 'Primary')&(Pop_data['Sex'] == 'Male')&(Pop_data['Year'] == Yearname[i])]['Population']
        lseconm = Pop_data.loc[(Pop_data['Scenario'] == 'SSP2')&(Pop_data['Education'] == 'Lower Secondary')&(Pop_data['Sex'] == 'Male')&(Pop_data['Year'] == Yearname[i])]['Population']
        useconm = Pop_data.loc[(Pop_data['Scenario'] == 'SSP2')&(Pop_data['Education'] == 'Upper Secondary')&(Pop_data['Sex'] == 'Male')&(Pop_data['Year'] == Yearname[i])]['Population']
        tertim = Pop_data.loc[(Pop_data['Scenario'] == 'SSP2')&(Pop_data['Education'] == 'Post Secondary')&(Pop_data['Sex'] == 'Male')&(Pop_data['Year'] == Yearname[i])]['Population']
        childf = Pop_data.loc[(Pop_data['Scenario'] == 'SSP2')&(Pop_data['Education'] == 'Under 15')&(Pop_data['Sex'] == 'Female')&(Pop_data['Year'] == Yearname[i])]['Population']
        nonedf = Pop_data.loc[(Pop_data['Scenario'] == 'SSP2')&(Pop_data['Education'] == 'No Education')&(Pop_data['Sex'] == 'Female')&(Pop_data['Year'] == Yearname[i])]['Population']
        incomf = Pop_data.loc[(Pop_data['Scenario'] == 'SSP2')&(Pop_data['Education'] == 'Incomplete Primary')&(Pop_data['Sex'] == 'Female')&(Pop_data['Year'] == Yearname[i])]['Population']
        primaf = Pop_data.loc[(Pop_data['Scenario'] == 'SSP2')&(Pop_data['Education'] == 'Primary')&(Pop_data['Sex'] == 'Female')&(Pop_data['Year'] == Yearname[i])]['Population']
        lseconf = Pop_data.loc[(Pop_data['Scenario'] == 'SSP2')&(Pop_data['Education'] == 'Lower Secondary')&(Pop_data['Sex'] == 'Female')&(Pop_data['Year'] == Yearname[i])]['Population']
        useconf = Pop_data.loc[(Pop_data['Scenario'] == 'SSP2')&(Pop_data['Education'] == 'Upper Secondary')&(Pop_data['Sex'] == 'Female')&(Pop_data['Year'] == Yearname[i])]['Population']
        tertif = Pop_data.loc[(Pop_data['Scenario'] == 'SSP2')&(Pop_data['Education'] == 'Post Secondary')&(Pop_data['Sex'] == 'Female')&(Pop_data['Year'] == Yearname[i])]['Population']
        
        gdp = Gdp_data.loc[(Gdp_data['Scenario'] == 'SSP2')&(Gdp_data['Year'] == Yearname[i])]['value']
        Ydata[i, j] = gdp.values[0]
        
        Cdata[i, j] = childm.values[0] + childf.values[0]
        Lmdata[i, j] = nonedm.values[0] + incomm.values[0] + primam.values[0]
        Lfdata[i, j] = nonedf.values[0] + incomf.values[0] + primaf.values[0]
        Hmdata[i, j] = lseconm.values[0] + useconm.values[0] + tertim.values[0]
        Hfdata[i, j] = lseconf.values[0] + useconf.values[0] + tertif.values[0]
        hmdata[i, j] = Hmdata[i, j]/Lmdata[i, j]
        hfdata[i, j] = Hfdata[i, j]/Lfdata[i, j]
        hdata[i, j] = (Hmdata[i, j] + Hfdata[i, j])/(Lmdata[i, j] + Lfdata[i, j])
        Nmdata[i, j] = Hmdata[i, j] + Lmdata[i, j]
        Nfdata[i, j] = Hfdata[i, j] + Lfdata[i, j]
        Ndata[i, j] = Nfdata[i, j] + Nmdata[i, j]
        Pdata[i, j] = Cdata[i, j] + Nmdata[i, j] + Nfdata[i, j]
        Ypdata[i, j] = Ydata[i, j]/Pdata[i, j]

Temp0 = 17.36
for i in range(T):
    for j in range(nrcp):
        if j == nrcp - 1 or i == 0:
            Tempdata[i, j] = Temp0
        else:
            Tempdata[i, j] = Temp_data.loc[(Temp_data['RCP'] == RCPName[j])&(Temp_data['year'] == Yearname[i])]['temperature_mean'].values[0]

scale_uf = max(Pro_data.uf)
scale_um = max(Pro_data.um)
scale_sf = max(Pro_data.sf)
scale_sm = max(Pro_data.sm)

N1980 = [8402, 8594]
[wum00, wsm00, wuf00, wsf00] = [5783, 8024, 3585, 4723]

# GDP constant 2010 USD (https://data.worldbank.org/indicator/ny.gdp.mktp.kd)
Y2000 = 267e9
Ydata[0, :] = Y2000/1000
### ============================================== Model Calibration ============================================== #
def Calib(SSPx, w0x, Labor0x, Laborx, Y0x, Temp0x, Abgrx):
    
    [wum0x, wsm0x, wuf0x, wsf0x] = w0x
    [N00x, Hm0, Hf0, Lm0, Lf0] = Labor0x
    [Nxx, hx] = Laborx
    
    MaxTemp0 = coeffTmax[0] + coeffTmax[1] * Temp0x
    MaxTempIndex0 = int(round((MaxTemp0 - 12) * 5))
        
    duf0 = (Pro_data.uf[MaxTempIndex0])/(scale_uf)
    dum0 = (Pro_data.um[MaxTempIndex0])/(scale_um)
    dsf0 = (Pro_data.sf[MaxTempIndex0])/(scale_sf)
    dsm0 = (Pro_data.sm[MaxTempIndex0])/(scale_sm)
    
    drf0 = duf0/dsf0
    drm0 = dum0/dsm0
    
    num0 = Lm0/(N00x)
    nsm0 = Hm0/(N00x)
    nuf0 = Lf0/(N00x)
    nsf0 = Hf0/(N00x)
    h0 = (Hm0 + Hf0)/(Lm0 + Lf0)
    nm0 = num0 + nsm0
    nf0 = nuf0 + nsf0
    nu0 = num0 + nuf0
    ns0 = nsm0 + nsf0
    
    trm0 = (dsm0 * wsm0x)/(dum0 * wum0x)
    trf0 = (dsf0 * wsf0x)/(duf0 * wuf0x)
    tur0 = (dum0 * wum0x)/(duf0 * wuf0x)
    
    tuf0 = gamma / (trf0 * nsf0 + trm0 * tur0 * nsm0 + tur0 * num0 + nuf0)
    tum0 = tuf0 * tur0
    tsm0 = tum0 * trm0
    tsf0 = tuf0 * trf0

    wur0x = wum0x/wuf0x
    wsr0x = wsm0x/wsf0x

    Lur0 = (Lf0 * duf0)/(Lm0 * dum0)
    Lsr0 = (Hf0 * dsf0)/(Hm0 * dsm0)
    
    za = 1/(1 + 1/(wur0x * duf0/dum0 * Lur0**(-1/eta)))
    zb = 1/(1 + 1/(wsr0x * dsf0/dsm0 * Lsr0**(-1/eta)))
    
    Lrx = np.exp(-eta * (np.log(wur0x) - (1 - 1/eta) * np.log(dum0/duf0) - np.log(za/(1 - za))))
    Hrx = np.exp(-eta * (np.log(wsr0x) - (1 - 1/eta) * np.log(dsm0/dsf0) - np.log(zb/(1 - zb))))
    stuf = tuf0 + Lrx * tum0 + hx * (1 + Lrx)/(1 + Hrx) * tsf0 + hx * (1 + Lrx)/(1 + Hrx) * Hrx * tsm0
    
    nufx = gamma/stuf
    numx = Lrx * nufx
    nsfx = hx * (1 + Lrx)/(1 + Hrx) * nufx
    nsmx = hx * (1 + Lrx)/(1 + Hrx) * Hrx * nufx
    
    Lmx = numx * Nxx
    Lfx = nufx * Nxx
    Hmx = nsmx * Nxx
    Hfx = nsfx * Nxx
       
    Lfr0 = (Hf0 * dsf0)/(Lf0 * duf0)
    Lfrx = (Hfx * dsf0)/(Lfx * duf0)

    H0 = Hm0 + Hf0
    L0 = Lm0 + Lf0
    
    N0 = Lm0 + Lf0 + Hm0 + Hf0
    Nm0 = Lm0 + Hm0
    Nf0 = Lf0 + Hf0
    
    hf0d = (Hf0 * dsf0)/(Lf0 * duf0)
    hf0 = Hf0/Lf0
    hm0 = Hm0/Lm0
    h0 = (Hm0 + Hf0)/(Lm0 + Lf0)
    
    Da0 = max(0.001, g0a + g1a * Temp0 + g2a * Temp0**2)
    Db0 = max(0.001, g0b + g1b * Temp0 + g2b * Temp0**2)
    Dr0 = Db0/Da0
   
    M0 = (zb * (Hm0 * dsm0)**((eta - 1)/eta) + (1 - zb) * (Hf0 * dsf0)**((eta - 1)/eta))/(za * (Lm0 * dum0)**((eta - 1)/eta) + (1 - za) * (Lf0 * duf0)**((eta - 1)/eta))
    Mx = (zb * (Hmx * dsm0)**((eta - 1)/eta) + (1 - zb) * (Hfx * dsf0)**((eta - 1)/eta))/(za * (Lmx * dum0)**((eta - 1)/eta) + (1 - za) * (Lfx * duf0)**((eta - 1)/eta))
    
    Ar0 = np.exp(eps/(1 - eps) * (np.log((1 - theta)/theta) + np.log((1 - zb)/(1 - za)) - np.log(trf0) - 2 * np.log(drf0) - (1/eta) * np.log(Lfr0) + (eps - eta)/(eps * (eta - 1)) * np.log(M0)) - np.log(Dr0))
    Arx = np.exp(eps/(1 - eps) * (np.log((1 - theta)/theta) + np.log((1 - zb)/(1 - za)) - np.log(trf0) - 2 * np.log(drf0) - (1/eta) * np.log(Lfrx) + (eps - eta)/(eps * (eta - 1)) * np.log(Mx)) - np.log(Dr0))
    
    Ab0 = (Y0x/Db0)/(theta * (((za * (Lm0 * dum0)**((eta - 1)/eta) + (1 - za) * (Lf0 * duf0)**((eta - 1)/eta))**(eta/(eta - 1))) / (Ar0 * Dr0))**((eps - 1)/eps) + (1 - theta) * ((zb * (Hm0 * dsm0)**((eta - 1)/eta) + (1 - zb) * (Hf0 * dsf0)**((eta - 1)/eta))**(eta/(eta - 1)))**((eps - 1)/eps))**(eps/(eps - 1))
    Aa0 = Ab0/Ar0
    Arg = np.exp((np.log(Arx/Ar0))/((Tend - Tstart)/20)) - 1
    
    Abgx = (1 + Abgrx)**20 - 1
    Aagx = (1 + Abgx)/(1 + Arg) - 1
    
    Ya0 = Aa0 * Da0 * (za * (Lm0 * dum0)**((eta - 1)/eta) + (1 - za) * (Lf0 * duf0)**((eta - 1)/eta))**(eta/(eta - 1))
    Yb0 = Ab0 * Db0 * (zb * (Hm0 * dsm0)**((eta - 1)/eta) + (1 - zb) * (Hf0 * dsf0)**((eta - 1)/eta))**(eta/(eta - 1))
    Yr0 = Yb0 / Ya0
    
    pr0 = (Yr0)**(-1/eps) * (1 - theta) / theta
    
    B0 = wsr0x * drf0 * trf0 * Hm0 + wur0x * Lm0 + trf0 * drf0 * Hf0 + Lf0
    
    cbum0 = Yb0 * wur0x / B0
    cbuf0 = Yb0 / B0
    cbsm0 = Yb0 * wsr0x * drf0 * trf0 / B0
    cbsf0 = Yb0 * drf0 * trf0/ B0

    caum0 = Ya0 * wur0x / B0
    cauf0 = Ya0 / B0
    casm0 = Ya0 * wsr0x * drf0 * trf0/ B0
    casf0 = Ya0 * drf0 * trf0/ B0
    
    cum0 = (theta * caum0**((eps - 1)/eps) + (1 - theta) * cbum0**((eps - 1)/eps))**(eps/(eps - 1))
    cuf0 = (theta * cauf0**((eps - 1)/eps) + (1 - theta) * cbuf0**((eps - 1)/eps))**(eps/(eps - 1))
    
    csm0 = (theta * casm0**((eps - 1)/eps) + (1 - theta) * cbsm0**((eps - 1)/eps))**(eps/(eps - 1))
    csf0 = (theta * casf0**((eps - 1)/eps) + (1 - theta) * cbsf0**((eps - 1)/eps))**(eps/(eps - 1))
    
    wum0 = cum0 / (1 - gamma)
    wuf0 = cuf0 / (1 - gamma)
    
    wsm0 = csm0 / (1 - gamma)
    wsf0 = csf0 / (1 - gamma)
    
    pa0 = wum0 / (za * dum0 * Aa0 * Da0 * (Lm0 * dum0)**(-1/eta) * (za * (Lm0 * dum0)**((eta - 1)/eta) + (1 - za) * (Lf0 * duf0)**((eta - 1)/eta))**(1/(eta - 1)))
    pb0 = wsm0 / (zb * dsm0 * Ab0 * Db0 * (Hm0 * dsm0)**(-1/eta) * (zb * (Hm0 * dsm0)**((eta - 1)/eta) + (1 - zb) * (Hf0 * dsf0)**((eta - 1)/eta))**(1/(eta - 1)))
    
    wrm0 = wsm0/wum0
    wrf0 = wsf0/wuf0
    
    wgaps0 = wsm0/wsf0
    wgapu0 = wum0/wuf0
    
    Outputx = [num0, nsm0, nuf0, nsf0, nu0, ns0, N0, Nm0, Nf0, Temp0, MaxTemp0, h0, hm0, hf0, duf0, dum0, dsf0, dsm0, H0, Hm0 * dsm0, Hf0 * dsf0, L0, Lm0 * dum0, Lf0 * duf0, Y0x, Ya0, Yb0, Yr0, Aa0, Ab0, Ar0, Da0, Db0, Dr0, cbum0, cbuf0, cbsm0, cbsf0, caum0, cauf0, casm0, casf0, cum0, cuf0, csm0, csf0, wum0, wuf0, wsm0, wsf0, wrm0, wrf0, wgaps0, wgapu0, pa0, pb0, pr0]
    Ratex = [Aagx, Abgx, tum0, tuf0, tsm0, tsf0, za, zb]
    Numx = [num0, nuf0, nsm0, nsf0, numx, nufx, nsmx, nsfx]
    return (Outputx, Ratex, Numx)

# ============================================== Model Dynamics ============================================== #

for j in range(nrcp):
    [Output, Rate, Num] = Calib(j, [wum00, wsm00, wuf00, wsf00], [sum(N1980), Hmdata[0, j], Hfdata[0, j], Lmdata[0, j], Lfdata[0, j]], [Ndata[T - 2, j], hdata[T - 1, j]], Ydata[0, j], Tempdata[0, 0], Abgr)
    [Aag[j], Abg[j], tum, tuf, tsm, tsf, zax, zbx] = Rate
    [num1, nuf1, nsm1, nsf1, num2, nuf2, nsm2, nsf2] = Num
    [num[0, j], nsm[0, j], nuf[0, j], nsf[0, j], nu[0, j], ns[0, j], N[0, j], Nm[0, j], Nf[0, j], Temp[0, j], MaxTemp[0, j], h[0, j], hm[0, j], hf[0, j], Suf[0, j], Sum[0, j], Ssf[0, j], Ssm[0, j], H[0, j], Hm[0, j], Hf[0, j], L[0, j], Lm[0, j], Lf[0, j], Y[0, j], Ya[0, j], Yb[0, j], Yr[0, j], Aa[0, j], Ab[0, j], Ar[0, j], Da[0, j], Db[0, j], Dr[0, j], cbum[0, j], cbuf[0, j], cbsm[0, j], cbsf[0, j], caum[0, j], cauf[0, j], casm[0, j], casf[0, j], cum[0, j], cuf[0, j], csm[0, j], csf[0, j], wum[0, j], wuf[0, j], wsm[0, j], wsf[0, j], wrm[0, j], wrf[0, j], wgaps[0, j], wgapu[0, j], pa[0, j], pb[0, j], pr[0, j]] = Output
                         
    for i in range(T - 1):   
        trmN = tsm/tum
        trfN = tsf/tuf
        tur = tum/tuf
        tsr = tsm/tsf
      
        Temp[i + 1, j] = Tempdata[i + 1, j]
        MaxTemp[i + 1, j] = coeffTmax[0] + coeffTmax[1] * Temp[i + 1, j]

        Da[i + 1, j] = max(0.001, g0a + g1a * Temp[i + 1, j] + g2a * Temp[i + 1, j]**2)
        Db[i + 1, j] = max(0.001, g0b + g1b * Temp[i + 1, j] + g2b * Temp[i + 1, j]**2)
        Dr[i + 1, j] = Db[i + 1, j]/Da[i + 1, j]
        DrN = Dr[i + 1, j]    
            
        MaxTempIndex = int(round((MaxTemp[i + 1, j] - 12) * 5))
        scale_uf = max(Pro_data.uf)
        scale_um = max(Pro_data.um)
        scale_sf = max(Pro_data.sf)
        scale_sm = max(Pro_data.sm)
        
        Suf[i + 1, j] = (Pro_data.uf[MaxTempIndex])/(scale_uf)
        dufN = Suf[i + 1, j]
        Sum[i + 1, j] = (Pro_data.um[MaxTempIndex])/(scale_um)
        dumN = Sum[i + 1, j]
        Ssf[i + 1, j] = (Pro_data.sf[MaxTempIndex])/(scale_sf)
        dsfN = Ssf[i + 1, j]
        Ssm[i + 1, j] = (Pro_data.sm[MaxTempIndex])/(scale_sm)
        dsmN = Ssm[i + 1, j]
        
# ========================================================== #        
# ===================  Climate supply model ================ #
# ======= No Climate damages to sectoral productivity ====== #
# ========================================================== #
        # Da[i + 1, j] = Da[i, j]
        # Db[i + 1, j] = Db[i, j]
        # Dr[i + 1, j] = Db[i + 1, j]/Da[i + 1, j]
        # DrN = Dr[i + 1, j]     
# ========================================================== #        
# ===================  Climate demand model ================ #
# ======== No Climate damages to labor availability ======== #
# ========================================================== #
        Suf[i + 1, j] = Suf[i, j]
        dufN = Suf[i + 1, j]
        Sum[i + 1, j] = Sum[i, j]
        dumN = Sum[i + 1, j]
        Ssf[i + 1, j] = Ssf[i, j]
        dsfN = Ssf[i + 1, j]
        Ssm[i + 1, j] = Ssm[i, j]
        dsmN = Ssm[i + 1, j]     
# ========================================================== #
# ========================================================== #
# ========================================================== #
        drfN = dufN/dsfN
        drmN = dumN/dsmN

        Aa[i + 1, j] = Aa[i, j] * (1 + Aag[j])
        Ab[i + 1, j] = Ab[i, j] * (1 + Abg[j])
        Ar[i + 1, j] = Ab[i + 1, j]/Aa[i + 1, j]
        ArN = Ar[i + 1, j]
        
        nsfy = nsf1 + (nsf2 - nsf1)/(T - 1) * (i + 1) 
        
        Lry = np.exp(-eta * (np.log(tur) - (2 - 1/eta) * np.log(dumN/dufN) - np.log(zax/(1 - zax))))
        Hry = np.exp(-eta * (np.log(tsr) - (2 - 1/eta) * np.log(dsmN/dsfN) - np.log(zbx/(1 - zbx))))
        
        def Nsolve(nHfx):
            nLfx = (gamma - (Hry * tsm + tsf) * nHfx)/(Lry * tum + tuf)
            nLmx = Lry * nLfx
            nHmx = Hry * nHfx
            Mx = (zbx * (nHmx * dsmN)**((eta - 1)/eta) + (1 - zbx) * (nHfx * dsfN)**((eta - 1)/eta))/(zax * (nLmx * dumN)**((eta - 1)/eta) + (1 - zax) * (nLfx * dufN)**((eta - 1)/eta))
            diff4 = 1 - ((1 - theta)/theta) * (ArN * DrN)**((eps - 1)/eps) * (dsmN/dumN)**(2 - 1/eta) * (1/trmN) * (zbx/zax) * (nHmx/nLmx)**(-1/eta) * Mx**((eps - eta)/(eps * (eta - 1)))
            return abs(diff4)
        
        nHf = optimize.newton(Nsolve, nsfy, rtol=0.001)
        nLf = (gamma - (Hry * tsm + tsf) * nHf)/(Lry * tum + tuf)
        nLm = Lry * nLf
        nHm = Hry * nHf
    
        [NLm, NLf, NHm, NHf] = [nLm * N[i, j], nLf * N[i, j], nHm * N[i, j], nHf * N[i, j]]
        
        Ma[i + 1, j] = (zax * (NLm * dumN)**((eta - 1)/eta) + (1 - zax) * (NLf * dufN)**((eta - 1)/eta))
        Mb[i + 1, j] = (zbx * (NHm * dsmN)**((eta - 1)/eta) + (1 - zbx) * (NHf * dsfN)**((eta - 1)/eta))
        
        Lm[i + 1, j] = NLm * dumN
        Lf[i + 1, j] = NLf * dufN
        Hm[i + 1, j] = NHm * dsmN
        Hf[i + 1, j] = NHf * dsfN

        num[i + 1, j] = NLm/N[i, j]
        nuf[i + 1, j] = NLf/N[i, j]
        nsm[i + 1, j] = NHm/N[i, j]
        nsf[i + 1, j] = NHf/N[i, j]
        
        nu[i + 1, j] = num[i + 1, j] + nuf[i + 1, j]
        ns[i + 1, j] = nsm[i + 1, j] + nsf[i + 1, j]
        
        Nm[i + 1, j] = NLm + NHm
        Nf[i + 1, j] = NLf + NHf
        
        L[i + 1, j] = NLm + NLf
        H[i + 1, j] = NHm + NHf
        N[i + 1, j] = Nm[i + 1, j] + Nf[i + 1, j]
        
        
        hm[i + 1, j] = NHm / NLm
        hf[i + 1, j] = NHf / NLf
        h[i + 1, j] = H[i + 1, j] / L[i + 1, j]
        
        wurN = (zax/(1 - zax)) * (dumN/dufN) * (Lf[i + 1, j]/Lm[i + 1, j])**(1/eta)
        wsrN = (zbx/(1 - zbx)) * (dsmN/dsfN) * (Hf[i + 1, j]/Hm[i + 1, j])**(1/eta)
        BN = wsrN * drfN * trfN * NHm + wurN * NLm + trfN * drfN * NHf + NLf
        
        Ya[i + 1, j] = Aa[i + 1, j] * Da[i + 1, j] * Ma[i + 1, j]**(eta/(eta - 1))
        Yb[i + 1, j] = Ab[i + 1, j] * Db[i + 1, j] * Mb[i + 1, j]**(eta/(eta - 1))
        Yr[i + 1, j] = Yb[i + 1, j] / Ya[i + 1, j]
        
        pr[i + 1, j] = (Yr[i + 1, j])**(-1/eps) * (1 - theta) / theta
        
        cbum[i + 1, j] = Yb[i + 1, j] * wurN / BN
        cbuf[i + 1, j] = Yb[i + 1, j] / BN
        cbsm[i + 1, j] = Yb[i + 1, j] * wsrN * trfN * drfN / BN
        cbsf[i + 1, j] = Yb[i + 1, j] * trfN * drfN / BN
    
        caum[i + 1, j] = Ya[i + 1, j] * wurN / BN
        cauf[i + 1, j] = Ya[i + 1, j] / BN
        casm[i + 1, j] = Ya[i + 1, j] * wsrN * trfN * drfN / BN
        casf[i + 1, j] = Ya[i + 1, j] * trfN * drfN / BN
    
        cum[i + 1, j] = (theta * caum[i + 1, j]**((eps - 1)/eps) + (1 - theta) * cbum[i + 1, j]**((eps - 1)/eps))**(eps/(eps - 1))
        cuf[i + 1, j] = (theta * cauf[i + 1, j]**((eps - 1)/eps) + (1 - theta) * cbuf[i + 1, j]**((eps - 1)/eps))**(eps/(eps - 1))
        
        csm[i + 1, j] = (theta * casm[i + 1, j]**((eps - 1)/eps) + (1 - theta) * cbsm[i + 1, j]**((eps - 1)/eps))**(eps/(eps - 1))
        csf[i + 1, j] = (theta * casf[i + 1, j]**((eps - 1)/eps) + (1 - theta) * cbsf[i + 1, j]**((eps - 1)/eps))**(eps/(eps - 1))

        wum[i + 1, j] = cum[i + 1, j] / (1 - gamma)
        wuf[i + 1, j] = cuf[i + 1, j] / (1 - gamma)
        
        wsm[i + 1, j] = csm[i + 1, j] / (1 - gamma)
        wsf[i + 1, j] = csf[i + 1, j] / (1 - gamma)
        
        pa[i + 1, j] = wum[i + 1, j] / (zax * dumN * Aa[i + 1, j] * Da[i + 1, j] * (Lm[i + 1, j])**(-1/eta) * Ma[i + 1, j]**(1/(eta - 1)))
        pb[i + 1, j] = wsm[i + 1, j] / (zbx * dsmN * Ab[i + 1, j] * Db[i + 1, j] * (Hm[i + 1, j])**(-1/eta) * Mb[i + 1, j]**(1/(eta - 1)))
        
        wrm[i + 1, j] = wsm[i + 1, j]/wum[i + 1, j]
        wrf[i + 1, j] = wsf[i + 1, j]/wuf[i + 1, j]
        
        wgaps[i + 1, j] = wsm[i + 1, j]/wsf[i + 1, j]
        wgapu[i + 1, j] = wum[i + 1, j]/wuf[i + 1, j]
        
        Y[i + 1, j] = (theta * Ya[i + 1, j]**((eps - 1)/eps) + (1 - theta) * Yb[i + 1, j]**((eps - 1)/eps))**(eps/(eps - 1))
        Yp[i, j] = Y[i, j] / (N[i + 1, j] + N[i, j])

# ===================================================== Output ===================================================== #    
x = [2000 + i * 20 for i in range(T)]

for k in range(nrcp):
    plt.plot(x, wgaps[:, k], label=RCPName[k])
plt.xlabel('Time')
plt.ylabel('Wage ratio of male to female')
plt.title('High-skilled gender pay gap')
axes = plt.gca()
plt.xticks(np.arange(min(x), max(x) + 1, 20))
plt.legend(loc=2, prop={'size':8})
plt.show()

for k in range(nrcp):
    plt.plot(x, wgapu[:, k], label=RCPName[k])
plt.xlabel('Time')
plt.ylabel('Wage ratio of male to female')
plt.title('low-skilled gender pay gap')
axes = plt.gca()
plt.xticks(np.arange(min(x), max(x) + 1, 20))
plt.legend(loc=2, prop={'size':8})
plt.show()

plt.plot(x, hf[:, 4], 'blue', label= 'Female (model)', )
plt.plot(x, hfdata[:, 4], 'b--', label= 'Female (data)')
plt.plot(x, hm[:, 4], 'red', label= 'Male (model)', )
plt.plot(x, hmdata[:, 4], 'r--', label= 'Male (data)')   
plt.xlabel('Time')
plt.ylabel('Ratio of high-skilled to low-skilled labor')
plt.title('No climate change')
axes = plt.gca()
plt.xticks(np.arange(min(x), max(x) + 1, 20))
plt.legend(loc=2, prop={'size':8})
plt.show()

plt.plot(x, hf[:, 0], 'blue', label= 'Female (model)', )
plt.plot(x, hfdata[:, 0], 'b--', label= 'Female (data)')
plt.plot(x, hm[:, 0], 'red', label= 'Male (model)', )
plt.plot(x, hmdata[:, 0], 'r--', label= 'Male (data)')   
plt.xlabel('Time')
plt.ylabel('Ratio of high-skilled to low-skilled labor')
plt.title('RCP 2.6')
axes = plt.gca()
plt.xticks(np.arange(min(x), max(x) + 1, 20))
plt.legend(loc=2, prop={'size':8})
plt.show()

plt.plot(x, hf[:, 1], 'blue', label= 'Female (model)', )
plt.plot(x, hfdata[:, 1], 'b--', label= 'Female (data)')
plt.plot(x, hm[:, 1], 'red', label= 'Male (model)', )
plt.plot(x, hmdata[:, 1], 'r--', label= 'Male (data)')   
plt.xlabel('Time')
plt.ylabel('Ratio of high-skilled to low-skilled labor')
plt.title('RCP 4.5')
axes = plt.gca()
plt.xticks(np.arange(min(x), max(x) + 1, 20))
plt.legend(loc=2, prop={'size':8})
plt.show()

plt.plot(x, hf[:, 2], 'blue', label= 'Female (model)', )
plt.plot(x, hfdata[:, 2], 'b--', label= 'Female (data)')
plt.plot(x, hm[:, 2], 'red', label= 'Male (model)', )
plt.plot(x, hmdata[:, 2], 'r--', label= 'Male (data)')   
plt.xlabel('Time')
plt.ylabel('Ratio of high-skilled to low-skilled labor')
plt.title('RCP 6.0')
axes = plt.gca()
plt.xticks(np.arange(min(x), max(x) + 1, 20))
plt.legend(loc=2, prop={'size':8})
plt.show()

plt.plot(x, hf[:, 3], 'blue', label= 'Female (model)', )
plt.plot(x, hfdata[:, 3], 'b--', label= 'Female (data)')
plt.plot(x, hm[:, 3], 'red', label= 'Male (model)', )
plt.plot(x, hmdata[:, 3], 'r--', label= 'Male (data)')   
plt.xlabel('Time')
plt.ylabel('Ratio of high-skilled to low-skilled labor')
plt.title('RCP 8.5')
axes = plt.gca()
plt.xticks(np.arange(min(x), max(x) + 1, 20))
plt.legend(loc=2, prop={'size':8})
plt.show()

# =============================================== Export into Excel =============================================== #
def output(filename, sheet, list1, v):
    book = xlwt.Workbook(filename)
    sh = book.add_sheet(sheet)

    v1_desc = 'za'
    v2_desc = 'zb'
    v3_desc = 'epsilon'
    v4_desc = 'tum'
    v5_desc = 'tsm'
    desc = [v1_desc, v2_desc, v3_desc, v4_desc, v5_desc]
    m = 0
    for v_desc, v_v in zip(desc, v):
        sh.write(m, 0, v_desc)
        sh.write(m, 1, v_v)
        m = m + 1
        
    varname = ['Time', 'Da', 'Db', 'Lm', 'Lf',  'L',  'Hm',  'Hf', 'H', 'Nm', 'Nf', 'N', 'hm', 'hf', 'h', 'wum', 'wuf', 'wsm', 'wsf', 'Temp', 'MaxTemp', 'Da', 'Db', 'Sum', 'Suf', 'Ssm', 'Ssf', 'Pa', 'Pb', 'Ya', 'Yb', 'Conc', 'Y']
    
    m = 5
    for jj in range(nrcp):
        for indx , qq in enumerate(range(2000, 2120, 20), 1):
            sh.write(m + 0, jj * 10 + indx, qq)
            # sh.write(m + 0, jj * 10, varname[0])
        for k in range(32):
            for indx in range(T):
                sh.write(m + k + 1, jj * 10 + indx + 1, list1[k][indx][jj])
                # sh.write(m + k + 1, jj * 10, varname[k + 1]) 
    book.save(filename)
    
output1 = [Da, Db, Lm, Lf, L, Hm, Hf, H, Nm, Nf, N, hm, hf, h, wum, wuf, wsm, wsf, Temp, MaxTemp, Da, Db, Sum, Suf, Ssm, Ssf, pa, pb, Ya, Yb, Condata, Y]
par = [zax, zbx, eps, tum, tsm]

output('CGE_SA_Output_RCP_19Nov21-D.xls', '1', output1, par)