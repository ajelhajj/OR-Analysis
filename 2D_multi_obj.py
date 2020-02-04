###############################################################################
##### Imports #################################################################
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.close('all')
###############################################################################
##### Font Set-Up #############################################################
###############################################################################
rc = {"font.family" : "sans-serif", 
      "mathtext.fontset" : "stix"}
plt.rcParams.update(rc)
plt.rcParams["font.sans-serif"] = ["Gill Sans MT"] + plt.rcParams["font.sans-serif"]

SMALL_SIZE = 15
MEDIUM_SIZE = 19
BIGGER_SIZE = 22

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
###############################################################################
##### Rational Return Function ################################################
###############################################################################
def rational_fxn(a, h, k):
    x_pos=np.arange(-0.02,1.02,0.0001)
    y_pos=[]
    #Calculate y values 
    for x in range(len(x_pos)):
        x_val=x_pos[x]
        y=(a/(x_val-h)) + k
        y_pos.append(y)
    return x_pos, y_pos
###############################################################################
##### Data and Normalization ##################################################
###############################################################################
df = pd.read_csv('Data/Monthly_Metrics.csv') # Read in dataset of monthly stats
dep = [df.Department == 'ACS', df.Department == 'BLUE'] # create list of series for conditions
dr = [(df.Date >= '2013-10') & (df.Date < '2015-10'), (df.Date >= '2015-10') & (df.Date <= '2017-10')] # repeat
Pre, Post = df[dr[0] & dep[1]], df[dr[1] & dep[1]]

x_1_all, x_2_all = Pre['Opportunity Unused Min'], Post['Opportunity Unused Min'] # Underutilization
x_1_all, x_2_all = x_1_all.tolist(), x_2_all.tolist()
x_min = min(x_1_all + x_2_all)
x_max = max(x_1_all + x_2_all)
x_1_all, x_2_all = [(float(i)-x_min)/(x_max-x_min) for i in x_1_all], [(float(i)-x_min)/(x_max-x_min) for i in x_2_all] # normalize against the maximum 

y_1_all, y_2_all = Pre['After hours Min'] + Pre['Out of Block Min'], Post['After hours Min'] + Post['Out of Block Min'] # Overutilization
y_1_all, y_2_all = y_1_all.tolist(), y_2_all.tolist()
y_min = min(y_1_all + y_2_all)
y_max = max(y_1_all + y_2_all)
y_1_all, y_2_all = [(float(i)-y_min)/(y_max-y_min) for i in y_1_all], [(float(i)-y_min)/(y_max - y_min) for i in y_2_all] # normalize against the maximum 

z_1_all, z_2_all = Pre['wRVU/FTE'], Post['wRVU/FTE'] # wRVU/FTE
z_1_all, z_2_all = z_1_all.tolist(), z_2_all.tolist()
z_min = min(z_1_all + z_2_all)
z_max = max(z_1_all + z_2_all)
z_1_all, z_2_all = [(float(i)-z_min)/(z_max-z_min) for i in z_1_all], [(float(i)-z_min)/(z_max-z_min) for i in z_2_all] # normalize against the maximum 

# Fuse 3 normalized lists into dataframe for selections:
dict_pre = {'Underutilization':x_1_all,'Overutilization':y_1_all, 'Productivity':z_1_all}
dict_post = {'Underutilization':x_2_all,'Overutilization':y_2_all, 'Productivity':z_2_all}

df_pre = pd.DataFrame(dict_pre)
df_post = pd.DataFrame(dict_post)
###############################################################################
##### Productivity vs Underutilization ########################################
###############################################################################
fig = plt.figure()
plt.axis([-0.02, 1.02, -0.05, 1.05])
plt.gca().invert_yaxis()
# Curve Plotting
x_pos, y_pos = rational_fxn(-0.16, 0.002, 1.3)
cond = x_pos >= 0.002
x_pos = x_pos[cond]
y_pos=np.array(y_pos)[cond]
plt.plot(x_pos, y_pos, linestyle='-', color='#b2abd2', lw = 2.5) # parabola line

x_pos, y_pos = rational_fxn(-0.2, -0.22, 1.35)
plt.plot(x_pos, y_pos,zorder=100, linestyle='-', color='#5e3c99', lw = 2.5) # parabola line

plt.scatter(x_1_all, z_1_all, label = 'Pre-Transition', c = '#b2abd2', ec = 'k', s = 65) #f0f0f0
plt.scatter(x_2_all, z_2_all, c = '#5e3c99', label = 'Post-Transition', ec = 'k', s = 65) #636363

plt.legend()
plt.legend(loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=2, 
            borderaxespad=0, frameon=False)

plt.xlabel('Underutilized Time (hours)') # labelpad addresses the issue of label overlapping axes numbers
plt.ylabel('Productivity (wRVU/FTE)')

plt.show()
plt.tight_layout()
###############################################################################
##### Productivity vs Overutilization #########################################
###############################################################################
fig = plt.figure()
plt.axis([-0.02, 1.02, -0.05, 1.05])
plt.gca().invert_yaxis()
# Curve Plotting
x_pos, y_pos = rational_fxn(-0.05, 0.01, 1.16)
cond = x_pos >= 0.01
x_pos = x_pos[cond]
y_pos=np.array(y_pos)[cond]
plt.plot(x_pos, y_pos, linestyle='-', color='#e66101', lw = 2.5) # parabola line

x_pos, y_pos = rational_fxn(-0.003, -0.018, 1.05)
cond = x_pos >= -0.018
x_pos = x_pos[cond]
y_pos=np.array(y_pos)[cond]
plt.plot(x_pos, y_pos,zorder=100, linestyle='-', color='#fdb863', lw = 2.5) # parabola line

plt.scatter(y_1_all, z_1_all, label = 'Pre-Transition', c = '#e66101', ec = 'k', s = 85)
plt.scatter(y_2_all, z_2_all, label = 'Post-Transition', c= '#fdb863', ec = 'k', s = 85)

plt.legend()
plt.legend(loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=2, 
            borderaxespad=0, frameon=False)

plt.xlabel('Overutilized Time (hours)', labelpad=15) # labelpad addresses the issue of label overlapping axes numbers
plt.ylabel('Productivity (wRVU/FTE)', labelpad=15)

plt.show()
plt.tight_layout()
###############################################################################