###############################################################################
##### Imports #################################################################
###############################################################################
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

plt.close('all')

###############################################################################
##### Font Set-Up #############################################################
###############################################################################
rc = {"font.family" : "sans-serif", 
      "mathtext.fontset" : "stix"}
plt.rcParams.update(rc)
plt.rcParams["font.sans-serif"] = ["Gill Sans MT"] + plt.rcParams["font.sans-serif"]

SMALL_SIZE = 15
MEDIUM_SIZE = 23
BIGGER_SIZE = 26

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
###############################################################################
##### Data and Normalization ##################################################
###############################################################################
df = pd.read_csv('Data/Monthly_Metrics.csv') # Read in dataset of monthly stats
dep = [df.Department == 'ACS', df.Department == 'BLUE'] # create list of series for conditions
dr = [(df.Date >= '2013-10') & (df.Date < '2015-10'), (df.Date >= '2015-10') & (df.Date <= '2017-10')] # repeat
Pre, Post = df[dr[0] & dep[1]], df[dr[1] & dep[1]]

x_1_all, x_2_all = Pre['Opportunity Unused Min'], Post['Opportunity Unused Min'] # Underutilization
x_1_all, x_2_all = x_1_all.tolist(), x_2_all.tolist()

y_1_all, y_2_all = Pre['After hours Min'] + Pre['Out of Block Min'], Post['After hours Min'] + Post['Out of Block Min'] # Overutilization
y_1_all, y_2_all = y_1_all.tolist(), y_2_all.tolist()

z_1_all, z_2_all = Pre['wRVU/FTE'], Post['wRVU/FTE'] # wRVU/FTE
z_1_all, z_2_all = z_1_all.tolist(), z_2_all.tolist()

# Fuse 3 normalized lists into dataframe for selections:
dict_pre = {'Underutilization':x_1_all,'Overutilization':y_1_all, 'Productivity':z_1_all}
dict_post = {'Underutilization':x_2_all,'Overutilization':y_2_all, 'Productivity':z_2_all}

df_pre = pd.DataFrame(dict_pre)
df_post = pd.DataFrame(dict_post)
###############################################################################
##### Select Interpolation Points #############################################
###############################################################################
x_1 = x_1_all
y_1 = y_1_all
z_1 = z_1_all

df_post = df_post.sort_values(by=['Productivity'])
x_2 = df_post['Underutilization'][0:16]
y_2 = df_post['Overutilization'][0:16]
z_2 = df_post['Productivity'][0:16]
###############################################################################
##### Pre-Transition Surface #################################################
###############################################################################
data = np.c_[x_1,y_1,z_1]
data_all = np.c_[x_1_all,y_1_all,z_1_all]
mn = np.min(data_all, axis=0)
mx = np.max(data_all, axis=0)
X,Y = np.meshgrid(np.linspace(mn[0], mx[0], 20), np.linspace(mn[1], mx[1], 20)) # np.linspace(mn[0], mx[0]-0.2, 20), np.linspace(mn[1], mx[1], 20)
XX = X.flatten()
YY = Y.flatten()

# best-fit quadratic curve
A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])
# evaluate it on a grid
Z_ = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)
Z = Z_ + 130
# plot fitted surface
fig = plt.figure()
ax = Axes3D(fig)
ax.invert_zaxis() 
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color = '#998ec3', alpha=0.6, edgecolors='k', lw=0.2, label = 'Pre-Transition') 
ax.scatter(x_1_all, y_1_all, z_1_all, c='#998ec3', s=80, edgecolors = 'k', marker = 'o', depthshade=False)
plt.show()
plt.xlabel('Underutilized Time (hours)', labelpad=20) # labelpad addresses the issue of label overlapping axes numbers
plt.ylabel('Overutilized Time (hours)', labelpad=20)
ax.set_zlabel('Productivity (wRVU/FTE)', labelpad=20)

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')
###############################################################################
##### Post-Transition Surface #################################################
###############################################################################
data = np.c_[x_2,y_2,z_2]
data_all = np.c_[x_2_all,y_2_all,z_2_all]

mn = np.min(data_all, axis=0)
mx = np.max(data_all, axis=0)
X,Y = np.meshgrid(np.linspace(mn[0], mx[0], 20), np.linspace(mn[1], mx[1]+500, 20))
XX = X.flatten()
YY = Y.flatten()

# best-fit quadratic curve
A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])
# evaluate it on a grid
Z = Z_ + 105 #Z = - np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape) + 1200

ax.plot_surface(X-400, Y-600, Z, rstride=1, cstride=1, alpha=0.6, edgecolors='k', lw=0.2, color = '#f1a340')
ax.scatter(x_2_all, y_2_all, z_2_all, c='#f1a340', s=80, edgecolors = 'k', marker = 'o', depthshade=False)

# Add legend with proxy artists
col1_patch = mpatches.Patch(color='#998ec3', label='Pre-Transition',ec='k')
col2_patch = mpatches.Patch(color='#f1a340', label='Post-Transition', ec='k', lw = 1.5)                        
plt.legend(handles=[col1_patch, col2_patch]) # legend
plt.show()