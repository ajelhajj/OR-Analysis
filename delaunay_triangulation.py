###############################################################################
##### Imports #################################################################
###############################################################################
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.tri as mtri
import numpy as np
from scipy.spatial import ConvexHull

plt.close('all')
###############################################################################
##### Font Set-Up #############################################################
###############################################################################
rc = {"font.family" : "serif", 
      "mathtext.fontset" : "stix"}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

SMALL_SIZE = 10
MEDIUM_SIZE = 19
BIGGER_SIZE = 22

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
###############################################################################
##### Data and Normalization ##################################################
###############################################################################
df = pd.read_csv('Data/Monthly_Metrics.csv') # Read in dataset of monthly stats
dep = [df.Department == 'ACS', df.Department == 'BLUE'] # create list of series for conditions
dr = [(df.Date >= '2013-10') & (df.Date < '2015-10'), (df.Date >= '2015-10') & (df.Date <= '2017-10')] # repeat
Pre, Post = df[dr[0] & dep[1]], df[dr[1] & dep[1]]

x_1, x_2 = Pre['Opportunity Unused Min'], Post['Opportunity Unused Min'] # Underutilization
x_1, x_2 = x_1.tolist(), x_2.tolist()
x_max = max(x_1 + x_2)
x_1, x_2 = [float(i)/x_max for i in x_1], [float(i)/x_max for i in x_2] # normalize against the maximum 

y_1, y_2 = Pre['After hours Min'] + Pre['Out of Block Min'], Post['After hours Min'] + Post['Out of Block Min'] # Overutilization
y_1, y_2 = y_1.tolist(), y_2.tolist()
y_max = max(y_1 + y_2)
y_1, y_2 = [float(i)/y_max for i in y_1], [float(i)/y_max for i in y_2] # normalize against the maximum 

z_1, z_2 = Pre['wRVU/FTE'], Post['wRVU/FTE'] # wRVU/FTE
z_1, z_2 = z_1.tolist(), z_2.tolist()
z_max = max(z_1 + z_2)
z_1, z_2 = [float(i)/z_max for i in z_1], [float(i)/z_max for i in z_2] # normalize against the maximum 

# Fuse 3 normalized lists into dataframe for selections:
dict_pre = {'Underutilization':x_1,'Overutilization':y_1, 'Productivity':z_1}
dict_post = {'Underutilization':x_2,'Overutilization':y_2, 'Productivity':z_2}

df_pre = pd.DataFrame(dict_pre)
df_post = pd.DataFrame(dict_post)
###############################################################################
##### Plane Border Printing Function ##########################################
###############################################################################
def outline(x, y, alpha, line):
    ''' This function takes arrays for x and y as well as transparency. It then
    returns the outlines of the convex hull from these points.
    '''
    points = x, y
    points = np.array(points).T
    hull = ConvexHull(points)
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], line, alpha = alpha)
###############################################################################
##### Plotting ################################################################
###############################################################################
outline(x_1, y_1, 1, 'k-') # Plane Boundaries
outline(x_2, y_2, 1, 'k-')
# cmap = cm.get_cmap(name='Greys', lut=None)
cmap = cm.plasma
triang = mtri.Triangulation(x_1, y_1)
pcb = plt.tricontourf(triang, z_1, origin='lower', levels = np.linspace(0, 1, 11),  cmap=cmap)
pca = plt.tricontour(triang, z_1, origin='lower', levels = np.linspace(0, 1, 11),  colors='k')
plt.scatter(x_1, y_1, c = z_1, edgecolors='black', s = 100, marker = 'o', cmap = cmap, lw = 1.5)

triang = mtri.Triangulation(x_2, y_2)
plt.clim(0, 1) # manually fix the range of the colorscale so that the scatter and contour match
pcd = plt.tricontourf(triang, z_2, origin='lower', levels = np.linspace(0, 1, 11),  cmap=cmap)
pcc = plt.tricontour(triang, z_2, origin='lower', levels = np.linspace(0, 1, 11),  colors='k')
plt.scatter(x_2, y_2, c = z_2, edgecolors='black', s = 100, marker = 'o', cmap = cmap, lw = 1.5)
plt.clim(0, 1) # manually fix the range of the colorscale so that the scatter and contour match

clb = plt.colorbar(pcb)
clb.set_label('Productivity $\it{(z)}$', labelpad=-40, y=1.05, rotation=0, fontsize=MEDIUM_SIZE)

plt.annotate(xy=[0.8, 0.09], s ='Pre-Transition')
plt.annotate(xy=[0.5, 0.01], s ='Post-Transition')

plt.xlabel('Underutilization $\it{(x)}$')
plt.ylabel('Overutilization $\it{(y)}$')
plt.grid(linestyle='dotted', color = "gray")
plt.axis([0, 1.02, -0.02, 1.02])
plt.show()