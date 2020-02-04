###############################################################################
##### Imports #################################################################
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st

plt.close('all')
###############################################################################
##### Font Set-Up #############################################################
###############################################################################
rc = {"font.family" : "sans-serif", 
      "mathtext.fontset" : "stix"}
plt.rcParams.update(rc)
plt.rcParams["font.sans-serif"] = ["Gill Sans MT"] + plt.rcParams["font.sans-serif"]

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
##### Inter-event Function ####################################################
###############################################################################
def inter_event(data):
    data = data.sort_values(by = 'Actual Room In DateTime')
    for i in range(1, len(data)):
        data.loc[data.index[i], 'Inter-Event Timedelta'] = data.loc[data.index[i], 'Actual Room In DateTime'] - data.loc[data.index[i-1], 'Actual Room Out DateTime']
        data.loc[data.index[i], 'Inter-Event Time (hr)'] = data.loc[data.index[i], 'Inter-Event Timedelta'].total_seconds()/3600
    return data
###############################################################################
##### Data Cleaning ###########################################################
###############################################################################
df = pd.read_csv('Data/surgical_case.csv') # Read in dataset of monthly stats
df['Room No.'] = pd.to_numeric(df['Room'].str[-2:]) # Extract OR Room digits and convert from object to int64
df['Case ID'] = pd.to_numeric(df['Case'].str[5:])  # Extract Case digits and convert from object to int64
df = df[df['Primary Department'].str.contains('Blue', na = False) | df['Primary Department'].str.contains('Acute', na = False)]
df['Primary Department'] = df['Primary Department'].replace({'UVMMC Acute Care Service Emergency': 'ACS', 'Acute Care Service': 'ACS'})
df['Actual Room In DateTime'] = pd.to_datetime(df['Actual Room In DateTime']) # Convert from object to datetime64 
df['Actual Room Out DateTime'] = pd.to_datetime(df['Actual Room Out DateTime']) # Convert from object to datetime64 
df = df[['Case ID','Primary Department', 'Room No.', 'Actual Room In DateTime', 'Actual Room Out DateTime']]
df = df[df['Room No.'] == 1]
#df = df[(df['Day'] == 'Mon') | (df['Day'] == 'Tue') | (df['Day'] == 'Wed') | (df['Day'] == 'Thu') | (df['Day'] == 'Fri')]

df_pre = df[(df['Actual Room Out DateTime'] >= '2013-10-01') & (df['Actual Room Out DateTime'] < '2015-10-01')]
df_post = df[(df['Actual Room Out DateTime'] >= '2015-10-01') & (df['Actual Room Out DateTime'] < '2017-10-01')]

df_pre = inter_event(df_pre)
df_post = inter_event(df_post)
df_post = df_post[60:131]
#start_biz_str = '07:00:00'
#end_biz_str = '17:30:00'
#df = df[(df['Actual Room In DateTime'].dt.time > pd.to_datetime(start_biz_str).time()) & (df['Actual Room In DateTime'].dt.time < pd.to_datetime(end_biz_str).time())]

###############################################################################
##### Visualization of Distributions ##########################################
###############################################################################
figure = plt.figure(figsize=(19.2,10.8))
binBoundaries = np.linspace(0,600,101)
plt.subplot(1, 2, 1)
plt.hist(df_pre['Inter-Event Time (hr)'], bins = binBoundaries, color = 'k')
plt.axis([0, 500, 0, 10])
plt.xlabel("OR 1 Inter-Event Time, $t$ (hr)")
plt.ylabel("Frequency")
plt.title("Pre-Transition")
plt.rc('grid', linestyle="dotted", color='gray')
plt.grid()

plt.subplot(1, 2, 2)
hist = plt.hist(df_post['Inter-Event Time (hr)'], bins = binBoundaries, color = 'k')
plt.axis([0, 500, 0, 10])
#plt.axis([0, 600, 0, 30])
plt.title("Post-Transition")
plt.xlabel("OR 1 Inter-Event Time, $t$ (hr)")
plt.ylabel("Frequency")
plt.grid()
# Using maximum likelihood estimation for power law fitting:
nbins = 101
smax = df_post['Inter-Event Time (hr)'].max()
tmean = df_post['Inter-Event Time (hr)'].mean()
rate = 1./tmean
dist_exp = st.expon.pdf(binBoundaries, scale=1./rate)
plt.plot(binBoundaries, dist_exp * len(df_post['Inter-Event Time (hr)']) * smax / nbins,'-b', lw=4)
plt.savefig('Figures/interevent_OR1_distr.png', dpi = 300, bbox_inches="tight")
# Kolmogorov-Smirnov test for goodness of fit:
dist = st.expon
args = dist.fit(df_post.dropna()['Inter-Event Time (hr)']) 
st.kstest(df_post['Inter-Event Time (hr)'].dropna(), dist.cdf, args)

plt.tight_layout()

# CDF
figure = plt.figure(figsize=(19.2,10.8))
X = sorted(df_pre['Inter-Event Time (hr)'])
N = len(df_pre['Inter-Event Time (hr)'])
Y = 1.0*np.arange(N)/N
#plt.plot(X,Y,'-.', label = 'Pre-Transition', color = 'k', lw = 2)
plt.xlabel("Inter-Event Time, $t$ (hr)")
plt.ylabel("$P_<(t)$") # shorthand for P(M < m)
X = sorted(df_post['Inter-Event Time (hr)'])
N = len(df_post['Inter-Event Time (hr)'])
Y = 1.0*np.arange(N)/N
plt.plot(X,Y,'-', label = 'Post-Transition', color = 'k', lw = 2)
plt.legend()
plt.grid()
plt.show()
plt.savefig('Figures/CDFs_OR1.png', dpi = 300, bbox_inches="tight")