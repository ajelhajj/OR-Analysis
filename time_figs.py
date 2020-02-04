###############################################################################
##### Imports #################################################################
###############################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
from statsmodels.nonparametric.smoothers_lowess import lowess
import pymannkendall as mk

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
##### Data Cleanup ############################################################
###############################################################################
df = pd.read_csv("Data/surgical_case.csv")
df['Room No.'] = pd.to_numeric(df['Room'].str[-2:]) # Extract OR Room digits and convert from object to int64
df['Case ID'] = pd.to_numeric(df['Case'].str[5:])*10  # Extract Case digits and convert from object to int64
df = df[df['Primary Department'].str #get only ACS and Blue Departments
        .contains('Blue', na = False) | df['Primary Department']
        .str.contains('Acute', na = False)]
#combine ACS deps
df['Primary Department'] = df['Primary Department'].replace({'UVMMC Acute Care Service Emergency': 'ACS',
                                                             'Acute Care Service': 'ACS'})
# Convert from object to datetime64 
df['Actual Room In DateTime'] = pd.to_datetime(df['Actual Room In DateTime']) 
df['Actual Room Out DateTime'] = pd.to_datetime(df['Actual Room Out DateTime']) 
df['Date'] = pd.to_datetime(df['Date']) 
#keep valuable columns 
df = df[['Date',
         'Case ID',
         'Primary Department', 
         'All Departments',
         'Room',
         'Room No.', 
         'Actual Room In DateTime', 
         'Actual Room Out DateTime']]
#rename to case
case = df.copy()
#make uppercase Primary Department to match block df
case['Primary Department'] = case['Primary Department'].apply(lambda x: x.upper())
#complete case data with relavent info
complete_df = case
#blank df
time_df = pd.DataFrame()
#copy ids
time_df.loc[:,'ID'] = complete_df['Case ID']
#copy department
time_df.loc[:,'Department'] = complete_df['Primary Department']
#copy date of surgery
time_df.loc[:,'Date'] = complete_df.Date
#get year, month, ordinal week, ordinal day and hour of surgery
time_df.loc[:,'Year'] = complete_df.Date.dt.year
time_df.loc[:,'Month'] = complete_df.Date.dt.month
time_df.loc[:,'Month Name'] = complete_df.Date.dt.month_name().str[:3]
time_df.loc[:,'Week'] = complete_df.Date.dt.week
time_df.loc[:,'WkDay'] = complete_df.Date.dt.weekday
time_df.loc[:,'Hour'] = complete_df['Actual Room In DateTime'].dt.hour
GrpStuff = time_df.groupby(['Year','Month','Department'])
#get month names used
M = [*GrpStuff['Month Name'].first()][::2]
#get matching year
Y = [*GrpStuff['Year'].first()][::2]
#initialize array for x-ticks
M_Y = []
#loop through months used
for i in np.arange(len(M)):
    #create string to label x axis
    #Mon-YYYY
    #ex: Dec-2019
    x = M[i] + '-' + str(Y[i])
    #add to list
    M_Y.append(x)
#unstack the grouping to split departments
#this gives department case count by month
CntStuff = GrpStuff.count()['ID'].unstack()
CntStuff.reset_index(level=0, inplace=True)
CntStuff.reset_index(level=0, inplace=True)
#CntStuff.reset_index(level=0, inplace=True)
df = CntStuff
df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(Day=1))

GrpDay = time_df.groupby(['Date','Department'])
CntDay = GrpDay.count()['ID'].unstack()
CntDay.reset_index(level=0, inplace=True)
CntDay = CntDay.fillna(0)
###############################################################################
##### Hourly Caseload #########################################################
###############################################################################
fig = plt.figure(figsize=(19.2,10.8))
ax = fig.add_subplot(1, 1, 1)
linestyles = ['-', "--"]
CntHour = time_df.groupby(['Hour','Department']).count()['ID'].unstack()
CntHour.reset_index(level=0, inplace=True)
plt.plot(CntHour['Hour'], CntHour['ACS'], '-o', color='k', lw = 1.8, linestyle = '--', label = 'ACS')
plt.plot(CntHour['Hour'], CntHour['BLUE'], '-o', color='k', lw = 1.8, linestyle = '-', label = 'GS')
plt.ylabel('Surgical Caseload')
plt.xlabel('Hour of Day')
plt.rc('grid', linestyle="dotted", color='gray')
major_ticks = np.arange(0, 24, 4)
minor_ticks = np.arange(0, 24, 1)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.grid(which='both')
plt.legend(labels = ['ACS', 'GS'])
plt.savefig('Figures/dailycaseload.png', dpi = 300, bbox_inches="tight")
###############################################################################
##### Daily Caseload ##########################################################
###############################################################################
fig, axs = plt.subplots(2, sharex=True, sharey=True, figsize=(19.2,10.8))
axs[0].plot(CntDay['Date'], CntDay['ACS'], '-s', color='k', lw = 1, linestyle = '-', label = 'ACS')
axs[1].plot(CntDay['Date'], CntDay['BLUE'], '-o', color='k', lw = 1, linestyle = '-', label = 'GS')
plt.gcf().autofmt_xdate()
plt.xlim([datetime.date(2013, 9, 1), datetime.date(2017, 10, 1)])
months = mdates.MonthLocator()  # every month
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.gca().xaxis.set_minor_locator(months)
plt.text('10/2015', 8.5, "Transition")
plt.ylabel("Surgical Caseload")
plt.axvline('10/2015', color='k', linestyle='--')
plt.rc('grid', linestyle="dotted", color='gray')
plt.grid(which='minor', alpha=0.3)
plt.grid(which='major', alpha=1)
plt.legend()
axs[0].legend()
axs[0].axvline('10/2015', color='k', linestyle='--')
axs[0].grid(which='minor', alpha=0.3)
axs[0].grid(which='major', alpha=1)
plt.yticks(np.arange(0, 10+1, 2))
plt.show()
plt.savefig('Figures/dailycaseload.png', dpi = 300, bbox_inches="tight")
###############################################################################
##### Sum Monthly Caseload ####################################################
###############################################################################
fig = plt.figure(figsize=(19.2,10.8))
plt.plot(df['Date'],df['ACS'], '-o', color='k', lw = 1.8, linestyle = '--', label = 'ACS')
plt.plot(df['Date'],df['BLUE'], '-o', color='k', lw = 1.8, linestyle = '-', label = 'GS')
plt.gcf().autofmt_xdate()
plt.xlim([datetime.date(2013, 9, 1), datetime.date(2017, 10, 1)])
months = mdates.MonthLocator()  # every month
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.gca().xaxis.set_minor_locator(months)
plt.text('10/2015', 3, "Transition")
plt.ylabel("Surgical Caseload")
plt.axvline('10/2015', color='k', linestyle='solid')
plt.rc('grid', linestyle="dotted", color='gray')
plt.grid(which='minor', alpha=0.3)
plt.grid(which='major', alpha=1)
plt.legend()
plt.show()
plt.savefig('Figures/monthlysums.png', dpi = 300, bbox_inches="tight")
###############################################################################
##### Lowess Smoothing ########################################################
###############################################################################
def make_lowess(series, frac):
    endog = series.values
    exog = series.index.values

    smooth = lowess(endog, exog, frac)
    index, data = np.transpose(smooth)

    return pd.Series(data, index=pd.to_datetime(index))
fig = plt.figure(figsize=(19.2,10.8))
plt.plot(df['Date'],df['ACS'], 'x', color='k', label = 'ACS')
plt.plot(df['Date'],df['BLUE'], 'o', color='k', label = 'GS')
plt.gcf().autofmt_xdate()
months = mdates.MonthLocator()  # every month
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.gca().xaxis.set_minor_locator(months)
plt.text('10/2015', 45, "Transition")
plt.ylabel("Surgical Caseload")
plt.axvline('10/2015', color='k', linestyle='solid')
plt.rc('grid', linestyle="dotted", color='gray')
plt.grid(which='minor', alpha=0.3)
plt.grid(which='major', alpha=1)
df.index = df['Date']
gs = make_lowess(df['BLUE'], 0.65)
gs.plot(label='GS', linestyle = '-', c = 'k')
acs = make_lowess(df['ACS'], 0.65)
acs.plot(label='ACS', linestyle = '--', c = 'k')
plt.xlim([datetime.date(2013, 9, 1), datetime.date(2017, 10, 1)])
handles, labels = plt.gca().get_legend_handles_labels()
newLabels, newHandles = [], []
for handle, label in zip(handles, labels):
  if label not in newLabels:
    newLabels.append(label)
    newHandles.append(handle)
plt.legend(newHandles, newLabels)
plt.show()
plt.savefig('Figures/lowesssmoothing.png', dpi = 300, bbox_inches="tight")

result_acs = mk.original_test(df['ACS']) # There is a monotonic increasing trend
result_gs = mk.original_test(df['BLUE']) # There is a monotonic decreasing trend
print(result_acs)
print('\n')
print(result_gs)
