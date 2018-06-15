#!/usr/bin/python

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt


# get all the subdirectory names
dirs=next(os.walk('.'))[1]		# [1] means it's a file!

# os.chdir(Time1)
os.chdir('0.02')
Sample_Fields=next(os.walk('.'))[2]	# [2] means it's a file!

# create empty data frame
Samples_Scalars_df = pd.DataFrame()
#Samples_Vec_df = pd.DataFrame()

# loop over the different Sampled Fields

# x='buffalo'    
# exec("%s = %d" % (x,2))

for i in range(0,len(Sample_Fields)):

	if Sample_Fields[i]!='U':

		temp=np.loadtxt(Sample_Fields[i],skiprows=4)
		# convert to pandas Table
		temp_df=pd.DataFrame(temp)
		temp[temp<0.0000000000001]=0.0
		# assign the column names
		temp_df.columns=['Time','Val_1','Val_2']
		#temp_df=temp_df.set_index('Time')
		if i==0:
			Samples_Scalars_df['Time']=temp_df['Time']

		Samples_Scalars_df[Sample_Fields[i]] = temp_df['Val_1']

Samples_Scalars_df=Samples_Scalars_df.set_index('Time')
###########################################################

#save the data
Samples_Scalars_df.to_csv('../Sampled_Data.csv')

###########################################################
# Do some plotting
 #CH4=plt.figure(0)
CH4_fields=['CH4_1', 'CH4_2','CH4_3','CH4_4','CH4_5','CH4_6','CH4_7','CH4_8','CH4']
my_colors = [(x/20.0, x/20.0, 0.8) for x in range(len(CH4_fields)-1)]
my_colors.append([0.75,0,0])

CH4=Samples_Scalars_df.plot(y=CH4_fields,grid='on',title='CH4 concentrations',color=my_colors)
CH4.lines[-1].set_linewidth(4)
CH4.set_ylabel("Concentration")

H2O_fields=['H2O_1', 'H2O_2', 'H2O_3', 'H2O_4', 'H2O_5', 'H2O_6', 'H2O_7', 'H2O_8', 'H2O']
H2O = Samples_Scalars_df.plot(y=H2O_fields,grid='on',title='H2O concentrations',colors=my_colors)
H2O.lines[-1].set_linewidth(4)
H2O.set_ylabel("Concentration")


he=(Samples_Scalars_df['he_1']+Samples_Scalars_df['he_2']+Samples_Scalars_df['he_3']+Samples_Scalars_df['he_4']+Samples_Scalars_df['he_5']+Samples_Scalars_df['he_6']+Samples_Scalars_df['he_7']+Samples_Scalars_df['he_8'])/8.0
Samples_Scalars_df['he']=he
he_fields=['he_1', 'he_2', 'he_3', 'he_4', 'he_5', 'he_6', 'he_7', 'he_8','he']
He = Samples_Scalars_df.plot(y=he_fields,grid='on',title='Enthalpy',colors=my_colors)
He.lines[-1].set_linewidth(4)
He.set_ylabel("kg/s^2/m^2")


plt.show()

#plt.close()








