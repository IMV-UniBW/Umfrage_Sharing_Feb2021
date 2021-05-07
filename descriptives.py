# -*- coding: utf-8 -*-
"""
Created on Wed May  5 08:56:42 2021

@author: katha
"""

# import modules
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from patsy import dmatrices
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ilonas Umfrage
# einlesen
import os
os.chdir(r"C:\Users\katha\OneDrive\Dokumente\To Do")
df = pd.read_csv(r'.\Umfrage_Ilona.csv', error_bad_lines=False, sep = ';', encoding='iso-8859-1')
df.rename(columns={'Unnamed: 0': 'interview'}, inplace=True)
df = df.drop([0,1])

#VAR'
varlist = ['A101',
'A107',
'RB02',
'RB03',
'RB04_01',
'RB04_02',
'RB04_03',
'RB04_04',
'RB05',
'SD09_01',
'SD01',
'SD03',
'SD19',
'SD20',
'SD11',
'SD11_10',
'SD17',
'SD18_01',
'SD22',
'V901',
'V903_01',
'V903_02',
'V903_03',
'V903_04',
'V903_05',
'V903_06',
'V903_07',
'V903_08',
'V903_09',
'V904',
'V904x01',
'V904x02',
'V905',
'V908x01',
'V908x11',
'V908x12',
'V908x13',
'V908x101',
'V908x02',
'V001',
'V003_01',
'V003_02',
'V003_03',
'V003_04',
'V003_05',
'V003_06',
'V003_07',
'V003_08',
'V003_09',
'V004',
'V004x01',
'V004x02',
'V004x03',
'V005',
'V007_CN',
'V007x01',
'V007x11',
'V007x12',
'V007x13',
'V007x101',
'V007x02',
'VM01_01',
'AV12',
'AV04',
'AV04x01',
'AV04x02',
'AV04x03',
'AV04x04',
'AV04x05',
'AV04x06',
'AV04x07',
'AV04x08',
'AV04x09',
'AV06',
'AV06_01',
'AV06_02',
'AV06_04',
'AV06_07',
'AV06_06',
'AV06_06a']

# kürzen
df_short = df[varlist]

# umbenennen
new_columns = {'A101': 'Autobesitz',
                'A107': 'Autoleistung',
'RB02': 'Zeitkarte',
'RB03': 'Fuehrerschein',
'RB04_01': 'Autozugang',
'RB04_02': 'Fahrradzugang',
'RB04_03': 'Gegenstandstransport',
'RB04_04': 'Personentransport',
'RB05': 'Campusbewohner',
'SD09_01': 'PLZ',
'SD01':	'Geschlecht',
'SD03':	'Alter',
'SD19':	'Haushalt',
'SD20':	'Kinder',
'SD11':	'Bildungsabschluss',
'SD11_10':	'Bildungsabschluss2',
'SD17':	'Einkommen',
'SD22':	'Gruppe',
'V901':	'Taetigkeit_2019',
'V908x01':	'Sharing_2019',
'V908x11':	'Sharing_2019_Carsharing',
'V908x12':	'Sharing_2019_Bikesharing',
'V908x13':	'Sharing_2019_Roller',
'V908x101':	'Sharing_2019_andere',
'V908x02': 'Sharing_2019_nein',
'V007x01':	'Sharing_2020',
'V007x11':	'Sharing_2020_Carsharing',
'V007x12':	'Sharing_2020_Bikesharing',
'V007x13':	'Sharing_2020_Roller',
'V007x101':	'Sharing_2020_andere',
'V007x02'	:'Sharing_2020_nein',
'VM01_01':	'Entfernung_Wohnort_Campus',
'AV12':	'Bereitschaft_Sharing',
'AV04':	'Anzahl_Anforderungen_Sharing',
'AV04x01':	'Anforderungen1' ,
'AV04x02':	'Anforderungen2',
'AV04x03':	'Anforderungen3' ,
'AV04x04':	'Anforderungen4',
'AV04x05':	'Anforderungen5',
'AV04x06':	'Anforderungen6' ,
'AV04x07':	'Anforderungen7' ,
'AV04x08':	'Anforderungen8' ,
'AV04x09':	'Anforderungen9' ,
'AV06':	'Anzahl_gewaehlter_Sharing_Verkehrsmittel',
'AV06_01':	 'desired_Sharing_Autos',
'AV06_02':	 'desired_Sharing_Fahrräder',
'AV06_04':	 'desired_Sharing_Lastenräder',
'AV06_07':	 'desired_Sharing_Roller',
'AV06_06':	 'desired_Sharing_Sonstige',
'AV06_06a':	 'desired_Sharing_Sonstige_open'}
df_short.rename(columns=new_columns, inplace=True)

# info
n = len(df_short)
######################################################################
# Univariate Descriptives
# -----------------------------------------------------
# Sharing Erfahrung 2019
# -----------------------------------------------------
per_sharing_all = sum(df_short.Sharing_2019[~np.isnan(df_short.Sharing_2019.astype('float'))].astype('int')-1)/sum(~np.isnan(df_short.Sharing_2019.astype('float')))
per_sharing_car = sum(df_short.Sharing_2019_Carsharing[~np.isnan(df_short.Sharing_2019_Carsharing.astype('float'))].astype('int')-1)/sum(~np.isnan(df_short.Sharing_2019_Carsharing.astype('float')))
per_sharing_bike = sum(df_short.Sharing_2019_Bikesharing[~np.isnan(df_short.Sharing_2019_Bikesharing.astype('float'))].astype('int')-1)/sum(~np.isnan(df_short.Sharing_2019_Bikesharing.astype('float')))
per_sharing_scooter = sum(df_short.Sharing_2019_Roller[~np.isnan(df_short.Sharing_2019_Roller.astype('float'))].astype('int')-1)/sum(~np.isnan(df_short.Sharing_2019_Roller.astype('float')))
per_sharing_other = sum(df_short.Sharing_2019_andere[~np.isnan(df_short.Sharing_2019_andere.astype('float'))].astype('int')-1)/sum(~np.isnan(df_short.Sharing_2019_andere.astype('float')))
labels = ['Alle', 'Auto', 'Fahrrad', 'Roller/Scooter', 'Andere']
x = [per_sharing_all, per_sharing_car, per_sharing_bike, per_sharing_scooter, per_sharing_other]
fig, ax = plt.subplots()
rects1 = ax.bar(range(len(x)), x )
ax.set_ylabel('Prozent')
plt.xticks(range(len(x)))

ax.set_title('Sharing Erfahrung mit...')
ax.set_xticklabels(labels)
ax.legend()

# -----------------------------------------------------
# Sharing Erfahrung 2020
# -----------------------------------------------------
per_sharing_all = sum(df_short.Sharing_2020[~np.isnan(df_short.Sharing_2020.astype('float'))].astype('int')-1)/sum(~np.isnan(df_short.Sharing_2020.astype('float')))
per_sharing_car = sum(df_short.Sharing_2020_Carsharing[~np.isnan(df_short.Sharing_2020_Carsharing.astype('float'))].astype('int')-1)/sum(~np.isnan(df_short.Sharing_2020_Carsharing.astype('float')))
per_sharing_bike = sum(df_short.Sharing_2020_Bikesharing[~np.isnan(df_short.Sharing_2020_Bikesharing.astype('float'))].astype('int')-1)/sum(~np.isnan(df_short.Sharing_2020_Bikesharing.astype('float')))
per_sharing_scooter = sum(df_short.Sharing_2020_Roller[~np.isnan(df_short.Sharing_2020_Roller.astype('float'))].astype('int')-1)/sum(~np.isnan(df_short.Sharing_2020_Roller.astype('float')))
per_sharing_other = sum(df_short.Sharing_2020_andere[~np.isnan(df_short.Sharing_2020_andere.astype('float'))].astype('int')-1)/sum(~np.isnan(df_short.Sharing_2020_andere.astype('float')))
labels = ['Alle', 'Auto', 'Fahrrad', 'Roller/Scooter', 'Andere']
x = [per_sharing_all, per_sharing_car, per_sharing_bike, per_sharing_scooter, per_sharing_other]
fig, ax = plt.subplots()
rects1 = ax.bar(range(len(x)), x )
ax.set_ylabel('Prozent')
ax.set_title('Sharing Erfahrung mit...')
ax.set_xticklabels(labels)
ax.legend()
plt.xticks(range(len(x)))

# ----------------------------------------------------
# Bereitschaft NUtzung 
#  -------------------------------------------------
df_short['Bereitschaft_Sharing2'] = ''
df_short.loc[df_short['Bereitschaft_Sharing'] == '1', ['Bereitschaft_Sharing2']] = '1'
df_short.loc[df_short['Bereitschaft_Sharing'] == '4', ['Bereitschaft_Sharing2']] = '1'
df_short.loc[df_short['Bereitschaft_Sharing'] == '5', ['Bereitschaft_Sharing2']] = '0'
df_short.loc[df_short['Bereitschaft_Sharing'] == '-9', ['Bereitschaft_Sharing2']] = 'NaN'
df_short.loc[np.isnan(df_short.Bereitschaft_Sharing.astype('float')), 'Bereitschaft_Sharing2'] = 'NaN'

sns.catplot(x="Bereitschaft_Sharing2", kind = 'count', data=df_short[~np.isnan(df_short.Bereitschaft_Sharing2.astype('float'))])

# ----------------------------------------------------
# Anforderungen
# ----------------------------------------------------
anforderungen = df_short.Anforderungen1.tolist()
for i in range(2,10):
    temp_list = eval("df_short.Anforderungen" + str(i))
    anforderungen.extend(temp_list.tolist())

anforderungen_sortiert = np.unique(anforderungen)
df_anforderungen = pd.DataFrame(anforderungen_sortiert, columns = ['Anforderungen'])
df_anforderungen.to_pickle("anforderungen_sharing.csv")

np.savetxt("anforderungen_sharing.csv", anforderungen_sortiert,  fmt = '%s') #delimiter=",",
