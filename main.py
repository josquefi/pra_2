#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 18:23:04 2023

@author: josep
"""

import pandas as pd
from scipy.stats import normaltest
from scipy.stats import levene
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt


#-------------------------
# Importem les dades
#-------------------------

df_c = pd.read_csv('data/fao_crops.csv')
df_f = pd.read_csv('data/fao_fertilizers_use.csv')
df_p = pd.read_csv('data/fao_pesticides_use.csv')

#-------------------------
# Selecció de les columnes da cada dataset
#-------------------------

df_c = df_c[['Area', 'Item', 'Element', 'Year', 'Value']]
df_c = df_c.pivot(index=['Area', 'Item', 'Year'], columns=['Element'])
df_c.reset_index(inplace=True)
df_c.columns = ['country', 'crop', 'year', 'area_harvested', 'production', 'yield']

# Com que les dades sobre producció tenen molt més detall del que necessitem fem un groupby i sumem
df_c = df_c.groupby(['year', 'country']).sum().reset_index()

df_f = df_f[['country_name_en', 'year', 'use_per_area_of_cropland_kg_ha']]
df_f = df_f.rename(columns={'country_name_en':'country','use_per_area_of_cropland_kg_ha':'nitrogen'})

df_p = df_p[['country_name_en', 'year', 'use_per_area_of_cropland_kg_ha']]
df_p = df_p.rename(columns={'country_name_en':'country', 'use_per_area_of_cropland_kg_ha':'pesticides'})

#-------------------------
# Integració dels diferents datasets
#-------------------------

df = df_c.merge(df_f, on=['country', 'year']).merge(df_p, on=['country', 'year'])

#-------------------------
# Selecció només de països
#-------------------------

# Eliminem no països

nopais = ['World', 'Americas', 'South America', 'Asia',
 'Northern America',
       'Net Food Importing Developing Countries', 'Europe',
       'European Union (28)', 'European Union (27)', 'Southern Europe',
       'Southern Asia', 'Africa', 'Low Income Food Deficit Countries',
       'Central America', 'Northern Africa',
       'Eastern Asia', 'Western Asia',
       'China, mainland', 'South-eastern Asia', 'Southern Africa',
        'Least Developed Countries',
        'Small Island Developing States',
       'Land Locked Developing Countries', 'Caribbean',
        'Western Africa', 'Oceania','Eastern Africa']


dfp = df[~df['country'].isin(nopais)]

#-------------------------
# Neteja de les dades
#-------------------------

# Valors nuls
dfp.isnull().sum()

#Zeros
dfp.eq(0).sum()

# Outliers

dfp_outliers = dfp[['country', 'area_harvested', 'production', 'yield', 'nitrogen', 'pesticides']]

# Calculem el IQR per cada columna
Q1 = dfp_outliers.quantile(0.25)
Q3 = dfp_outliers.quantile(0.75)
IQR = Q3 - Q1

# Trobem els valors que estan fora del rang interquartílic
outliers = (dfp_outliers < (Q1 - 1.5 * IQR)) | (dfp_outliers > (Q3 + 1.5 * IQR))

# Seleccionem els outliers
outlier_rows = dfp_outliers[outliers.any(axis=1)]

#-------------------------
# Test de normalitat
#-------------------------

for column in dfp[['area_harvested', 'production', 'yield', 'nitrogen', 'pesticides']].columns:
    stat, p_value = normaltest(dfp[column])
    alpha = 0.05

    print(f"Column: {column}")
    print(f"Test statistic: {stat}")
    print(f"P-value: {p_value}")

    if p_value > alpha:
        print("Les dades es distribueixen normalment.\n")
    else:
        print("Les dades no es distribueixen normalment.\n")

#-------------------------
# Test d'homocedasticitat
#-------------------------

stat, p_value = levene(dfp['area_harvested'], dfp['production'], dfp['yield'], 
                       dfp['nitrogen'], dfp['pesticides'], dfp['yield'])
alpha = 0.05

print("Test d'homoscedasticitat")
print(f"Test statistic: {stat}")
print(f"P-value: {p_value}")

if p_value > alpha:
    print("Les dades mostren homoscedasticitat.\n")
else:
    print("Les dades no mostren homoscedasticitat.\n")
    
#-------------------------
# Test de correlació
#-------------------------

dfp_xina = dfp[dfp.country == 'China']
dfp_xina = dfp_xina.set_index('year')

corr, p_value = pearsonr(dfp_xina['yield'], dfp_xina['nitrogen'])

print(f"Correlation coefficient: {corr}")
print(f"P-value: {p_value}")

if p_value < 0.05:
    print("Hi ha una correlació significativa.\n")
else:
    print("No hi ha una correlació significativa.\n")
    
corr, p_value = pearsonr(dfp_xina['yield'], dfp_xina['pesticides'])

print(f"Correlation coefficient: {corr}")
print(f"P-value: {p_value}")

if p_value < 0.05:
    print("Hi ha una correlació significativa.\n")
else:
    print("No hi ha una correlació significativa.\n")

#-------------------------
# Test ADF
#-------------------------

for column in dfp_xina[['area_harvested', 'production', 'yield', 'nitrogen', 'pesticides']].columns:
    result = adfuller(dfp_xina[column])
    
    test_statistic = result[0]
    p_value = result[1]

    print(f"Column: {column}")
    print(f"Test Statistic: {test_statistic}")
    print(f"P-value: {p_value}")

    if p_value < 0.05:
        print("És estacionària.\n")
    else:
        print("No és estacionària.\n")
        
#-------------------------
# Regressió
#-------------------------

model = sm.OLS(dfp['yield'], sm.add_constant(dfp[['nitrogen', 'pesticides']]))
fixed_effects_model = model.fit(cov_type='cluster', cov_kwds={'groups': dfp['country']})

print(fixed_effects_model.summary())

#-------------------------
# Visualització
#-------------------------

# Evolució area_harvested

sns.lineplot(data=dfp, x='year', y='area_harvested')

# Evolució productivitat

sns.lineplot(data=dfp, x='year', y='yield')

# Evolució ús nitrogen i pesticides

sns.lineplot(data=dfp, x='year', y='nitrogen')
sns.lineplot(data=dfp, x='year', y='pesticides')

# Visualització mapa

def vis_mapa(df):
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    dfp_2019 = df[df.year == 2019]

    # Unim les dades amb el shapefile
    merged = world.merge(dfp_2019, left_on='name', right_on='country', how='left')

    # Fem el plot del mapa
    fig, ax = plt.subplots(figsize=(10, 6))
    merged.plot(column='yield', cmap='coolwarm', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)

    # Indiquem el titol
    plt.title('Productivitat')

    plt.show()

vis_mapa(dfp)

# Relacions entre variables

# Cas Xina

sns.regplot(x = "yield", y = "nitrogen", data = dfp_xina)
sns.regplot(x = "yield", y = "pesticides", data = dfp_xina)

# Cas Espanya

dfp_sp = dfp[dfp.country == 'Spain']

sns.regplot(x = "yield", y = "nitrogen", data = dfp_sp)
sns.regplot(x = "yield", y = "pesticides", data = dfp_sp)

# Cas Inida

dfp_in = dfp[dfp.country == 'India']

sns.regplot(x = "yield", y = "nitrogen", data = dfp_in)
sns.regplot(x = "yield", y = "pesticides", data = dfp_in)
