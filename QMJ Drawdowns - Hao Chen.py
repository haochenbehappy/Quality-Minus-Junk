#!/usr/bin/env python
# coding: utf-8

# # [Quality Minus Junk](https://www.aqr.com/Insights/Datasets/Quality-Minus-Junk-Factors-Monthly)
# A strategy from AQR Capital Management that goes long in safe, profitable, growing and well-managed stocks while goes short in stocks with the opposite characteristics. 
# Below is the derivation for the maximum drawdowns of the QMJ monthly portfolios. (1957-2019)

# In[279]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[280]:


# Path should be modified according to file path on your computer
path ='/Users/haochen/Downloads/Alternative Investment Strategies/Quality Minus Junk 10 QualitySorted Portfolios Monthly.xlsx'

df = pd.read_excel(path, sheet_name='Sheet2', header=0, index_col=0, parse_dates=True)

df.head()


# In[281]:


# Rename column names to avoid confusion
print(list(df.columns))

col_rename = ['P1 (low quality)', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10 (high quality)', 'P10-P1', 'P1_cumu', 'P10_cumu', 'P10-P1_cumu', 
          'P1 (low quality).1', 'P2.1', 'P3.1', 'P4.1', 'P5.1', 'P6.1', 'P7.1', 'P8.1', 'P9.1', 'P10 (high quality).1', 'P10.1-P1.1_cumu']

df = df.reindex(col_rename, axis=1)
print(list(df.columns))


# In[282]:


# Checking column names
df.iloc[:,:14].head()


# In[283]:


# Caculating cumu returns (Net Asset Values)
def cumulative(dframe, col_a, col_b):
    
    index_a = list(dframe.columns).index(col_a)
    index_b = list(dframe.columns).index(col_b)
    
    for i in range(0,len(dframe)):
        if ((dframe.iloc[i, index_a] == np.nan ) or (dframe.iloc[i, index_b] == np.nan) ):
            continue
        if ((i ==0) or (dframe.iloc[i-1, index_a] == np.nan)):
            dframe.iloc[0, index_b] = 1*(1+dframe.iloc[0, index_a])
        else:
            dframe.iloc[i, index_b] = dframe.iloc[i-1, index_b] * (1+dframe.iloc[i, index_a])


# In[284]:


# US sample
cumulative(df, 'P1 (low quality)', 'P1_cumu')
cumulative(df, 'P10 (high quality)', 'P10_cumu')
cumulative(df, 'P10-P1', 'P10-P1_cumu')

# # Global sample: first initiate two columns as cumu returns for P1.1 and P10.1
# df['P1.1_cumu'] = np.nan
# df['P10.1_cumu'] = np.nan
# df['P10.1-P1.1'] = df['P10 (high quality).1'] - df['P1 (low quality).1']

# cumulative(df, 'P1 (low quality).1', 'P1.1_cumu')
# cumulative(df, 'P10 (high quality).1', 'P10.1_cumu')
# cumulative(df, 'P10.1-P1.1', 'P10.1-P1.1_cumu')


# In[285]:


# Plotting cumu returns
fig, axes= plt.subplots(1,2, sharex=True, figsize=[18,6])
axes[0].plot(df[['P1_cumu', 'P10_cumu', 'P10-P1_cumu']])
axes[0].set_title('Cumulative Return since 1957')
axes[0].legend(['short', 'long', 'long-short'])
plt.tight_layout(True)


axes[1].plot(df[['P1 (low quality)', 'P10 (high quality)', 'P10-P1']])
axes[1].set_title('Monthly Return since 1957')
axes[1].legend(['short', 'long', 'long-short'])
plt.tight_layout(True)


# # Calculating Maximum Drawdown

# In[286]:


# Caculating drawdown
def drawdown(dframe, nav):


    max_cumu = 'Max_'+nav
    drawdown_cumu = 'Drawdown_'+nav
    dframe[max_cumu] = np.nan
    dframe[drawdown_cumu] = np.nan

    for i in range(0,len(dframe)):
        dframe[max_cumu][i] = max( dframe[nav][0:(i+1)] )
        dframe[drawdown_cumu] = dframe[nav] / dframe[max_cumu] - 1
    
    max_drawdown = min(dframe[drawdown_cumu])
    return max_drawdown


# In[287]:


drawdowns = pd.DataFrame([[drawdown(df, 'P1_cumu')], [drawdown(df, 'P10_cumu')], [drawdown(df, 'P10-P1_cumu')]], 
                         index=['short', 'long', 'long-short'], columns=['Drawdown'])

# From float to percentage with two decimal points
drawdowns['Drawdown'] = drawdowns['Drawdown'].apply(lambda x: format(x, '.2%'))
drawdowns

