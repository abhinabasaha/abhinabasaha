---
title: "Time Series Analysis"
date: 2019-04-10T15:34:30-04:00
categories:
  - Blog
tags:
  - Python
  - Time Series
---

This post is meant to provide a quick, concise but comprehensive overview of how to deal with time series datasets  and its typical pre-processing steps. It also gives a brief introduction to stationary and non-stationary data.

**Time Series:** <br>
Time series is a sequence of information which attaches a time period to each value. The value can be pretty much anything measurable that depends on time in some way. Like stock prices, humidity or number of people. As long as the values record are unambiguous any medium could be measured with time series. There aren't any limitations regarding the total time span of the time series. It could be a minute a day a month or even a century. All that is needed is a starting and an ending point usually.
<br>
Time Series Data donot follow any of the standard distributions because they never satisfies Gauss-Markov assumptions unlike regular linear regression data.<br>

**DATASET**<br>
Here I will be using a weather time series dataset recorded by the Max Planck Institute for Biogeochemistry.
<br>
This dataset contains 14 different features such as air temperature, atmospheric pressure, and humidity. These were collected every 10 minutes,between 2009 and 2016.<br>


```python
import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

sns.set()
```


```python
zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)

# Storing it into a data frame
df = pd.read_csv(csv_path)

```

**Creating a copy of the data**


```python
data = df.copy()
```

**Data Preprocessing**

**Visualizing and processing the Column names**


```python
data.columns
```




    Index(['Date Time', 'p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)',
           'rh (%)', 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
           'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
           'wd (deg)'],
          dtype='object')



Here it can be seen the column names have gaps/blank spaces in between, this can cause error while selecting particular columns by mentioning the coloumn names. So eliminating those gaps can solve the problem.


```python
data.columns = data.columns.str.replace(' ', '')
data.columns
```




    Index(['DateTime', 'p(mbar)', 'T(degC)', 'Tpot(K)', 'Tdew(degC)', 'rh(%)',
           'VPmax(mbar)', 'VPact(mbar)', 'VPdef(mbar)', 'sh(g/kg)',
           'H2OC(mmol/mol)', 'rho(g/m**3)', 'wv(m/s)', 'max.wv(m/s)', 'wd(deg)'],
          dtype='object')



**Checking for duplicate instances.**


```python
len(data) -len(data['DateTime'].unique())
```




    327



So there are 327 duplicate values. These can be removed by the drop_duplicate method.


```python
data.drop_duplicates(subset ="DateTime", 
                     keep = 'first',  # Considering the first value as unique and rest of the same values as duplicate.
                     inplace = True) 
```

Rechecking for duplicate instances shows null. Thus duplicates have been succesfully removed.


```python
len(data) -len(data['DateTime'].unique())
```




    0



**Indexing with date-time stamps**<br>
Visualizing the data with three time stamps. Here the columns represent instances and the attributes are shown in rows. <br>
The first row contains the time stamps. <br>
The dataset contains **14 attributes.**<br>
Moreover, the date-time stamp is present as an attribute. Using it as the index of the dataframe is more convenient.


```python
data.head(3).T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>DateTime</th>
      <td>01.01.2009 00:10:00</td>
      <td>01.01.2009 00:20:00</td>
      <td>01.01.2009 00:30:00</td>
    </tr>
    <tr>
      <th>p(mbar)</th>
      <td>996.52</td>
      <td>996.57</td>
      <td>996.53</td>
    </tr>
    <tr>
      <th>T(degC)</th>
      <td>-8.02</td>
      <td>-8.41</td>
      <td>-8.51</td>
    </tr>
    <tr>
      <th>Tpot(K)</th>
      <td>265.4</td>
      <td>265.01</td>
      <td>264.91</td>
    </tr>
    <tr>
      <th>Tdew(degC)</th>
      <td>-8.9</td>
      <td>-9.28</td>
      <td>-9.31</td>
    </tr>
    <tr>
      <th>rh(%)</th>
      <td>93.3</td>
      <td>93.4</td>
      <td>93.9</td>
    </tr>
    <tr>
      <th>VPmax(mbar)</th>
      <td>3.33</td>
      <td>3.23</td>
      <td>3.21</td>
    </tr>
    <tr>
      <th>VPact(mbar)</th>
      <td>3.11</td>
      <td>3.02</td>
      <td>3.01</td>
    </tr>
    <tr>
      <th>VPdef(mbar)</th>
      <td>0.22</td>
      <td>0.21</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>sh(g/kg)</th>
      <td>1.94</td>
      <td>1.89</td>
      <td>1.88</td>
    </tr>
    <tr>
      <th>H2OC(mmol/mol)</th>
      <td>3.12</td>
      <td>3.03</td>
      <td>3.02</td>
    </tr>
    <tr>
      <th>rho(g/m**3)</th>
      <td>1307.75</td>
      <td>1309.8</td>
      <td>1310.24</td>
    </tr>
    <tr>
      <th>wv(m/s)</th>
      <td>1.03</td>
      <td>0.72</td>
      <td>0.19</td>
    </tr>
    <tr>
      <th>max.wv(m/s)</th>
      <td>1.75</td>
      <td>1.5</td>
      <td>0.63</td>
    </tr>
    <tr>
      <th>wd(deg)</th>
      <td>152.3</td>
      <td>136.1</td>
      <td>171.6</td>
    </tr>
  </tbody>
</table>
</div>



Before using the date-time as index its better to change it to python's date time object. 


```python
data['DateTime'].describe()
```




    count                  420224
    unique                 420224
    top       13.12.2012 05:00:00
    freq                        1
    Name: DateTime, dtype: object




```python
data['DateTime'] = pd.to_datetime(data['DateTime'])
data['DateTime'].describe()
```




    count                  420224
    unique                 420224
    top       2012-10-26 12:50:00
    freq                        1
    first     2009-01-01 00:10:00
    last      2017-01-01 00:00:00
    Name: DateTime, dtype: object



Setting date-time as the dataframe index.


```python
data.set_index('DateTime', inplace=True)

```


```python
data.head(5).T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>DateTime</th>
      <th>2009-01-01 00:10:00</th>
      <th>2009-01-01 00:20:00</th>
      <th>2009-01-01 00:30:00</th>
      <th>2009-01-01 00:40:00</th>
      <th>2009-01-01 00:50:00</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>p(mbar)</th>
      <td>996.52</td>
      <td>996.57</td>
      <td>996.53</td>
      <td>996.51</td>
      <td>996.51</td>
    </tr>
    <tr>
      <th>T(degC)</th>
      <td>-8.02</td>
      <td>-8.41</td>
      <td>-8.51</td>
      <td>-8.31</td>
      <td>-8.27</td>
    </tr>
    <tr>
      <th>Tpot(K)</th>
      <td>265.40</td>
      <td>265.01</td>
      <td>264.91</td>
      <td>265.12</td>
      <td>265.15</td>
    </tr>
    <tr>
      <th>Tdew(degC)</th>
      <td>-8.90</td>
      <td>-9.28</td>
      <td>-9.31</td>
      <td>-9.07</td>
      <td>-9.04</td>
    </tr>
    <tr>
      <th>rh(%)</th>
      <td>93.30</td>
      <td>93.40</td>
      <td>93.90</td>
      <td>94.20</td>
      <td>94.10</td>
    </tr>
    <tr>
      <th>VPmax(mbar)</th>
      <td>3.33</td>
      <td>3.23</td>
      <td>3.21</td>
      <td>3.26</td>
      <td>3.27</td>
    </tr>
    <tr>
      <th>VPact(mbar)</th>
      <td>3.11</td>
      <td>3.02</td>
      <td>3.01</td>
      <td>3.07</td>
      <td>3.08</td>
    </tr>
    <tr>
      <th>VPdef(mbar)</th>
      <td>0.22</td>
      <td>0.21</td>
      <td>0.20</td>
      <td>0.19</td>
      <td>0.19</td>
    </tr>
    <tr>
      <th>sh(g/kg)</th>
      <td>1.94</td>
      <td>1.89</td>
      <td>1.88</td>
      <td>1.92</td>
      <td>1.92</td>
    </tr>
    <tr>
      <th>H2OC(mmol/mol)</th>
      <td>3.12</td>
      <td>3.03</td>
      <td>3.02</td>
      <td>3.08</td>
      <td>3.09</td>
    </tr>
    <tr>
      <th>rho(g/m**3)</th>
      <td>1307.75</td>
      <td>1309.80</td>
      <td>1310.24</td>
      <td>1309.19</td>
      <td>1309.00</td>
    </tr>
    <tr>
      <th>wv(m/s)</th>
      <td>1.03</td>
      <td>0.72</td>
      <td>0.19</td>
      <td>0.34</td>
      <td>0.32</td>
    </tr>
    <tr>
      <th>max.wv(m/s)</th>
      <td>1.75</td>
      <td>1.50</td>
      <td>0.63</td>
      <td>0.50</td>
      <td>0.63</td>
    </tr>
    <tr>
      <th>wd(deg)</th>
      <td>152.30</td>
      <td>136.10</td>
      <td>171.60</td>
      <td>198.00</td>
      <td>214.30</td>
    </tr>
  </tbody>
</table>
</div>



Lets check the statistical data (such as count, mean etc) of each attribute.<br>
It can seen the **pressure**, **temperature**, **density**, **wind degree(WD)** have relatively high mean comparing to other attributes.


```python
data.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>p(mbar)</th>
      <td>420224.0</td>
      <td>989.214157</td>
      <td>8.360888</td>
      <td>913.60</td>
      <td>984.20</td>
      <td>989.58</td>
      <td>994.73</td>
      <td>1015.35</td>
    </tr>
    <tr>
      <th>T(degC)</th>
      <td>420224.0</td>
      <td>9.442421</td>
      <td>8.421135</td>
      <td>-23.01</td>
      <td>3.36</td>
      <td>9.40</td>
      <td>15.46</td>
      <td>37.28</td>
    </tr>
    <tr>
      <th>Tpot(K)</th>
      <td>420224.0</td>
      <td>283.484880</td>
      <td>8.502206</td>
      <td>250.60</td>
      <td>277.43</td>
      <td>283.46</td>
      <td>289.52</td>
      <td>311.34</td>
    </tr>
    <tr>
      <th>Tdew(degC)</th>
      <td>420224.0</td>
      <td>4.953472</td>
      <td>6.731171</td>
      <td>-25.01</td>
      <td>0.23</td>
      <td>5.21</td>
      <td>10.07</td>
      <td>23.11</td>
    </tr>
    <tr>
      <th>rh(%)</th>
      <td>420224.0</td>
      <td>76.028738</td>
      <td>16.460467</td>
      <td>12.95</td>
      <td>65.24</td>
      <td>79.30</td>
      <td>89.40</td>
      <td>100.00</td>
    </tr>
    <tr>
      <th>VPmax(mbar)</th>
      <td>420224.0</td>
      <td>13.568642</td>
      <td>7.734770</td>
      <td>0.95</td>
      <td>7.77</td>
      <td>11.81</td>
      <td>17.59</td>
      <td>63.77</td>
    </tr>
    <tr>
      <th>VPact(mbar)</th>
      <td>420224.0</td>
      <td>9.532333</td>
      <td>4.183996</td>
      <td>0.79</td>
      <td>6.21</td>
      <td>8.86</td>
      <td>12.35</td>
      <td>28.32</td>
    </tr>
    <tr>
      <th>VPdef(mbar)</th>
      <td>420224.0</td>
      <td>4.036225</td>
      <td>4.891287</td>
      <td>0.00</td>
      <td>0.87</td>
      <td>2.18</td>
      <td>5.29</td>
      <td>46.01</td>
    </tr>
    <tr>
      <th>sh(g/kg)</th>
      <td>420224.0</td>
      <td>6.021503</td>
      <td>2.656043</td>
      <td>0.50</td>
      <td>3.92</td>
      <td>5.59</td>
      <td>7.80</td>
      <td>18.13</td>
    </tr>
    <tr>
      <th>H2OC(mmol/mol)</th>
      <td>420224.0</td>
      <td>9.638778</td>
      <td>4.235244</td>
      <td>0.80</td>
      <td>6.28</td>
      <td>8.96</td>
      <td>12.48</td>
      <td>28.82</td>
    </tr>
    <tr>
      <th>rho(g/m**3)</th>
      <td>420224.0</td>
      <td>1216.097805</td>
      <td>39.967936</td>
      <td>1059.45</td>
      <td>1187.54</td>
      <td>1213.83</td>
      <td>1242.79</td>
      <td>1393.54</td>
    </tr>
    <tr>
      <th>wv(m/s)</th>
      <td>420224.0</td>
      <td>1.700930</td>
      <td>65.472111</td>
      <td>-9999.00</td>
      <td>0.99</td>
      <td>1.76</td>
      <td>2.86</td>
      <td>28.49</td>
    </tr>
    <tr>
      <th>max.wv(m/s)</th>
      <td>420224.0</td>
      <td>3.054884</td>
      <td>69.043660</td>
      <td>-9999.00</td>
      <td>1.76</td>
      <td>2.96</td>
      <td>4.73</td>
      <td>23.50</td>
    </tr>
    <tr>
      <th>wd(deg)</th>
      <td>420224.0</td>
      <td>174.748064</td>
      <td>86.685323</td>
      <td>0.00</td>
      <td>124.90</td>
      <td>198.10</td>
      <td>234.10</td>
      <td>360.00</td>
    </tr>
  </tbody>
</table>
</div>



**Checking for null value and fixing data frequency**<br>
While dealing with time series data, the frequency of each time stamp is important. Fixing the frequency to a constant value can be done by the python's asfreq() method. <br>
Our data has been recorded at an interval of 10 minutes, everyday from 01/01/2009 to 01-01-2017.<br>
Before setting the frequency, lets check if there's any null value.<br>


```python
data.isna().sum()
```




    p(mbar)           0
    T(degC)           0
    Tpot(K)           0
    Tdew(degC)        0
    rh(%)             0
    VPmax(mbar)       0
    VPact(mbar)       0
    VPdef(mbar)       0
    sh(g/kg)          0
    H2OC(mmol/mol)    0
    rho(g/m**3)       0
    wv(m/s)           0
    max.wv(m/s)       0
    wd(deg)           0
    dtype: int64



So there is no null value. Lets set the time interval to 10 mins. If it inserts any extra instance, that can be filled with the following values (same approach can be attained to fill null values):
1. mean
2. median
3. the immediate before or after value
4. Any user defined value 

Here I will be using the immediate previous value. <br>
But before that, its better to ensure that the indices are all in ascending order. This can be done by sorting the dataframe in ascending order, specifically here as the index values are date-time values.


```python
data = data.sort_index(axis=0, level=None, ascending=True, inplace=False, kind='mergesort',
                na_position='last', sort_remaining=True)
```


```python
data = data.asfreq(freq='10min',method='ffill')
(data.head().T)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>DateTime</th>
      <th>2009-01-01 00:10:00</th>
      <th>2009-01-01 00:20:00</th>
      <th>2009-01-01 00:30:00</th>
      <th>2009-01-01 00:40:00</th>
      <th>2009-01-01 00:50:00</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>p(mbar)</th>
      <td>996.52</td>
      <td>996.57</td>
      <td>996.53</td>
      <td>996.51</td>
      <td>996.51</td>
    </tr>
    <tr>
      <th>T(degC)</th>
      <td>-8.02</td>
      <td>-8.41</td>
      <td>-8.51</td>
      <td>-8.31</td>
      <td>-8.27</td>
    </tr>
    <tr>
      <th>Tpot(K)</th>
      <td>265.40</td>
      <td>265.01</td>
      <td>264.91</td>
      <td>265.12</td>
      <td>265.15</td>
    </tr>
    <tr>
      <th>Tdew(degC)</th>
      <td>-8.90</td>
      <td>-9.28</td>
      <td>-9.31</td>
      <td>-9.07</td>
      <td>-9.04</td>
    </tr>
    <tr>
      <th>rh(%)</th>
      <td>93.30</td>
      <td>93.40</td>
      <td>93.90</td>
      <td>94.20</td>
      <td>94.10</td>
    </tr>
    <tr>
      <th>VPmax(mbar)</th>
      <td>3.33</td>
      <td>3.23</td>
      <td>3.21</td>
      <td>3.26</td>
      <td>3.27</td>
    </tr>
    <tr>
      <th>VPact(mbar)</th>
      <td>3.11</td>
      <td>3.02</td>
      <td>3.01</td>
      <td>3.07</td>
      <td>3.08</td>
    </tr>
    <tr>
      <th>VPdef(mbar)</th>
      <td>0.22</td>
      <td>0.21</td>
      <td>0.20</td>
      <td>0.19</td>
      <td>0.19</td>
    </tr>
    <tr>
      <th>sh(g/kg)</th>
      <td>1.94</td>
      <td>1.89</td>
      <td>1.88</td>
      <td>1.92</td>
      <td>1.92</td>
    </tr>
    <tr>
      <th>H2OC(mmol/mol)</th>
      <td>3.12</td>
      <td>3.03</td>
      <td>3.02</td>
      <td>3.08</td>
      <td>3.09</td>
    </tr>
    <tr>
      <th>rho(g/m**3)</th>
      <td>1307.75</td>
      <td>1309.80</td>
      <td>1310.24</td>
      <td>1309.19</td>
      <td>1309.00</td>
    </tr>
    <tr>
      <th>wv(m/s)</th>
      <td>1.03</td>
      <td>0.72</td>
      <td>0.19</td>
      <td>0.34</td>
      <td>0.32</td>
    </tr>
    <tr>
      <th>max.wv(m/s)</th>
      <td>1.75</td>
      <td>1.50</td>
      <td>0.63</td>
      <td>0.50</td>
      <td>0.63</td>
    </tr>
    <tr>
      <th>wd(deg)</th>
      <td>152.30</td>
      <td>136.10</td>
      <td>171.60</td>
      <td>198.00</td>
      <td>214.30</td>
    </tr>
  </tbody>
</table>
</div>



**PLOTTING THE DATA** <br>
Lets, plot the pressure attribute


```python
data['p(mbar)'].plot(figsize=(20,5))
plt.title('Time_vs_Pressure', fontsize=20, color='r')
plt.ylabel('Pressure',fontsize=15)
plt.xlabel('Date_Time',fontsize=15)
plt.show()
```


![png](/output_files/output_33_0.png)


**Q-Q Plot** <br>
A Q–Q plot is a probability plot, which is a graphical method for comparing two probability distributions by plotting their quantiles against each other.
To determine whether the data is distributed in a certain way. It usually shows how the data fits the normal distribution.
It plots all the values a variable contains and arranges in order, where the y-axis represent the ordered values from min to max. The x-axis shows the theoretical quantiles of the data i.e. how many standard deviations away from the mean these values are. <br>
Below is a Q-Q plot of the pressure variable.
The **Red line** shows how the data should have followed if they were normally distributed. Thus it shows the data is not normally distributed.


```python
import scipy.stats as ssplt
import pylab
```


```python
ssplt.probplot(data['p(mbar)'], plot=pylab)
pylab.show()
```


![png](/output_files/output_36_0.png)


# **Different kinds of time-series data**
1. Stationary 
2. Non-stationary

A stationary series is a stochastic process whose statistical properties such as mean, variance (the average degree to which each point differs from the average of all data points) and autocorrelation do not change over time. Hence, a non-stationary series is one whose statistical properties change over time because of presence of a certain trend.<br>
Types of stationary time-series:<br>
<t> a) **Strict Stationary:** In a series when samples of identical size have identical distributions, i.e. $Sample_1$ ($x_t$, $x_{t+k}$) and $Sample_2$ ($x_{t+e}$, $x_{t+e+k}$) have same distributions **(Dist($\mu$,$\sigma^2$))**. This is very restrictive and rarelyobserved in nature.<br>
<t> b) **Weak Stationary:** having a **constant mean** ($\mu$ = constant), **constant variance** ($\rho$=constant) and **consistent covariance** between periods at an identical distance from one another i.e. Cov($x_n$, $x_{n+k}$) = Cov($x_m$, $x_{m+k}$). Eg: **white noise**. <br>

Non-stationary data should be converted to stationary before applying any forecasting model, ny removing its underlying trend which is also a function of time.

**WHITE NOISE** <br>
A white noise is a sequence with no specific pattern. In case of time series, when the mean and the variance doesn't change over time, and there is no clear relationship between present and past values.<br>
Conditions for which a time series is a white noise: <br>
1. mean ($\mu$) = 0 (constant)
2. varaince ($\sigma$) = constant
3. Auto-correlation ($\rho$) = **cor $(x_t, x_{t-1})$ = 0** <br>(thus constant co-variance, since Cov($x_t$,$x_{t_1}$) = $\rho$($x_t$,$x_{t-1}$)*($\sigma_{x_t}$, $\sigma_{x_{t-1}}$) = 0  <br>

Thus white noise is a sequence of random data without having any periodicity, happens sporadically, and there's no scope of projecting it into the future. <br> NOTE: When the values of the sequence are drawn from from a Gaussian Distribution, then it is called Gaussian White Noise.
<br>
Lets, check if our pressure data is a white noise or not. In order to do so, we can create a white noise (random values from a normal distribution) having mean and variance of the pressure attribute.





```python
white_noise = np.random.normal(loc = data['p(mbar)'].mean(), #mean of the pressure attribute 
                               scale = data['p(mbar)'].std(), #standard deviation of pressure attribute
                               size = len(data))
```

Lets, compare the original pressure data with its white noise. For that its better to create a separate dataframe with these two time series.


```python
press_noise = pd.DataFrame(data['p(mbar)'])
press_noise['White_Noise'] = white_noise
press_noise.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>p(mbar)</th>
      <th>White_Noise</th>
    </tr>
    <tr>
      <th>DateTime</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2009-01-01 00:10:00</th>
      <td>996.52</td>
      <td>990.787520</td>
    </tr>
    <tr>
      <th>2009-01-01 00:20:00</th>
      <td>996.57</td>
      <td>991.901143</td>
    </tr>
    <tr>
      <th>2009-01-01 00:30:00</th>
      <td>996.53</td>
      <td>994.329886</td>
    </tr>
    <tr>
      <th>2009-01-01 00:40:00</th>
      <td>996.51</td>
      <td>992.027583</td>
    </tr>
    <tr>
      <th>2009-01-01 00:50:00</th>
      <td>996.51</td>
      <td>987.105276</td>
    </tr>
  </tbody>
</table>
</div>




```python
press_noise.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>p(mbar)</th>
      <th>White_Noise</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>420768.000000</td>
      <td>420768.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>989.218778</td>
      <td>989.214672</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.357717</td>
      <td>8.365454</td>
    </tr>
    <tr>
      <th>min</th>
      <td>913.600000</td>
      <td>950.854988</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>984.210000</td>
      <td>983.574355</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>989.590000</td>
      <td>989.212760</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>994.720000</td>
      <td>994.858965</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1015.350000</td>
      <td>1027.655144</td>
    </tr>
  </tbody>
</table>
</div>



We can see the newly generated white noise has similar mean and standard deviation with that of the pressure series. Plotting the graph of these series will further clarify the comparison.


```python
fig = plt.figure(figsize=(20,10))
plt.subplot(2,1,1)
plt.subplots_adjust(hspace = 0.3)

press_noise['White_Noise'].plot()
plt.title('Time_vs_White_Noise')
plt.xlabel('time')
plt.ylabel('Value')
plt.ylim(910, 1030)

plt.subplot(2,1,2)
press_noise['p(mbar)'].plot()
plt.title('Time_vs_Pressure')
plt.xlabel('time')
plt.ylabel('Value')

plt.show()
plt.savefig('noise_vs_data.png')
```


![png](/output_files/output_44_0.png)



    <Figure size 432x288 with 0 Axes>


As the series contains 420224 data points, the graph is clumpsy, plotting with fewer data points will solve the issue.


```python
fig = plt.figure(figsize=(20,10))
plt.subplot(2,1,1)
plt.subplots_adjust(hspace = 0.4)

press_noise['White_Noise'][:3000].plot()
plt.title('Time_vs_White_Noise')
plt.xlabel('time')
plt.ylabel('Value')
plt.ylim(955,1020)

plt.subplot(2,1,2)
press_noise['p(mbar)'][:3000].plot()
plt.title('Time_vs_Pressure')
plt.xlabel('time')
plt.ylabel('Value')
plt.ylim(955,1020)
plt.show()
plt.savefig('noise_vs_data_trunc.png')
```


![png](/output_files/output_46_0.png)



    <Figure size 432x288 with 0 Axes>


Now, it is easy to see the differences between the pattern of the two series. So there are smaller jumps between periods in the pressure value, since the values are not random. Thus to make an accurate forecast, the pattern has to be recognised.

