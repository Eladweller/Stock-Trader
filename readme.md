# Using sentiment signals to improve technical analysis based stock trading
##
Background
<i>Technical analysis is a trading discipline employed to evaluate
investments and identify trading opportunities by analyzing statistical trends
gathered from trading activity, such as price movement and volume. Unlike
fundamental analysts, who attempt to evaluate a security's intrinsic value,
technical analysts focus on patterns of price movements, trading signals and
various other analytical charting tools to evaluate a security's strength or
weakness. </i>(Investopedia)
Because of its statistical nature, using only
technical analysis in trading will result in a lot of information that is
available for a human trader and that may, in the future, affect the stock's
value. Therefore, it is a common practice to use it along with external
information as a complimentary technique.<br>
The sentiment signals data was
supplied by Prof. Ronen Feldman and was gathered by analysing periodical Earning
Calls and identifing positive and negative words within these calls. By
aggregation of these positive and negative occurences, its possible to have a
notion of the general state of which the company is in to, hereby called
"sentiment signals".
Using both sets of tools may allow us to resemble the
information available to a human trader to a Python program desinged to trade in
stocks.
## Mission
Our mission is to trade using technical analysis and using
signal data and to see whether having extra set of information, like sentiment
signals, may potentially improve the return in comparison to using only
techincal analysis tools.
## Workplan
In order to see whether extra information
will improve our results, we first need to develop a technical analysis based
strategy. Then we will add the sentiment signal data on top of the strategy and
will see whether it is possible to improve the results.
### Trading
For
simplicity, we will use a stock by stock method and deal with each stock
individually. Furthermore, we will use the signals to open and then close each
position, hence, in each action we will either buy a specific stock for a sum of
100,000 USD or sell the entire holding.

### Evaluating
We will evaluate the
performance of our strategy based on the % of the succesful positions that were
opened and closed. If the percentage will go up after using the signals data, it
may indicate that extra information can improve against just using technical
analysis.

# Instructions
In order to start the program, put the
"DATA_signals.csv" into the same folder as this notebook and run all.
<br><b>WARNING:</b> It may take more than 2 days.

### Load Functions

```python
import pandas as pd
import numpy as np
import quandl
from datetime import datetime
import csv
import itertools
import random
import os
```

### Functions

```python
''' Converts a timestamp from string to datetime.srtptime object'''
def strToTime(string): 
    return datetime.strptime(str(string),'%Y-%m-%d %H:%M:%S')

def strToTime2(string): 
    return datetime.strptime(str(string),'%Y-%m-%d')

''' Aligns all datetime.strptime objects to same hour of the day'''
def RemoveHours(string):
    return datetime.strptime(str(string)[0:10],'%Y-%m-%d')

''' given the returns of the stocks, counts how many are within the bins defined'''
def count_freq(df):
    bins=[-5,-3,0,3,5]
    previous=0
    results=[]
    for num in bins:
        current=df[df<=num].count()-previous
        previous=current
        results.append(current)
    results.append(df[df>num].count())
#     print(results)
    return results
```

### Load Signals Data
In order to download the daily information regarding the
specific stocks that we have sentiment signals for, we will first load the
signals data provided by Ronen Feldman.

```python
signals=pd.read_csv("DATA_signals.csv")
temp=signals["Ticker"].str.split(":",expand = True)
signals["ticker"]=temp[1] 
signals["country"]=temp[0]
X=signals[(signals["country"]=="US")].dropna(subset=["cusip"]) #removing all non-US stocks
groups=X.groupby("ticker")["Date"]
signal_dates={} #getting all relevant days

```

### Get the distribution of returns in the periods after Earning Calls
By
looking at the distribution of returns, we can see that more companies have had
positive returns then negative returns.

```python
results=[]
for item in ["Return5 After","Return10 After","Return20 After","Return30 After","Return60 After"]:
    results.append(count_freq(signals.loc[:,item]))
print(pd.DataFrame(results, columns=["-5% and less","-5% to-3%","-3% to 0%","0% to 3%","3%-5%","5% and more"],index=[["Return5 After","Return10 After","Return20 After","Return30 After","Return60 After"]]))

```

### for all stocks get the relevant dates for the signals

```python
signal_range={}
for (name1,start),(name2,end) in zip(groups.min().iteritems(),groups.max().iteritems()):
    signal_range[name1]={"start":strToTime(start),"end":strToTime(end)}
print("Total number of symbols: {}".format(len(signal_range.keys())))

signal_dates=groups.unique().apply(lambda x: [strToTime(i) for i in x])
signal_dates_dict=signal_dates.to_dict()
```

### get the stocks data from Quandl

```python
if os.path.isfile("quandle_wiki.csv"):
    df=pd.read_csv("quandle_wiki.csv")
    loaded=set(df["ticker"].unique())
    df["date"]=df["date"].apply(strToTime2).apply(RemoveHours)
else:             
    quandl.ApiConfig.api_key = "auaWaFEjzyKC7myZg4a-"
    loaded=set()
    not_found=set()
    data=[]
    idx=0
    for ticker,dates in signal_range.items():
        getdata = quandl.get_table('WIKI/PRICES', ticker = ticker, 
                            date = { 'gte': dates["start"]-pd.to_timedelta(1,"y"),
                                    'lte': dates["end"]+ pd.to_timedelta(3,"m")}, paginate=True)
        if getdata.shape[0]==0:
            not_found.add(ticker)
        else:
            loaded.add(ticker)
            datatemp=getdata.loc[:,["ticker","date","adj_open","adj_high","adj_low","adj_close","adj_volume"]]
            data.append(datatemp)
        idx+=1
        if idx%500==0:
            print("gone over {0}/{1} {2} symbols were loaded, {3}  not found".format(idx,len(signal_dates.keys()),len(loaded),len(not_found)))
    df=pd.concat(data,ignore_index=True).sort_values(["ticker","date"])
    df.reset_index(drop=True)
    print("gone over {0}/{1} {2} symbols were loaded, {3}  not found".format(idx,len(signal_dates.keys()),len(loaded),len(not_found)))
    df.to_csv("quandle_wiki.csv",index_label=False)
df.info()
```

### filter the signal data to include only stocks found in Quandl

```python
signals_data=signals[signals["ticker"].isin(loaded)]
del signals_data["Ticker"]
del signals_data["cusip"]
del signals_data["ID"]
signals_data=signals_data.reset_index(drop=True)
```

### Arrange the signal data to be merged and adding aggscore1 and aggscore2 to
the DataFrame
#### Aggscore1 and Aggscore2 are an aggregation of the number of
positive and negative sentiment signals that are found in a single stock on a
single date, extracted from Earning Calls.
<br>$
\begin{align}
aggscore1 & =
\frac{Positive-Negative}{Positive+Negative+1}\
\end{align}
$
<br>
<br>$
\begin{align}
aggscore2 & = \frac{Positive}{Positive+Negative+1}\
\end{align}
$<br>

#### The SD scores are calculated per each aggregated score in order to
measure the sentiment on a timeframe basis and ascertaining a trend in the
sentiment of each company.
-  SD is defined as the change in sentiment score
from the previous period, it should reflect the short term change in the
sentiment of the stock.
<t><br><br>$
\begin{align}
SD & = SentimentScore_t-
SentimentScore_{t-1}
\end{align}
$<br><br>
-  SD4 is defined as the average
change in sentiment for the last 4 periods, at should reflect the medium term
change in the sentiment of the stock.
<t><br><br>$
\begin{align}
SD4 & =
SentimentScore_t-\frac{SentimentScore_{t-1}+SentimentScore_{t-2}+SentimentScore_{t-3}}3\
\end{align}
$<br>

```python
signals_data_small=signals_data.loc[:,["ticker","Date","Positives","Negatives"]].rename(columns={"Date":"date"}).sort_values(["date","ticker"])
signals_data_small["date"]=signals_data_small["date"].apply(RemoveHours)
signals_data_small=signals_data_small.groupby(by=["ticker","date"]).sum().reset_index(drop=False)
signals_data_small["aggscore1"]= (signals_data_small["Positives"]-signals_data_small["Negatives"])/(signals_data_small["Positives"]+signals_data_small["Negatives"]+1)
signals_data_small["aggscore2"]=(signals_data_small["Positives"])/(signals_data_small["Positives"]+signals_data_small["Negatives"]+1)

ToConcat=[]
for idx,ticker in enumerate(signals_data_small["ticker"].unique()):
#     print(ticker)
    temp=signals_data_small[signals_data_small["ticker"]==ticker].reset_index(drop=True)
    if temp.empty or temp.shape[0]<=4:
        continue
    temp=temp[4:]
    for agg in ["aggscore1","aggscore2"]:
        for i in range(1,5):
            temp["lag{}".format(i)]=temp[agg].shift(periods=i)
#         print(temp.shape)
        temp["ave4"]=temp.apply(lambda x: sum([x["lag{}".format(j)] for j in range(1,5)])/4,axis=1)
        temp["{}_SD".format(agg)]=temp[agg]-temp["lag1"]
        temp["{}_SD4".format(agg)]=temp[agg]-temp["ave4"]
        for i in range(1,5):
            del temp["lag{}".format(i)]
        del temp["ave4"]
    ToConcat.append(temp)
#     break
#     if idx%250==0:
#         print(idx)
signals_data_small=pd.concat(ToConcat,ignore_index=True)
signals_dates=signals_data_small.groupby("ticker")["date"].unique()
```

## Merge the Quandl data with the signals data

```python
ToConcat=[]
for idx,ticker in enumerate(signals_data_small["ticker"].unique()):
    tempS=signals_data_small[signals_data_small["ticker"]==ticker].sort_values("date").reset_index(drop=True)
    tempD=df[df["ticker"]==ticker].sort_values("date").reset_index(drop=True)
    ToConcat.append(pd.merge_asof(tempD,tempS,on="date",direction="backward"))
#     break
DATA=pd.concat(ToConcat,ignore_index=True)
del DATA["ticker_y"]
DATA=DATA.rename(columns={"ticker_x":"ticker"})
```

## Getting the Technical Analysis indicators
The indicators are calculated using
talib library. After offline trying of most indicators in the library, 6
indicators were chosen based on relative success with random stocks:

## MACD

```python
from talib import MACDFIX

def macd(close):
    macd, macdsignal, macdhist = MACDFIX(close)
    return series_crossover(macd,macdsignal)
```

### What is Moving Average Convergence Divergence (MACD)?

Moving Average
Convergence Divergence (MACD) is a trend-following momentum indicator that shows
the relationship between two moving averages of a security’s price. The MACD is
calculated by subtracting the 26-period Exponential Moving Average (EMA) from
the 12-period EMA. The result of that calculation is the MACD line. A nine-day
EMA of the MACD, called the "signal line," is then plotted on top of the MACD
line, which can function as a trigger for buy and sell signals. Traders may buy
the security when the MACD crosses above its signal line and sell - or short -
the security when the MACD crosses below the signal line. Moving Average
Convergence Divergence (MACD) indicators can be interpreted in several ways, but
the more common methods are crossovers, divergences, and rapid rises/falls.
MACD Line: (12-day EMA - 26-day EMA)

Signal Line: 9-day EMA of MACD Line
Strategy:

1) Use MACD crossover with its signal

 ## Aroon Oscillator

```python
from talib import AROONOSC

def aroonosc(high, low):
    return thresholds(AROONOSC(high,low),85,-85)
```

### What is the Aroon Oscillator

An Aroon Oscillator is a trend-following
indicator that uses aspects of the Aroon Indicator ("Aroon Up" and "Aroon Down")
to gauge the strength of a current trend and the likelihood that it will
continue. The Aroon Oscillator is calculated by subtracting Aroon Up from Aroon
Down. Readings above zero indicate that an uptrend is present, while readings
below zero indicate that a downtrend is present.

The Aroon Oscillator is a line
that can fall between -100 and 100. A high Oscillator value is an indication of
an uptrend while a low Oscillator value is an indication of a downtrend.
Strategy:

2) Use high and lows values of the Aroon Oscillator

## Balance Of Power

```python
from talib import BOP

def bop(Open,high,low,close):
    return thres_crossover(BOP(Open,high,low,close))
```

The Balance of Power is a simple indicator and it is used in technical analysis
to compare the strength of buyers vs. sellers. The BOP oscillates around zero
center line in the range from -1 to +1. Positive BOP reading is an indication of
buyers' dominance and negative BOP reading is a sign of the stronger selling
pressure. When BOP is equal zero it indicates that buyers and sellers are
equally strong.

Strategy:

3) Use the crossover of the BOP with 0.

## Commodity Channel Index

```python
from talib import CCI

def cci(high,low,close):
    return thresholds(CCI(high,low,close),100,-100)
```

 CCI measures the difference between a security's price change and its average
price change. High positive readings indicate that prices are well above their
average, which is a show of strength. Low negative readings indicate that prices
are well below their average, which is a show of weakness.

The Commodity
Channel Index (CCI) can be used as either a coincident or leading indicator. As
a coincident indicator, surges above +100 reflect strong price action that can
signal the start of an uptrend. Plunges below -100 reflect weak price action
that can signal the start of a downtrend. 

Strategy:

4) Use the CCI values
that are above or below 100 and -100

## Absolute Price Oscillator

```python
from talib import APO

def apo(close):
    return thres_crossover(APO(close,20,50))
```

### What is The Absolute Price Oscillator (APO)?
The Absolute Price Oscillator
displays the difference between two exponential moving averages of a security's
price and is expressed as an absolute value. It rates the trends strength in
relation to the moving between the two moving averages with short-term momentum
being the catalyst. The most popular setups are to use the 14-day and 30-day
EMAs, respectively.

All signals are generated by the signal line crossing above
zero-level which is considered bullish while crossing below zero-level is deemed
bearish. As short-term momentum increases or decreases and eclipses long-term
momentum the signal is generated. As is common with most oscillators, divergence
in the stock price and indicator can also alert investors to early turnarounds.
Strategy:

5) use the crossover of APO with 0

## Stochastic Relative Strength Index

```python
from talib import STOCHRSI

def stochrsi(close):
    fastk,fastd=STOCHRSI(close)
    return thresholds(fastk,80,20)    
```

### What Is the Stochastic RSI?

The Stochastic RSI (StochRSI) is an indicator
used in technical analysis that ranges between zero and one (or zero and 100 on
some charting platforms) and is created by applying the Stochastic oscillator
formula to a set of relative strength index (RSI) values rather than to standard
price data. Using RSI values within the Stochastic formula gives traders an idea
of whether the current RSI value is overbought or oversold.

The StochRSI
oscillator was developed to take advantage of both momentum indicators in order
to create a more sensitive indicator that is attuned to a specific security's
historical performance rather than a generalized analysis of price change.

A
StochRSI reading above 0.8 is considered overbought, while a reading below 0.2
is considered oversold. On the zero to 100 scale, above 80 is overbought, and
below 20 is oversold.

Strategy:

6) Use the STORSI values of above and below 80
and 20

### Time series functions
To extract the signals out of the technical analysis
functios, we defined few functions that returns the relevant signal. If function
returns 1 it means that we get a positive signal, if the function returns 0 it
means that there is no relevant signal and if it return -1 it means that a
negative signal was returned.

```python
'''gets two time-series, and returns 1 every time the first series crossed-over from below the second series
and returns -1 every time the first series crossed over the second series from above'''
def series_crossover(series1,series2):
    CO=[]
    c=series1-series2
    p=series1.shift(1)-series2.shift(1)
    for current,previous in zip(c,p):
        if current>0 and previous<0:
            CO.append(1)
            continue
        if current<0 and previous>0:
            CO.append(-1)
            continue
        CO.append(0)
    return pd.Series(CO)

'''gets a series and a threshold and returns 1 if the series crossed over from below the thresholdes
and -1 if the series crossed over from above the thresholder'''
def thres_crossover(series,threshold=0):
    CO=[]
    c=series-threshold
    p=series.shift(1)-threshold
    for current,previous in zip(c,p):
        if current>0 and previous<0:
            CO.append(1)
            continue
        if current<0 and previous>0:
            CO.append(-1)
            continue
        CO.append(0)
    return pd.Series(CO)

'''gets a series and 2 thresholds and returns 1 if the series
is above the higher threshold and -1 if the series is below the lower threshold'''
def thresholds(series,higher,lower):
    time_series=[]
    for item in series:
        if item > higher:
            time_series.append(1)
            continue
        if item < lower:
            time_series.append(-1)
            continue
        time_series.append(0)
    return pd.Series(time_series) 
```

### Appying the functions on the relevant indicators and appending the results
to the DataFrame

```python
ToConcat=[]
for idx,ticker in enumerate(DATA["ticker"].unique()):
    temp=DATA[DATA["ticker"]==ticker].reset_index(drop=True)
    temp["MACD"]=macd(temp["adj_close"])
    temp["AROONOSC"]=aroonosc(temp["adj_high"],temp["adj_low"])
    temp["BOP"]=bop(temp["adj_open"],temp["adj_high"],temp["adj_low"],temp["adj_close"])
    temp["CCI"]=cci(temp["adj_high"], temp["adj_low"], temp["adj_close"])
    temp["APO"]=apo(temp["adj_close"])
    temp["STOCHRSI"]=stochrsi(temp["adj_close"])
    ToConcat.append(temp)
DATA=pd.concat(ToConcat,ignore_index=True)
DATA.to_csv("merged_data.csv", index_label=False)
```

### Arranging the signal data
In order to have an understanding of the sentiment
signal data values, in relation to the rest of the market, we will trade based
on the relative position of the signals for the specific time. therefore, we
will divide the signals for 4 quartiles, and each stock will get a rating
between 1 and 4 for its relative position for the specific day. Later, we will
only buy stocks that are in the top quartile and sell only stocks that are in
the bottom one. It means that we will not sale or buy stocks that are not in the
2nd and 3rd quartiles.

```python
def get_quantiles(df,head,parts=4):
    subset=df.loc[:,["ticker",head]].sort_values(head,ascending=False).reset_index(drop=True).dropna()
    length=subset.shape[0]
    subset["quantiles"]=pd.qcut(subset[head],q=parts,labels=False,duplicates='drop')+1
    quantiles=subset.set_index("ticker").loc[:,["quantiles"]].to_dict()
    return quantiles["quantiles"]

ToConcat=[]
to_get_quntiles=['aggscore1','aggscore2','aggscore1_SD', 'aggscore1_SD4', 'aggscore2_SD', 'aggscore2_SD4']
for date in DATA["date"].unique():
    temp=DATA[DATA["date"]==date].reset_index(drop=True)
    for to_get in to_get_quntiles:
        quantiles=get_quantiles(temp,to_get)
        temp["Q_{}".format(to_get)]=temp["ticker"].map(quantiles)
    ToConcat.append(temp)
DATA=pd.concat(ToConcat,ignore_index=True).sort_values(["ticker","date"])
DATA.reset_index(drop=True).to_csv("merged_data.csv",index_label=False)
```

## Trading

Trading will be done per stock, and with 2 possible options per
stock per day, "open position" and "closed position".

### Trading functions

```python
'''Opens a position'''
def OpenPosition(account,price,amount,date):
    AmountToBuy=amount//price
    account["Holding"]=AmountToBuy
    account["Cash"]-=price*AmountToBuy
#     print("({}) bought {} stocks at {}".format(date,AmountToBuy,price))
    return account
'''Closes a position'''
def ClosePosition(account,price,date):
#     print("({}) sold {} stocks at {}".format(date,account["Holding"],price))
    account["Cash"]+=price*account["Holding"]
    account["Holding"]=0
    return account

def geo_mean(series):
    if len(series)==0:
        return 0
    a = np.array(series)
    return a.prod()**(1.0/len(a))

'''Trades the stocks based on the parameters given to it '''
def trade(param,stock,account):
    BZ=param[0]
    BC=param[1]
    SZ=param[2]
    SC=param[3]
#     print(BZ,BC,SZ,SC)
    Position=False
    PositionsGrade=[]
    exitValue=None
    entryValue=None
    PositionsGrade=[]

    for idx,row in stock.iterrows():
        Buy=False
        Sell=False
        BuyZone = (row["AROONOSC"]==1) + (row["CCI"]==1) + (row["STOCHRSI"]==1)
        SellZone = (row["AROONOSC"]==-1) + (row["CCI"]==-1) + (row["STOCHRSI"]==-1)
        SellCrossover= (row["MACD"]==-1) + (row["BOP"]==-1) + (row["APO"]==-1)
        BuyCrossover= (row["MACD"]==1) + (row["BOP"]==1) + (row["APO"]==1)
#         print(BuyZone,BuyCrossover,SellZone,SellCrossover)
        if ((BuyZone == BZ) and (BuyCrossover==BC)) and (Position is False):
#             print("opened position at {}".format(row["adj_close"]))
#             Positions=[row["date"],row["adj_close"],row["date"],row["AROONOSC"],row["CCI"],row["STOCHRSI"],row["MACD"],row["BOP"],row["APO"]]
            account=OpenPosition(account,row["adj_close"],account["Cash"],row["date"])
            Buy=True
            Position=True
            entryValue=row["adj_close"]
        if ((SellZone ==SZ) and (SellCrossover==SC)) and (Position is True):
#             print("closed position at {}".format(row["adj_close"]))
            account=ClosePosition(account,row["adj_close"],row["date"])
            Position=False
#             Positions.extend([row["date"],row["adj_close"],row["date"],row["AROONOSC"],row["CCI"],row["STOCHRSI"],row["MACD"],row["BOP"],row["APO"]])
            Sell=True
            exitValue=row["adj_close"]
            PositionsGrade.append([entryValue,exitValue])
#     print(PositionsGrade)
    return account,PositionsGrade

'''Calculates the success rate on the positions we had'''
def SuccessRate(series):
    count=0
    length=len(series)
    if length==0:
        return 0
    for i in series:
        if i[0]<i[1]:
            count+=1
    return count/length

'''Calculates the monitary value of the holding a given stock'''
def CalcValue(account,value):
    return account["Cash"]+account["Holding"]*account["Cash"]

'''Trades when given a sentiment and parameters'''
def tradeSentiment(param,stock,account,sentiment):
#     print(param)
    BZ=param[0]
    BC=param[1]
    SZ=param[2]
    SC=param[3]
    Position=False
    PositionsGrade=[]
    exitValue=None
    entryValue=None
    PositionsGrade=[]
    for idx,row in stock.iterrows():
        close=row["adj_close"]
        if sentiment==None:
            higherSentiment=True
            lowerSentiment=True
        else:
            higherSentiment=row[sentiment]==4
            lowerSentiment=row[sentiment]==1
        Buy=False
        Sell=False
        BuyZone = (row["AROONOSC"]==1) + (row["CCI"]==1) + (row["STOCHRSI"]==1)
        SellZone = (row["AROONOSC"]==-1) + (row["CCI"]==-1) + (row["STOCHRSI"]==-1)
        SellCrossover= (row["MACD"]==-1) + (row["BOP"]==-1) + (row["APO"]==-1)
        BuyCrossover= (row["MACD"]==1) + (row["BOP"]==1) + (row["APO"]==1)
#         print(higherSentiment)
#         print(BuyZone,BuyCrossover,SellZone,SellCrossover)
        if ((BuyZone == BZ) and (BuyCrossover==BC)) and (Position is False) and (higherSentiment==True):
#             print("opened position at {}".format(row["adj_close"]))
#             Positions=[row["date"],row["adj_close"],row["date"],row["AROONOSC"],row["CCI"],row["STOCHRSI"],row["MACD"],row["BOP"],row["APO"]]
            account=OpenPosition(account,row["adj_close"],account["Cash"],row["date"])
            Buy=True
            Position=True
            entryValue=row["adj_close"]
        if (((SellZone ==SZ) and (SellCrossover==SC)) and (lowerSentiment==True)) and (Position is True):
#             print("closed position at {}".format(row["adj_close"]))
            account=ClosePosition(account,row["adj_close"],row["date"])
            Position=False
#             Positions.extend([row["date"],row["adj_close"],row["date"],row["AROONOSC"],row["CCI"],row["STOCHRSI"],row["MACD"],row["BOP"],row["APO"]])
            Sell=True
            exitValue=row["adj_close"]
            PositionsGrade.append(exitValue/entryValue)
#     print("{0:.0f}, {1:.0f}".format(CalcValue(account,row["adj_close"]),stock["adj_close"].iloc[-1]/stock["adj_close"].iloc[0]*100000))
    ratio=sum([i>1 for i in PositionsGrade])
    try:
        return CalcValue(account,row["adj_close"]),stock["adj_close"].iloc[-1]/stock["adj_close"].iloc[0]*100000,len(PositionsGrade),ratio
    except:
        return 0,0,0,0
```

## Identifing the best parameters
In order to get the best parameters to later
by applied our model for the full dataset, we will divide the indicators into 2
groups, 'buy' and 'sell' signals, with crossovers and general zones as
subgroups. 
<br><b> Buying and selling indicators:</b>
- MACD
- BOP
- APO
<b>Buying and selling zones</b>
- AROONOSC
- CCI
- STORSI

We will sum-up the
Buying and selling stocks and zone seprately, and get 4 parameters that
represent the 6 techical analysis indicators. Then we will try different values
for the 4 parameters, ranging from 0 (no indicator is relevant) to 3 (all
indicators are relevant).
In order to find the best possible parameters, we will
try all in 10 iterations, each time with 5 different random selected stocks.
After that, Best performing parameters will be applied to our full model with
the sentiment data.

### defining initial parameters

```python
DATA=pd.read_csv("merged_data.csv")
stock_list=DATA["ticker"].unique().tolist()
timer=0
stocks=random.sample(stock_list,len(stock_list))
```

```python
cash=100000
params_performance=[]
Performance=[]
time1=datetime.now()
DATA.sort_values(["ticker","date"])
for check in range(0,10):
    params=itertools.product([2,1,3,0],repeat=4)
    stocks=random.sample(stock_list,len(stock_list))
    for loop,param in enumerate(params,1):
        if sum(param)<2:
            continue
        FinalValues=[]
        time2=datetime.now()
        Performance=[]
        cash=100000
        Positions=[]
        for idxs,CurrentStock in enumerate(stocks):
            account={"Holding":0,"Cash":cash}
            stock=DATA[DATA["ticker"]==CurrentStock].dropna(subset=["Positives"]).sort_values("date").reset_index(drop=True)
            account,PositionsGrade=trade(param,stock,account)
            SRate=SuccessRate(PositionsGrade)
            Positions.append(len(PositionsGrade))
#             print(PositionsGrade)
#             print(len(PositionsGrade))
            try:
                Performance_temp=([CurrentStock,
                                   (CalcValue(account,stock["adj_close"].iloc[-1]))/100000,
                                   stock["adj_close"].iloc[-1]/stock["adj_close"].iloc[0],
                                    SRate,len(PositionsGrade)])
                Performance.append(Performance_temp)
                FinalValues.append(CalcValue(account,stock["adj_close"].iloc[-1]))
            except:
                pass
                print("failed")
                print(account.get("Holding"))
                print(stock["adj_close"].iloc[-1])
                print()
                      
            if idxs==5:
                break
#         print(Performance)
        results=pd.DataFrame(Performance,columns=["ticker","strategy_return","stock_return","Succssess_rate","count_positions"])
        results["ratio"]=results["strategy_return"]/results["stock_return"]
        results.to_csv("performance/{}_{} results.csv".format(check,param),index_label=False)
        print("{0})with parameters {1}, Performace: {2:.2f}, mPos:{3:.2f},SRate:{4:.2f} time:{5}, total_runtime{6}"
              .format(loop,param,geo_mean(results["ratio"]),results["count_positions"].mean(),SRate
                      ,datetime.now()-time2,datetime.now()-time1))
        if len(FinalValues)==0:
            final=0
        else:
            final=sum(FinalValues)/len(FinalValues)
        params_performance.append([check,param,geo_mean(results["ratio"]),final,sum(Positions)])
#         if loop>=1:
#             break
    performance_check=pd.DataFrame(params_performance,columns=["round","params","results","final_values","Positions"])
performance_check.to_csv("PerformanceCheck.csv",index_label=False)
```

```python
# performance_check=pd.read_csv("PerformanceCheck.csv")
candidates=performance_check.groupby("params",as_index=False)["params","results","Positions","final_values"].mean().sort_values(by="results",ascending=False)
parameters1=[[int(j) for j in i[1:-1].split(",")] for i in candidates[candidates["Positions"]>=75]["params"].tolist()]
parameters=[]
for i in parameters1:
    if i[2]==0 and i[3]==0:
        continue
    else:
        parameters.append(i)
    if len(parameters)==5:
        break
parameters
```

## Applying the model
After extracting 5 candidates from our limited model, we
can add the sentiment signals that we assembled, and try all different
combination in order to see which one will give us the best results.

```python
stock_list=DATA["ticker"].unique().tolist()
sentiments=[None,"Q_aggscore1","Q_aggscore2","Q_aggscore1_SD","Q_aggscore1_SD4","Q_aggscore2_SD","Q_aggscore2_SD4"]
cash=100000
params_performance=[]
time1=datetime.now()
ToConcat=[]
DATA.sort_values(["ticker","date"])
iterations=0
for sentiment in sentiments:
    params=parameters
    stocks=random.sample(stock_list,len(stock_list))
    for loop,param in enumerate(params,1):
        tradePerformance=[]
        stockPerformance=[]
        lenPosition=[]
        successfullPositions=[]
        cash=100000
        time2=datetime.now()
        print("Sentiment index is: {}, Parameters are: {}, Elapsed time: {}, Elapsed iterations: {}".
              format(sentiment,param,datetime.now()-time1,iterations))
        for idxs,CurrentStock in enumerate(stocks):
            iterations+=1
            account={"Holding":0,"Cash":cash}
            stock=DATA[DATA["ticker"]==CurrentStock].dropna(subset=["Positives"]).sort_values("date").reset_index(drop=True)
            tP,sP,lP,rP=tradeSentiment(param,stock,account,sentiment)
            tradePerformance.append(tP)
            stockPerformance.append(sP)
            lenPosition.append(lP)
            successfullPositions.append(rP)
            if idxs%500==0:
                pass
#                 print("{}) Elapsed time: {}, Elapsed iterations: {}".format(idxs,datetime.now()-time1,iterations))
        params_performance.append([sentiment,param,sum(tradePerformance)/sum(stockPerformance),sum(lenPosition),sum(successfullPositions),datetime.now()-time2])
#         if loop==3:
#             break
#     break
print("Total Time: {}, # of iterations: {}".format(datetime.now()-time1,iterations))
results=pd.DataFrame(params_performance,columns=["sentiment","params","performance","positions","successfull","time"])
results.to_csv("final_results.csv",index_label=False)
results["ratio"]=results.successfull/results.positions
```

## Results
The results dataframe shows, per sentiment index and parameters, what
was the performance (relating to the buying every stock in the beginning of the
period and selling it on the last day of the period) and the success ratio of
each position taken. The success ratio is defined whether the specific position
had a positive return.
<br>From looking at the top performing parameters and
sentiments, it can be seen that:
- Using the sentiment data improves the
results, both on the "performance" and the "ratio" indicators.
- Adding another
condition to enter and exit a position results in much less positions but leads
to better performance and much better success ratio.
- Using time based
sentiment indices (such as SD and SD4) improves both performance and success
ratio compared to a single timeframe sentiment.

## Conclusion
#### This project
has shown that adding a sentiment to a purely technical analysis may improve
results.
Although the best performing indicators have supplied us with at most
50% success over the market performance, it can be a result of lacking Technical
Analysis indicators. However, given better performing technical analysis
technique and/or indicators, we saw that using sentiment analysis <b>can
possibly</b> improve result compared to baseline performance.
The fact that our
best perfomance indicators have managed to get a success ratio of above 60%
shows that the rational behind entering positions was solid, but not enough
positions were opened, resulting in only 50% performance compared to the market.
It can also be possible given better indicators for entering to a position. The
fact that a portfolio-based approach wasn't taken, and that positions can only
be opened or closed (hence, cannot be expanded or narrowed) possibly minimized
that possible return out of successful signals, given a success rate of above
50%.

### Results by success ratio

```python
results.sort_values("ratio",ascending=False)
```

### Results by performance

```python
results.sort_values("performance",ascending=False)
```
