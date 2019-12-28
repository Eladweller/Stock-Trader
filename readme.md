# Using sentiment signals to improve technical analysis based stock trading
## Background
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

