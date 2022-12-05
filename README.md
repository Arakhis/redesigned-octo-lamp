# LSTMCell-based Forex trader

Forex trading robot using FCSApi for data and MetaApi as MT5 server.
Use your own API key

## Disclaimer
This bot dont predict prices!!! It predicts trend for next hour.
This strategy have good negative Z-score, theoretically you can increase lot after each win (not tested).
I dont care about your losses ;)


## How it works
1. Preparing 3 years of data
2. Training model on data
3. Predict next hour price
4. Sleep for 1 hour
4. Parse new candle data
5. Re-train model on single batch
6. Predict next hour price
8. Calculate difference between previous and current predict
9. Open position based on difference
10. Profit
11. Continuing every hour...

## Best results
 Best results specs:
 - 1H timeframe
 - 20 epochs for first train
 - 8 batch size for first train
 - 5 window size for additional trainings
 
 Only forward tested!!!
 
 You can modify bot to use trend direction from daily timeframe and open positions on 1H timeframe.
 Also bot can be upgraded to use RL-networks to confirm predictions, that will increase winrate up to 90%.
 
## Dependencies:
- Python 3.8+
- Tensorflow
- TA-Lib
- Matplotlib
- metaapi-cloud-sdk
- Sklearn
