# LSTM-based Forex trader

Forex trading robot using FSCApi for data and MetaApi for MT5 server.
Use your own API key

How it works?

1. Preapring 3 years of data
2. Training model on data
3. Predict next hour price
4. Sleep for 1 hour
4. Parse new candle data
5. Re-train model on single batch
6. Predict next hour price
8. Calculate difference between previous and current predict
9. Open position based on difference
10. Profit

Dependencies:
Python 3.8+
Tensorflow
TA-Lib
Matplotlib
metaapi-cloud-sdk
Sklearn
