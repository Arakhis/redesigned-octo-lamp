import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot
import mt_part
from time import sleep
import talib
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential, layers
tf.get_logger().setLevel('ERROR')


    ########################### SETTING VARIABLES ###########################


url = 'https://fcsapi.com/api-v3/forex/history'
api_key = 'foLmNoRHQHs4VrIBVXutMH'
symbol = 'XAU/USD'
period = '1h'
level = '3'
resultser = []

params = f'symbol={symbol}&' \
         f'period={period}&' \
         f'level={level}&' \
         f'access_key={api_key}'

frame = pd.DataFrame()
time = []
open = []
high = []
low = []
close = []

    ########################### PARSING DATA ###########################

raw_data = requests.get(url, params).json()

for x in raw_data['response']:
        time.append(raw_data['response'][x]['tm'])
        open.append(raw_data['response'][x]['o'])
        high.append(raw_data['response'][x]['h'])
        low.append(raw_data['response'][x]['l'])
        close.append(raw_data['response'][x]['c'])

frame = frame.assign(Date=time, Open=open, High=high, Low=low, Close=close)

    ########################### PREPARING DATA ###########################

def talibate(frame):
    frame['RSI'] = talib.RSI(frame['Close'], timeperiod=14)
    frame['upperBB'], frame['middleBB'], frame['lowerBB'] = talib.BBANDS(frame['Close'], timeperiod=21, nbdevup=2,
                                                                         nbdevdn=2, matype=0)
    frame['STOCH_K'], frame['STOCH_D'] = talib.STOCH(frame['High'], frame['Low'], frame['Close'], 14, 3, 0, 3, 0)
    frame['CCI'] = talib.CCI(frame['High'], frame['Low'], frame['Close'], timeperiod=14)
    frame['EMA'] = talib.EMA(frame['Close'], timeperiod=14)
    frame['MACD'], frame['MACD_SIGNAL'], frame['MACD_HIST'] = talib.MACD(frame['Close'], 12, 26, 9)
    frame['ROC'] = talib.ROC(frame['Close'], 12)
    frame['STOCHRSI_K'], frame['STOCHRSI_D'] = talib.STOCH(frame['RSI'], frame['RSI'], frame['RSI'], 14, 3, 0, 3, 0)
    frame['WILLR'] = talib.WILLR(frame['High'], frame['Low'], frame['Close'], 14)
    frame.dropna(inplace=True)
    frame.reset_index(inplace=True)
    frame.drop(['Date', 'High', 'Low', 'Open'], axis=1, inplace=True)
    frame['Close'] = frame['Close'].astype('float')
    frame['Close'] = frame['Close'].shift(-1) ### Shifting value by 1 day
    return frame

frame['Date'] = pd.to_datetime(frame['Date'])
frame = frame.set_index('Date')
raw_frame = frame.copy()
frame = talibate(frame)

    ########################### DATA PREPROCESSING ###########################

scaler = MinMaxScaler(feature_range=(0, 1))
pred_scaler = MinMaxScaler(feature_range=(0, 1))


def scaleData(frame):
    pred_scaler.fit(frame['Close'].values.reshape(len(frame['Close']), 1))
    for column in frame.columns:
        frame[column] = scaler.fit_transform(frame[column].values.reshape(len(frame[column]), 1))
    label = frame.dropna()
    features = frame.drop('Close', axis=1)
    to_predict_value = features.values[-1].reshape(1, 15)
    features = features.iloc[0:-1, :]
    return label, features, to_predict_value


label, features, to_predict_value = scaleData(frame)
label = label['Close'].iloc[0:].values.reshape(len(label['Close'].iloc[0:]), 1)
features = features.values

    ########################### DEFINE MODEL ###########################


model = Sequential(
        [
            layers.RNN(layers.LSTMCell(34), return_sequences=True, input_shape=(15, 1)),
            layers.Dropout(0.05),
            layers.RNN(layers.LSTMCell(34)),
            layers.Dense(1)
        ]
    )
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse', metrics=['mape'])

    ########################### MAIN MODEL TRAINING ###########################


model.fit(features, label, epochs=10, batch_size=12, shuffle=False)


    ########################### PREPARING EXT BATCH ###########################

def prepareBatch(s_frame):
    url = 'https://fcsapi.com/api-v3/forex/candle'
    params = f'symbol={symbol}&' \
             f'period={period}&' \
             f'access_key={api_key}&'\
             f'candle=close'
    time = []
    open = []
    high = []
    low = []
    close = []
    n_frame = pd.DataFrame()
    raw_data = requests.get(url, params).json()

    time.append(raw_data['response'][0]['tm'])
    open.append(raw_data['response'][0]['o'])
    high.append(raw_data['response'][0]['h'])
    low.append(raw_data['response'][0]['l'])
    close.append(raw_data['response'][0]['c'])

    n_frame = n_frame.assign(Date=time, Open=open, High=high, Low=low, Close=close)
    n_frame['Date'] = pd.to_datetime(n_frame['Date'])
    s_frame = pd.concat([s_frame, n_frame])
    new_frame = talibate(s_frame.copy())
    label, features, to_predict_value = scaleData(new_frame)
    features = features.iloc[-5:, :].values
    return label, features, to_predict_value, s_frame


def trainBatch(model, label, features):
    model.train_on_batch(features, label)
    return model

def createPlot(value1, value2):
    pyplot.clf()
    pyplot.title('Real-time 1H predicts')
    pyplot.xlabel('Hour')
    pyplot.ylabel('Price')
    pyplot.plot(value1, label='Preds')
    pyplot.plot(value2, label='Reals')
    pyplot.legend()


    ########################### MAIN LOOP ###########################

preds = [pred_scaler.inverse_transform(model.predict(to_predict_value))[0][0]]
reals = [pred_scaler.inverse_transform(label[-1].reshape(-1,1))[0][0]]
createPlot(preds, reals)

tr = 0


while tr == 0:
    sleep(3600)
    label, features, to_predict_value, raw_frame = prepareBatch(raw_frame)
    label = label['Close'].iloc[-5:].values.reshape(len(label['Close'].iloc[-5:]), 1)
    trainBatch(model, label, features)
    preds.append(pred_scaler.inverse_transform(model.predict(to_predict_value))[0][0])
    reals.append(pred_scaler.inverse_transform(label[-1].reshape(-1,1))[0][0])

    if len(preds) > 24:
        preds.pop(0)
        reals.pop(0)
    if preds[-1] > preds[-2]:
        mt_part.make_order('Buy')
    elif preds[-1] < preds[-2]:
        mt_part.make_order('Sell')
    else:
        pass

    createPlot(preds, reals)



