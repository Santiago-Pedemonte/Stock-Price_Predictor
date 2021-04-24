# Stock-Price Predictor
A simple predictor that runs on Jupyter Notebook to gather and predict the prices of stocks (also works for cryptocurrencies supported by the Yahoo Finance API) 

### Necessary Imports

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import pandas_datareader as web
    import datetime as dt
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, LSTM

### Setup
First thing is to select the tickers and set a start & end dates:
    
    tickers = ['ETH-USD']
    start = dt.datetime(2012,1,1)
    end = dt.datetime(2020,1,1)
    data = web.DataReader(tickers, 'yahoo', start, end)

In this case, our only ticker will be gathering the price of ETH in terms of USD from Jan. 2012 - Jan. 2020

### Prediction Days
Set the number of days the network should look into the past at one time to predict the following price (default = 60):

    prediction_days = 60 
    
### Build the model
Can change the number of epochs, layers, units per layer (except the last one which should be 1, it takes the value of the price for the following day), and batch_size to modify training.

Defaults:

    epochs = 50
    batch_size = 64
    units = 50

### Test the model
Now gather data not available to the model. In our case, that would be from Jan. 2020 onwards:

    test_start = dt.datetime(2020,1,1)  # Gather data not available to the model
    test_end = dt.datetime.now()
    test_data = web.DataReader(tickers, 'yahoo', test_start, test_end)
    
### Predictions and plot
Now we can make the price prediction for the next day and plot the results. To get the concrete price from the output neuron's value, we need to inverse scale the prediction:

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    print("Prediction:", prediction)
