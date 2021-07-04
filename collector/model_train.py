seed_distribution = 10
import os
os.environ['PYTHONHASHSEED'] = str(seed_distribution)
import tensorflow as tf
import random
import pandas as pd
import numpy as np

tf.random.set_seed(seed_distribution)
np.random.seed(seed_distribution)
random.seed(seed_distribution)
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers import BatchNormalization
from selenium import webdriver
import glob
import time
import pickle
import sqlite3


def selenium_func():
    """
    The function to download data using Selenium
    """
    sed = 10
    os.environ['PYTHONHASHSEED'] = str(sed)
    tf.random.set_seed(sed)
    np.random.seed(sed)
    random.seed(sed)
    brouser = webdriver.Chrome("C:\\сhromdriverr\chromedriver.exe")
    url = "https://catalog.data.gov/dataset?q=Family_Home_Sale_Prices_by_Census_" \
          "Tract&sort=score+desc%2C+name+asc&as_sfid=AAAAAAWg7-Jeo4iYCBnxS_hCDyRGhL" \
          "Mtj97XuEWCanXLfcAmiPhlx_BLirMjorXjXtjR7QVj9cd8KE8_lNiaabQRWeXZhZ5ThE1nX4-8JoKjttoj1Imt0I6cb" \
          "oVZh7t2BcWZSUg%3D&as_fid=518deb3b8ebe1f62e1b6e0e164b24eadd0f754a1"
    brouser.get(url)
    time.sleep(5)
    xpath = f'//*[@id="content"]/div[2]/div/section[1]/div[2]/ul/li[1]/div/ul/li[4]/a'
    brouser.find_element_by_xpath(xpath).click()
    time.sleep(20)
    brouser.close()
    time.sleep(5)


def etl():
    """
    The function to clean the data and save it to the SQL data base
    """
    global conn
    df = pd.read_csv(r"C:\Users\Denis\Downloads\Single-Family_Home_Sale_Prices_by_Census_Tract.csv")
    df["COST"].replace({'High': 0, 'Medium': 2, 'Low': 1, 'Other': -1}, inplace=True)
    df["CHANGE"].replace({'Significant Rise': 2, 'Rise': 1, 'Stable': 3, 'Decline': 0, 'Other': -1}, inplace=True)
    df = df.apply(lambda x: x.fillna(x.mean()), axis=0)
    df = df[['PRICE_SQFT', 'PRICE_NOM', 'CHANGE', 'SALE', 'COST', 'YEAR', 'OBJECTID',
             'CHANGE_SQFT', 'CHANGE_NOM', 'Shape__Length', 'Shape__Area', 'TRACT']]
    df.drop("OBJECTID", inplace=True, axis=1)
    df.drop("Shape__Area", inplace=True, axis=1)
    df.drop("Shape__Length", inplace=True, axis=1)
    df.drop("CHANGE_SQFT", inplace=True, axis=1)
    conn = sqlite3.connect(r"C:\Users\Denis\Documents\pythonProject8\Fixtures"
                           r"\serverSQL_project8.sqlite")
    time.sleep(5)
    df.to_sql('SQL_project8', conn, if_exists='replace', index=False)


def train_the_model():
    """
    The function to read the data from SQl data base, train LSTM model,
    save the model and StandardScaler parameters
    """
    df = pd.read_sql('select * from SQL_project8', conn)
    x = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    x = np.array(x)
    y = np.array(y)
    sc_x = StandardScaler()
    x = sc_x.fit_transform(np.array(x))
    pickle.dump(sc_x, open(r"C:\Users\Denis\Documents\pythonProject8\Fixtures\scaler.pkl", 'wb'))
    y = y / 100
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = Sequential()
    model.add(LSTM(30, input_shape=(x.shape[1], 1), return_sequences=True, recurrent_dropout=0.25))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(LSTM(30, return_sequences=True, recurrent_dropout=0.25))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(LSTM(30, return_sequences=False, recurrent_dropout=0.25))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss="mse", metrics=['mse'])
    EPOCHS = 200
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=64, verbose=0)
    model.save(r"C:\Users\Denis\Documents\pythonProject8\Fixtures\LSTM_model.hdf5")
    y_pred_test = model.predict(X_test)
    y_test = y_test * 100
    y_pred_test = y_pred_test * 100
    print('Score is :', metrics.r2_score(y_test, y_pred_test).round(2), "%")

    files = glob.glob(r'C:\Users\Denis\Downloads\*')
    for f in files:
        os.remove(f)


if __name__ == '__main__':
    seconds_per_week = 700000
    seconds_per_day = 86400
    while True:
        try:
            selenium_func()
            etl()
            train_the_model()
            time.sleep(seconds_per_week)
        except:
            time.sleep(seconds_per_day)
