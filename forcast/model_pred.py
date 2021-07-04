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
from tensorflow.keras.models import load_model
import pickle


def prediction(User_parameters):
    """
    The function receives data from the user and returns the result of prediction
    """
    model = load_model(r"C:\Users\Denis\Documents\pythonProject8\Fixtures\LSTM_model.hdf5")
    df_advance = User_parameters
    labels = ['PRICE_NOM', 'CHANGE', 'SALE', 'COST', 'YEAR',
              'CHANGE_NOM', 'TRACT']
    df = pd.DataFrame(df_advance, columns=labels)
    sc_x = pickle.load(open(r"C:\Users\Denis\Documents\pythonProject8\Fixtures\scaler.pkl", 'rb'))
    df = np.array(df)
    x = sc_x.transform(df)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    res = model.predict(x) * 100
    res = res[0][0]
    return round(res)

if __name__ == '__main__':
    prediction()
