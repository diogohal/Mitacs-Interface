import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

class HydroModel:
  def __init__(self):
    self.model = None

  def build(self, input_shape, output_shape, layers):
    self.model = Sequential()
    self.model.add(InputLayer(shape=input_shape))
    
    for layer in layers:
      if layer[0] == 'lstm':
        self.model.add(LSTM(layer[1]))
      else:
        self.model.add(Dense(layer[1], activation='relu'))
    self.model.add(Dense(output_shape))    
    

  def compile(self):
    self.model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

  def train(self, x, y, val, epochs, callbacks):
    self.model.fit(x, y, validation_data=val, epochs=epochs, callbacks=callbacks)

  def predict(self, x):
    return self.model.predict(x)