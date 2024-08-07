import pandas as pd
import numpy as np
from Models.data_preprocessor import DataPreprocessor
from Models.modeling import HydroModel
from tensorflow.keras.callbacks import ModelCheckpoint

df = pd.read_csv('crc_new_dataset.csv', sep=',', low_memory=False).fillna(np.nan)
df = df[['Datetime','levi_cr', 'Precipitation_carolyn_cr']]
df['levi_cr'] = pd.to_numeric(df['levi_cr'], errors='coerce')
df['Precipitation_carolyn_cr'] = pd.to_numeric(df['Precipitation_carolyn_cr'], errors='coerce')

data = DataPreprocessor('interpolation', 'continuous', '2021-04-01', '2021-11-30')
data.fill_data(df)
data.create_windows(20, 'levi_cr', 1, 1, 0)
data.separete_dataset()
hydro = HydroModel()
print(len(data.train_windows), len(data.train_labels))
hydro.build(np.array(data.train_windows).shape[1:], np.array(data.train_labels).shape[1], 64, 32, 32)
hydro.compile()
cp = ModelCheckpoint(f'test.keras', save_best_only=True)
hydro.train(data.train_windows, data.train_labels, (data.val_windows, data.val_labels), 20, [cp])