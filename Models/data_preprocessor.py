from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    def __init__(self, fill_nan_method, splitdf_method, start_date, end_date):
        self.data = None
        self.dates = None
        self.windows = None
        self.labels = None
        self.train_dates = None
        self.train_windows = None
        self.train_labels = None
        self.test_dates = None
        self.test_windows = None
        self.test_labels = None
        self.val_dates = None
        self.val_windows = None
        self.val_labels = None
        self.fill_nan_method = fill_nan_method
        self.splitdf_method = splitdf_method
        self.start_date = start_date
        self.end_date = end_date

    @property
    def fill_nan_method(self) -> str:
        return self._fill_nan_method
    
    @fill_nan_method.setter
    def fill_nan_method(self, value: str) -> None:
        if value not in ['interpolation', 'fill0', 'equal_dis']:
            raise ValueError('Wrong method')
        self._fill_nan_method = value

    @property
    def splitdf_method(self) -> str:
        return self._splitdf_method
    
    @splitdf_method.setter
    def splitdf_method(self, value: str) -> None:
        if value not in ['striped', 'continuous']:
            raise ValueError('Wrong method')
        self._splitdf_method = value

    @property
    def start_date(self) -> str:
        return self._start_date
    
    @start_date.setter
    def start_date(self, value: str) -> None:
        if not value:
            raise ValueError('Wrong date')
        self._start_date = datetime.strptime(value, "%Y-%m-%d")

    @property
    def end_date(self) -> str:
        return self._end_date
    
    @end_date.setter
    def end_date(self, value: str) -> None:
        if not value:
            raise ValueError('Wrong date')
        self._end_date = datetime.strptime(value, "%Y-%m-%d")

    def fill_data(self, df):
        # Normalize datetimes
        data_df = df.copy()
        data_df['Datetime'] = pd.to_datetime(data_df['Datetime'])
        df_dates = data_df.pop('Datetime')
        data_df.index = df_dates

        # Interested years
        data_df = data_df[(data_df.index >= self._start_date) & (data_df.index <= self.end_date)]
        data_df = data_df[(data_df.index.month >= 4) & (data_df.index.month <= 11)]

        # Fill missing values
        if self.fill_nan_method == 'interpolation':
            data_df.interpolate(inplace=True)
        elif self.fill_nan_method == 'fill0':
            data_df['Precipitation_carolyn_cr'] = data_df['Precipitation_carolyn_cr'].fillna(0)
        elif self.fill_nan_method == 'equal_dis':
            for i in range(4, len(data_df), 4):
                value_prec = data_df.iloc[i]['Precipitation_carolyn_cr'] / 4
                data_df.iloc[i-3:i+1, data_df.columns.get_loc('Precipitation_carolyn_cr')] = value_prec
            data_df = data_df.fillna(0)

        # Normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_scaled = scaler.fit_transform(data_df)
        df_scaled = pd.DataFrame(df_scaled, columns=data_df.columns)
        df_scaled.index = data_df.index
        data_df = df_scaled
        self.data = data_df.copy()
        data_df.to_csv('teste.csv', sep=',', index=False)
        return data_df

    def create_windows(self, window_size, target, shift=1, label_size=1, steps_future=0):
        windows = []
        labels = []
        dates = []
        for start in range(0, len(self.data) - window_size + 1, shift):
            end = start + window_size
            if(end+steps_future > len(self.data)):
                break
            window = self.data.iloc[start:end-label_size]
            date = window.index
            window = window.to_numpy()
            label = self.data.iloc[end-label_size+steps_future:end+steps_future][target].to_numpy()
            windows.append(window)
            labels.append(label)
            dates.append(date)
        self.dates = dates
        self.windows = windows
        self.labels = labels

    def strided_dataset(dates, windows, labels, train_size, test_size, val_size):
        dates_train, dates_test, dates_val = [], [], []
        windows_train, windows_test, windows_val = [], [], []
        labels_train, labels_test, labels_val = [], [], []

        total_size = train_size + test_size + val_size
        num_windows = len(windows) // total_size

        for i in range(num_windows):
            start = i * total_size
            windows_train.extend(windows[start:start+train_size])
            windows_test.extend(windows[start + train_size:start+train_size+test_size])
            windows_val.extend(windows[start+train_size+test_size:start+total_size])

            dates_train.extend(dates[start:start+train_size])
            dates_test.extend(dates[start + train_size:start+train_size+test_size])
            dates_val.extend(dates[start+train_size+test_size:start+total_size])

            labels_train.extend(labels[start:start+train_size])
            labels_test.extend(labels[start + train_size:start+train_size+test_size])
            labels_val.extend(labels[start+train_size+test_size:start+total_size])

        return dates_train, windows_train, labels_train, dates_test, windows_test, labels_test, dates_val, windows_val, labels_val

    def separete_dataset(self, train_size=0.8, test_size=0.1, val_size=0.1, train_strip=10, test_strip=2, val_strip=2):
        if self.splitdf_method == 'continuous':
            self.train_windows = np.array(self.windows[:int(len(self.windows)*train_size)])
            self.train_labels = np.array(self.labels[:int(len(self.labels)*train_size)]) 
            self.train_dates = np.array(self.dates[:int(len(self.dates)*train_size)]) 
            self.test_windows = np.array(self.windows[:int(len(self.windows)*test_size)]) 
            self.test_labels = np.array(self.labels[:int(len(self.labels)*test_size)]) 
            self.test_dates = np.array(self.dates[:int(len(self.dates)*test_size)]) 
            self.val_windows = np.array(self.windows[:int(len(self.windows)*val_size)]) 
            self.val_labels = np.array(self.labels[:int(len(self.labels)*val_size)])
            self.val_dates = np.array(self.dates[:int(len(self.dates)*val_size)])

        else:
            result = self.strided_dataset(self.dates, self.windows, self.labels, train_strip, test_strip, val_strip)
            self.train_dates = np.array(result[0])
            self.train_windows = np.array(result[1])
            self.train_labels = np.array(result[2])
            self.test_dates = np.array(result[3])
            self.test_windows = np.array(result[4])
            self.test_labels = np.array(result[5])
            self.val_dates = np.array(result[6])
            self.val_windows = np.array(result[7])
            self.val_labels = np.array(result[8])
    
