from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from Models.data_preprocessor import DataPreprocessor
from Models.modeling import HydroModel
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.callbacks import ModelCheckpoint

app = Flask(__name__)

@app.route('/')
def index():
    global train_list
    return render_template('index.html', qt_models=len(train_list))


@app.route('/configure', methods=['POST'])
def configure():
    global data
    global train_list

    if request.method == 'POST':
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        method = request.form.get('method')
        splitdf = request.form.get('splitdf')
        window_size = int(request.form.get('window_size'))
        label_size = int(request.form.get('label_size'))
        shift = int(request.form.get('shift'))
        steps_future = int(request.form.get('steps_future'))
        
        data = DataPreprocessor(method, splitdf, start_date, end_date)
        data.fill_data(df)
        data.create_windows(window_size, 'levi_cr', shift, label_size, steps_future)
        data.separete_dataset()
        return render_template('index.html', qt_models=len(train_list), preprocess=True)

    return render_template('index.html', qt_models=len(train_list))

@app.route('/configure_model', methods=['POST'])
def configure_model():
    global hydro
    global epochs
    global train_list

    epochs = int(request.form.get('epochs'))
    lstm_units = int(request.form.get('lstm_units'))
    dense1_units = int(request.form.get('dense1_units'))
    dense2_units = int(request.form.get('dense2_units'))
    hydro = HydroModel()
    hydro.build(data.train_windows.shape[1:], data.train_labels.shape[1], lstm_units, dense1_units, dense2_units)
    hydro.compile()
    return render_template('index.html', qt_models=len(train_list), model_configured=True)

@app.route('/add_model', methods=['POST'])
def add_model():
    global hydro
    global epochs
    global train_list

    model_name = request.form.get('model_name')
    cp = ModelCheckpoint(f'Models/{model_name}.keras', save_best_only=True)
    #hydro.train(data.train_windows, data.train_labels, (data.val_windows, data.val_labels), epochs, [cp])
    train_list.append([hydro, data, cp, epochs])
    return render_template('index.html', qt_models=len(train_list), model_training=True)

@app.route('/train_model', methods=['POST'])
def train_model():
    global hydro
    global epochs
    global train_list

    for m in train_list:
        m[0].train(m[1].train_windows, m[1].train_labels, (m[1].val_windows, m[1].val_labels), m[3], m[2])
    
    train_list = []
    return render_template('index.html', qt_models=len(train_list))

if __name__ == '__main__':
    df = pd.read_csv('crc_new_dataset.csv', sep=',', low_memory=False).fillna(np.nan)
    df = df[['Datetime','levi_cr', 'Precipitation_carolyn_cr']]
    df['levi_cr'] = pd.to_numeric(df['levi_cr'], errors='coerce')
    df['Precipitation_carolyn_cr'] = pd.to_numeric(df['Precipitation_carolyn_cr'], errors='coerce')
    train_list = []
    app.run(debug=True)