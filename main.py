from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from Models.data_preprocessor import DataPreprocessor
from Models.modeling import HydroModel
import os
import copy
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.callbacks import ModelCheckpoint

app = Flask(__name__)

@app.route('/')
def index():
    global train_list
    global model_conf
    
    return render_template('index.html', model_conf=model_conf, data_conf=data_conf, qt_models=len(train_list), layers=model_conf['layers'])


@app.route('/configure', methods=['POST'])
def configure():
    global train_list
    global data_conf
    global model_conf
    
    if request.method == 'POST':
        data_conf['start_date'] = request.form.get('start_date')
        data_conf['end_date'] = request.form.get('end_date')
        data_conf['method'] = request.form.get('method')
        data_conf['splitdf'] = request.form.get('splitdf')
        data_conf['window_size'] = int(request.form.get('window_size'))
        data_conf['label_size'] = int(request.form.get('label_size'))
        data_conf['shift'] = int(request.form.get('shift'))
        data_conf['steps_future'] = int(request.form.get('steps_future'))
        
        data = DataPreprocessor(data_conf['method'], data_conf['splitdf'], data_conf['start_date'], data_conf['end_date'])
        data.fill_data(df)
        data.create_windows(data_conf['window_size'], 'levi_cr', data_conf['shift'], data_conf['label_size'], data_conf['steps_future'])
        data.separete_dataset()
        data_conf['data'] = data
        return render_template('index.html', model_conf=model_conf, data_conf=data_conf, qt_models=len(train_list), layers=model_conf['layers'], preprocess=True)

    return render_template('index.html', model_conf=model_conf, data_conf=data_conf, qt_models=len(train_list), layers=model_conf['layers'])

@app.route('/layer_conf', methods=['POST'])
def layer_conf():
    global train_list
    global data_conf
    global model_conf
    
    layer = request.form.get('layer_type')
    units = int(request.form.get('layer_units'))
    model_conf['layers'].append([layer, units])
    
    return render_template('index.html', model_conf=model_conf, data_conf=data_conf, qt_models=len(train_list), layers=model_conf['layers'])

@app.route('/layer_reset', methods=['POST'])
def layer_reset():
    global model_conf
    
    model_conf['layers'] = []
    return render_template('index.html', model_conf=model_conf, data_conf=data_conf, qt_models=len(train_list), layers=model_conf['layers'])

@app.route('/configure_model', methods=['POST'])
def configure_model():
    global train_list
    global data_conf
    global model_conf

    model_conf['epochs'] = int(request.form.get('epochs'))
    hydro = HydroModel()
    hydro.build(data_conf['data'].train_windows.shape[1:], data_conf['data'].train_labels.shape[1], model_conf['layers'])
    hydro.compile()
    model_conf['hydro'] = hydro
    return render_template('index.html', model_conf=model_conf, data_conf=data_conf, qt_models=len(train_list), layers=model_conf['layers'], model_configured=True)

@app.route('/add_model', methods=['POST'])
def add_model():
    global train_list
    global data_conf
    global model_conf

    model_name = request.form.get('model_name')
    cp = ModelCheckpoint(f'Models/{model_name}.keras', save_best_only=True)
    train_list.append([copy.deepcopy(model_conf), copy.deepcopy(data_conf), cp])
    print(train_list)
    return render_template('index.html', model_conf=model_conf, data_conf=data_conf, qt_models=len(train_list), layers=model_conf['layers'], model_training=True, train_list=train_list)

@app.route('/train_model', methods=['POST'])
def train_model():
    global train_list
    global model_conf
    global data_conf

    for m in train_list:
        m[0]['hydro'].train(m[1]['data'].train_windows, m[1]['data'].train_labels, (m[1]['data'].val_windows, m[1]['data'].val_labels), m[0]['epochs'], m[2])
    
    train_list = []
    return render_template('index.html', model_conf=model_conf, data_conf=data_conf, qt_models=len(train_list), layers=model_conf['layers'], train_list=train_list)

if __name__ == '__main__':
    
    data_conf = {
        'data': None,
        'start_date': None,
        'end_date': None,
        'method': None,
        'splitdf': None,
        'window_size': None,
        'label_size': None,
        'shift': None,
        'steps_future': None
    }

    model_conf = {
        'hydro': None,
        'epochs': None,
        'layers': [['lstm', 64], ['dense', 32], ['dense', 32]]
    }
    
    df = pd.read_csv('crc_new_dataset.csv', sep=',', low_memory=False).fillna(np.nan)
    df = df[['Datetime','levi_cr', 'Precipitation_carolyn_cr']]
    df['levi_cr'] = pd.to_numeric(df['levi_cr'], errors='coerce')
    df['Precipitation_carolyn_cr'] = pd.to_numeric(df['Precipitation_carolyn_cr'], errors='coerce')
    train_list = []
    app.run(debug=True)
