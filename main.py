import socket
import time

from dash import Dash, dcc, html, Input, Output, callback, State
from figures import init_figs
from tcp_server import tcp_client_processing
from multiprocessing import Process, Manager, Queue
import numpy as np
from analysis import baseline_shift, filtered, show_psd, clc_power
import datetime
from sklearn import preprocessing  # scale and center data
from sklearn.svm import SVC
from joblib import load
import pandas as pd

my_global_fig, my_psd_fig = init_figs()
realtime_flag = False
clf_svm = load('svm_model.joblib')
scaler = load('scaler.joblib')

# Dash display
app = Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.Img(src="./assets/ncl_logo.png", className='ncl-logo'),
        html.P(["Real-time Electroencephalogram Analysis System"], className="title"),
        html.Div([
            html.P(["George Zhao"], className="contact"),
            html.A(["Email: j.zhao36@newcastle.ac.uk"], className="contact", href='mailto:j.zhao36@newcastle.ac.uk')
        ])
    ], className="app-header"),
    html.Div([
        html.Div([
            dcc.RadioItems([
                {
                    "label": html.Div(["File"], className="nav-item", id="nav-item-1"),
                    "value": "File"
                },
                {
                    "label": html.Div(["Real-Time"], className="nav-item", id="nav-item-2"),
                    "value": "Real-Time"
                }
            ], id="select-model", value="File", inline=True, className="nav", inputClassName="nav-rad"),
        ], style={'textAlign': 'left', 'width': '50%', 'padding-top': '20px', 'padding-left': '50px'}),
        html.Div([
            html.Button('Start', id='start-stop-button', n_clicks=0),
            html.Button('Exit', id='exit-button', n_clicks=0),
        ], style={'textAlign': 'right', 'width': '50%', 'padding-top': '20px', 'padding-right': '50px'})
    ], style={'display': 'flex'}),
    html.Div([
        html.Div([dcc.Graph(id='sample-graph', figure=my_global_fig)], className="global-graph-graph"),
        dcc.Interval(id='interval-component', interval=1 * 1000, n_intervals=0, disabled=True),
        html.Div([
            dcc.RangeSlider(min=0, max=10, value=[0, 10],
                            tooltip={"placement": "bottom", "always_visible": True},
                            id='my-range-slider')
        ], className="range-slider", )
    ]),
    html.Div([
        html.Div([dcc.Graph(id='psd-graph', figure=my_psd_fig, className='psd-graph-graph')], className="psd-graph"),
        html.Div([
            html.Div([
                html.Div([
                    html.Label(["Sample Frequency: 1000Hz"], id='sample-freq'),
                    html.Div(["Window Range: "], id='output-container-range-slider'),
                    html.Label('Range Select: '),
                    html.Button('1s', className='dash-board-button', style={'width': '50px'}, id='step-1'),
                    html.Label(''),
                    html.Button('2s', className='dash-board-button', style={'width': '50px'}, id='step-2'),
                    html.Label(''),
                    html.Button('3s', className='dash-board-button', style={'width': '50px'}, id='step-3'),
                    html.Br(),
                    html.Label('Move: '),
                    html.Button('Backward', className='dash-board-button', id='step-b'),
                    html.Label(''),
                    html.Button('Forward', className='dash-board-button', id='step-f'),
                    html.Br(),
                    html.Label('Bandpass Filter: '),
                    dcc.Input(value=8, type='number', id='bandpass-low', style={'width': '50px'}),
                    html.Label('Hz ~ '),
                    dcc.Input(value=30, type='number', id='bandpass-high', style={'width': '50px'}),
                    html.Label('Hz'),
                    html.Br(),
                    html.Label("Welch PSD", style={'font-weight': 'bold'}),
                    html.Br(),
                    html.Label("Time Window Size: "),
                    dcc.Input(value=2, type='number', id='welch-tw', style={'width': '50px'}),
                    html.Br(),
                    html.Label('Absolute Power: ', style={'font-weight': 'bold'}),
                    # html.Br(),
                    dcc.Input(value=8, type='number', style={'width': '50px'}, id='abs-low'),
                    html.Label('Hz ~ '),
                    dcc.Input(value=12, type='number', style={'width': '50px'}, id='abs-high'),
                    html.Label('Hz'),
                    html.Br(),
                    html.Label('0', id='abs-power'),
                    html.Div(id='file-info', children='')
                ], className='dash-board-text'),
            ], className='dash-board-frame'),
        ], className='dash-board'),
        html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        html.Div(className='circle', id='stop-id'),
                        html.Label(['Stop'], style={'padding-top': '30px'}),
                        html.Div(className='circle', style={'background-color': '#003300'}, id='move-id'),
                        html.Label(['Move'], style={'padding-top': '30px'}),
                    ], className='circle-container'),

                    html.Div([
                        html.Div(className='rectangle') for _ in range(8)
                    ], className='rectangle-container')
                ], className='container_light')
            ], className='dash-board-frame', style={'width': '80%'})
        ], style={'width': '20%', 'margin-left': '100px'}),
    ], style={'display': 'flex'})
])


# Callback functions
# for choosing the data analyse mode
@app.callback(
    Output('nav-item-1', 'style'),
    Output('nav-item-2', 'style'),
    Output('my-range-slider', 'value'),
    Output('interval-component', 'disabled'),
    Input('select-model', 'value'),
    Input('my-range-slider', 'value'))
def update_model(value, value1):
    global realtime_flag
    if value == "Real-Time":
        realtime_flag = True
        # processing for tcp/ip function
        buggy_processing.start()
        tcp_processing.start()
        return {'background-color': 'white', 'color': 'black'}, {'background-color': '#163a6c', 'color': 'white'}, \
            [9, 10], False
    if value == "File":
        if realtime_flag:
            realtime_flag = False
            buggy_processing.join()
            tcp_processing.join()
        return {'background-color': '#163a6c', 'color': 'white'}, {'background-color': 'white', 'color': 'black'}, \
            value1, True


@app.callback(
    Output('start-stop-button', 'n_clicks'),
    Output('exit-button', 'n_clicks'),
    Output('file-info', 'children'),
    Input('start-stop-button', 'n_clicks'),
    Input('exit-button', 'n_clicks'),
    Input('select-model', 'value'),
)
def button_clicked(start_stop_clicks, exit_clicks, value):
    file_name = f"datas.txt"
    if value == "Real-Time":
        if not exit_clicks > 0:
            if start_stop_clicks > 0:
                if start_stop_clicks % 2 == 0:
                    q.put('1')  # stop
                else:
                    current_time = datetime.datetime.now().strftime("datas_%Y-%m-%d_%H-%M-%S")
                    file_name = f"{current_time}.txt"
                    q.put('0')  # start
        else:
            q.put('2')
            tcp_processing.join()
    return start_stop_clicks, exit_clicks, file_name


@callback(
    Output('sample-graph', 'figure'),
    Output('psd-graph', 'figure'),
    Output('abs-power', 'children'),
    Output('stop-id', 'style'),
    Output('move-id', 'style'),
    Input('select-model', 'value'),
    Input('interval-component', 'n_intervals'),
    State('file-info', 'children'),
)
def update_metrics(value, n, file_info):
    data_list = list(d)
    if value == "Real-Time" and data_list:
        array_length = 10000
        my_global_fig.update_traces(
            x=np.linspace(0, 11, array_length),
            y=data_list,
        )
        # global
        welch_tw = 0.8
        psd_sr = 1000
        freq_low_delta, freq_high_delta = 2, 4
        freq_low_theta, freq_high_theta = 4, 8
        freq_low_alpha, freq_high_alpha = 8, 14
        freq_low_beta, freq_high_beta = 14, 30
        freq_low_gamma, freq_high_gamma = 30, 100
        start_time = len(data_list) // 1000 - 1
        end_time = len(data_list) // 1000
        if start_time < 0:
            start_time = 0
        bs_data = baseline_shift(data_list, t_start=start_time, t_end=end_time)
        # filter_data = filtered(bs_data, f1=3, f2=30, sr=psd_sr)
        # psd_data = show_psd(filter_data, welch_tw=welch_tw, sr=psd_sr)
        # my_psd_fig.update_traces(
        #     x=psd_data[0],
        #     y=psd_data[1],
        # )
        # abs_power = clc_power(psd_data[0], psd_data[1], freq_low=8, freq_high=12)
        # Delta
        filter_delta = filtered(bs_data, f1=freq_low_delta, f2=freq_high_delta, sr=psd_sr)
        psd_delta = show_psd(filter_delta, welch_tw=welch_tw, sr=psd_sr)
        power_delta = clc_power(psd_delta[0], psd_delta[1], freq_low=freq_low_delta, freq_high=freq_high_delta)
        # Theta
        filter_theta = filtered(bs_data, f1=freq_low_theta, f2=freq_high_theta, sr=psd_sr)
        psd_theta = show_psd(filter_theta, welch_tw=welch_tw, sr=psd_sr)
        power_theta = clc_power(psd_theta[0], psd_theta[1], freq_low=freq_low_theta, freq_high=freq_high_theta)
        # Alpha
        filter_alpha = filtered(bs_data, f1=freq_low_alpha, f2=freq_high_alpha, sr=psd_sr)
        psd_alpha = show_psd(filter_alpha, welch_tw=welch_tw, sr=psd_sr)
        power_alpha = clc_power(psd_alpha[0], psd_alpha[1], freq_low=freq_low_alpha, freq_high=freq_high_alpha)
        # Beta
        filter_beta = filtered(bs_data, f1=freq_low_beta, f2=freq_high_beta, sr=psd_sr)
        psd_beta = show_psd(filter_beta, welch_tw=welch_tw, sr=psd_sr)
        power_beta = clc_power(psd_beta[0], psd_beta[1], freq_low=freq_low_beta, freq_high=freq_high_beta)
        # Gamma
        filter_gamma = filtered(bs_data, f1=freq_low_gamma, f2=freq_high_gamma, sr=psd_sr)
        psd_gamma = show_psd(filter_gamma, welch_tw=welch_tw, sr=psd_sr)
        power_gamma = clc_power(psd_gamma[0], psd_gamma[1], freq_low=freq_low_gamma, freq_high=freq_high_gamma)

        abs_power = [power_delta, power_theta, power_alpha, power_beta, power_gamma]
        my_psd_fig.update_traces(y=abs_power)

        # Convert test data to DataFrame with appropriate feature names
        test_a_df = pd.DataFrame([abs_power], columns=["Delta", "Theta", "Alpha", "Beta", "Gamma"])

        # Preprocess the new data using the loaded scaler
        test_a_scaled = scaler.transform(test_a_df)

        # Make a prediction using the loaded model
        prediction = clf_svm.predict(test_a_scaled)
        show_abs = f'{prediction[0]},{type(prediction[0])},{abs_power}'

        q1.put(str(prediction[0]))
        if str(prediction[0]) == '1':
            style_stop = {'background-color': '#FF0000'}
            style_start = {'background-color': '#003300'}
        else:
            style_stop = {'background-color': '#660000'}
            style_start = {'background-color': '#00CC00'}

        # abs_power = f'{abs_power[0]},\t{abs_power[1]},\t{abs_power[2]},\t{abs_power[3]},\t{abs_power[4]}\n'
        #
        # # File writing
        # file_name = file_info.split()[-1]
        # with open(file_name, "a") as file:
        #     if not file_name == 'datas.txt':
        #         file.write(abs_power)

    else:
        show_abs = 0
        style_stop = {'background-color': '#660000'}
        style_start = {'background-color': '#003300'}
    return my_global_fig, my_psd_fig, show_abs, style_stop, style_start


def create_server(q1):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server_address = ('192.168.134.50', 12345)
    server_socket.bind(server_address)

    server_socket.listen(1)

    print(f"服务器在 {server_address} 上等待连接...")

    connection, client_address = server_socket.accept()

    while True:
        user_action = str(q1.get())
        if user_action == '1':
            connection.sendall(b'm\n')
        elif user_action == '0':
            connection.sendall(b'm\n')
        else:
            pass
    # while True:
    #     connection.sendall(b'm\n')
    #     time.sleep(1)
    #
    #     connection.sendall(b's\n')
    #     time.sleep(1)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    key = 0
    # shared variables between dash app and tcp/ip server
    with Manager() as manager:
        d = Manager().list()  # raw datas
        q = Queue()  # control signal
        q1 = Queue()
    tcp_processing = Process(target=tcp_client_processing, args=(d, q))
    buggy_processing = Process(target=create_server, args=(q1,))
    # dash app run
    app.run(debug=True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
