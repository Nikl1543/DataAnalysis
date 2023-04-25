# from jupyter_dash import JupyterDash
import dash
from dash import dcc, Dash, html, Input, Output, dash_table, State, ctx, ALL
import dash_bootstrap_components as dbc
import pandas as pd
import base64
import io
import webbrowser
import os
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import json
from fpdf import FPDF, HTMLMixin
from statsmodels.tsa.seasonal import seasonal_decompose
import tempfile
from PIL import Image
import urllib.request
import ssl

class someSetup():
    def __init__(self, version, digits=2, port=8050, host = "127.0.0.1"):
        self.version = version
        self.digits = digits
        self.port = port
        self.host = host
        self.box_style = {'width': '99%',
        'height': '235mm',
        'lineHeight': '60px',
        'border': '1px solid #999',
        'textAlign': 'center',
        'margin': '10px',
        'padding': '5px'}

        self.color = {'titlecolor': '#60893c', 'plot_background': '#efefef', 'gridcolor': 'c4c4c4', 'plotcolor2': 'dadada'}


def run_server(HOST, PORT):
    
    # The reloader has not yet run - open the browser
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open_new('http://127.0.0.1:8050/')

    # Otherwise, continue as normal
    app.run(host=HOST, port=PORT)

def extract_info_from_filename(file):
    if len(file) >1:
        filename, *file_info = file
        file_info = html.Table([html.Tr([html.Th(html.Strong(i), style = {'padding': '10px'}) for i in file_info])], style={'padding': '10px', 'overflow': 'hidden', 'textOverflow': 'ellipsis', 'maxWidth': 0, 'height': 'auto'})

        return html.P([
            dbc.Row([html.Label('Filename:'), html.Div([html.Label([html.Strong(filename)])],style = {'overflow': 'hidden','text-overflow': 'ellipsis', 'width': '90%'}),
                # dcc.Markdown(f'Filename: **{filename}** \n\n Details:'), 
            dbc.Row([html.Label('Details:'),file_info])])], 
            style = { 'width': '99%',
        'border': '1px solid #999',
        'borderRadius': '5px',
        'textAlign': 'left',
        'margin': '10px',
        'padding': '5px'})
    else:
        return html.P([dbc.Row([dcc.Markdown(f'**Filename**: {file[0]}')])
        ], style = SETUP.box_style)


SETUP = someSetup(version = '2.1.5')
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, use_pages=True, external_stylesheets=external_stylesheets, pages_folder='pages')
server = app.server

sidebar = dbc.Nav(
            [
                dbc.NavLink(
                    [
                        html.Div('Home', className="ms-2"),
                    ],
                    href='/',
                    active="exact",
                    style = {'color': 'black'},
                ),
                                dbc.NavLink(
                    [
                        html.Div('Data Analysis', className="ms-2"),
                    ],
                    href='/dataanalysis',
                    active='partial',
                    style = {'color': 'black'},
                ),
                                dbc.NavLink(
                    [
                        html.Div('Report', className="ms-2"),
                    ],
                    href='/report',
                    active="exact",
                    style = {'color': 'black'},
                )
            ],
            vertical=True,
            pills=True,
)

upload = dcc.Upload(
    children=html.Div([
        'Drag and Drop or ',
            html.A('Select File',href="#", style={'color': '#1d8fe5'})
    ]),
    style={
        'width': '99%',
        'height': '60px',
        'lineHeight': '60px',
        'border': '1px dashed #999',
        'borderRadius': '5px',
        'textAlign': 'center',
        'margin': '10px',
        'padding': '5px'
    },
    # Allow multiple files to be uploaded
    multiple=True,
    id ='upload' 
)

def get_ListGroup(list_of_names, active_file):
    def get_row(idx, name, active=False):
        if active == True:
            idx_col = html.Th(str(idx), scope='row', style={'background-color': '#60893c'})
            name_col = html.Td(html.Strong(name.replace('.csv',''), style={'font-size': '0.8em'}), style={'color': '#60893c', 'text-align':'left'})
        else:
            idx_col = html.Th(str(idx), scope='row')
            name_col = html.Td(name, style={'text-align':'left'})

        row = html.Tr([
                    idx_col,
                    name_col,
                    html.Td([
                        html.Ul(className='list-inline-m-0', children=[
                            html.Li([
                                html.Button(className='btn btn-light btn-sm rounded-0', title='Select',children=[html.Img(src=app.get_asset_url('Cursor.svg'), alt='logo', style={'height': '5mm', 'width':'5mm', 'float': 'right'})], id = {'type': 'ListFocusBtn', 'index': name})
                            ], className='list-inline-item'),
                            html.Li([
                                html.Button(className='btn btn-danger btn-sm rounded-0', title='Delete',children=[html.Img(src=app.get_asset_url('trash.svg'),  alt='logo', style={'height': '5mm', 'width':'5mm', 'float': 'right'})], id = {'type': 'ListDeleteBtn', 'index': name})
                            ], className='list-inline-item'),
                        ])
                    ])
                ])
        return row

    ListGroup = html.Div([
        html.Table( 
            children = [html.Thead([
                html.Tr([
                    html.Th(['#'],scope='col'),
                    html.Th(['Filename'],scope='col'),
                ])
            ]),
            html.Tbody([
            get_row(idx, name, active=True) if name == active_file else get_row(idx, name) for idx, name in enumerate(list_of_names)])      
    ])
    ],     style={
        'textAlign': 'center',
        'width': '100%'
    })
    return ListGroup




app.layout = dbc.Container([

    dbc.Row([dbc.Col([html.Div([f'version({SETUP.version})'], style={'font-size': '12pt'})],width=2), dbc.Col([html.H1('Nolek Data Analysis', style={'color': SETUP.color['titlecolor'], 'text-align': 'center'})], width = 8), dbc.Col([html.Img(src=app.get_asset_url('nolek_logo.png'), alt='logo', style={'height': '2cm', 'float': 'right'})],width=2),
            html.Hr(style = {'color': SETUP.color['titlecolor']})]),

    html.Hr(),

    dbc.Row(
        [
            dbc.Col(
                [
                    sidebar,
                    upload,
                    dcc.Location(id='url', refresh=False),
                    html.Label(id='path_info'),
                    data_store := dcc.Store(data = {}, id='store-data', storage_type='session'), 
                    file_info_store := dcc.Store(id='store-file_info', storage_type = 'session'),
                    file_info := html.Div(id = 'file_info'),
                    list_of_files:= html.Div(id = 'list_of_files', style={'width': '99%','margin': '10px','padding': '0px'}),
                    one_file_store := dcc.Store(id='store-one_file', storage_type='local'), 
                    dcc.Store(data = {'active_file': '', 'delete_file': [], }, id = 'log_of_btns'),
                    dcc.Store(data = {}, id='file-store'),
                ], xs=4, sm=4, md=2, lg=2, xl=2, xxl=2),

            dbc.Col(
                [
                    dash.page_container
                ], xs=8, sm=8, md=10, lg=10, xl=10, xxl=10)
        ]
    )
], fluid=True)

@app.callback(
    Output('log_of_btns', 'data'),
    Output('file-store', 'data'),
    Input({'type': 'ListDeleteBtn', 'index': ALL}, 'n_clicks'), 
    Input({'type': 'ListFocusBtn', 'index': ALL}, 'n_clicks'),
    Input('upload', 'filename'),
    State('upload', 'contents'),
    State('log_of_btns', 'data'),
    State('file-store', 'data'),
    prevent_initial_call = True
)
def update_ListGroup_and_BtnsLog(DeleteListGroup, listGroup, filenames, contents, BtnsLog, FileStore):
    if [contents, filenames] == [None, None]:
        return dash.no_update
    
    trigger_id = ctx.triggered_id
    if trigger_id == 'upload':
        for file, content in zip(filenames, contents):
            if file not in FileStore:
                FileStore[file] = content.split(',')[-1]
            
            if file in BtnsLog['delete_file']:
                BtnsLog['delete_file'].remove(file)

        if BtnsLog['active_file'] == '':
            first_key = list(FileStore.keys())[0]
            BtnsLog['active_file'] = first_key
        return BtnsLog, FileStore

    elif trigger_id['type'] == 'ListFocusBtn':
        BtnsLog['active_file'] = trigger_id['index']

        return BtnsLog, dash.no_update
    
    elif trigger_id['type']=='ListDeleteBtn':
        BtnsLog['delete_file'].append(trigger_id['index'])
        if BtnsLog['active_file'] in BtnsLog['delete_file']:
            del FileStore[trigger_id['index']]

            if len(FileStore.keys()) == 0:
                BtnsLog['active_file'] = ''
            else:
                first_key = list(FileStore.keys())[0]
                BtnsLog['active_file'] = first_key


            return BtnsLog, FileStore
        else:
            del FileStore[trigger_id['index']]
            return BtnsLog, FileStore


@app.callback(Output('store-data', 'data'),
              Output('store-one_file', 'data'),
              Output('list_of_files', 'children'),
              Output('file_info', 'children'),
              Input('log_of_btns', 'data'),
              State('file-store', 'data'),
              prevent_initial_call=True)
def update_output(BtnsLog, FileStore):
    #Make List Group
    active_file = BtnsLog['active_file']
    list_group = get_ListGroup(list(FileStore.keys()), active_file)
    if len(FileStore) == 0:
        return {}, {}, list_group, ''

    stored_data = {}
    for name, content_string in FileStore.items():
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')),
                sep='[;]', 
                skiprows = [0,2],
                engine='python',
                decimal=','
                    )
        
        stored_data[name] = df.to_dict('records')
        if name == active_file: 
    
            one_file = df.to_dict('records')
            file_info = pd.read_csv(io.StringIO(decoded.decode('utf-8')),
                        sep='[;]', 
                        engine='python',
                        decimal=',',
                        nrows=0
                        )
            file_info = [name for name in file_info.columns if 'Unnamed' not in name]
            info_from_filename = extract_info_from_filename(file_info)


    return (stored_data, one_file, list_group, info_from_filename)
    
        

if __name__ == "__main__":
#    run_server(SETUP.host, SETUP.port)
    app.run(host="127.0.0.1", port=8050, debug=True)


