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
from plotly.subplots import make_subplots
import json
from fpdf import FPDF, HTMLMixin
from statsmodels.tsa.seasonal import seasonal_decompose
import tempfile
from PIL import Image
import urllib.request as url_request
import ssl

app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP], pages_folder='pages')
app.config.suppress_callback_exceptions=True

class AppConstants:
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
        self.fig_style = dict(height = 1080,
            title = None,
            plot_bgcolor = '#efefef', 
            legend = dict(title='', x=1.0, y=0.5),
            # yaxis = dict(gridcolor = '#c4c4c4', title = None),
            # xaxis = dict(gridcolor = '#c4c4c4', title = None),
            hovermode="x unified",
            margin = dict(b=10, t=5),
            )

_AppSetup = AppConstants('0.2.0')

def convert_to_datetime(input):
    convert_to_datetime_object = []
    for val in input:
        try:
            convert_to_datetime_object += [datetime.strptime(val, '%d/%m/%Y %H.%M.%S').isoformat(timespec='seconds')]
        except:
            convert_to_datetime_object += [datetime.strptime(val, '%d/%m/%Y').isoformat(timespec='seconds')]
    
    return convert_to_datetime_object
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
        ], style = _AppSetup.box_style) 
def get_ListGroup(list_of_names, active_file):
    def get_row(idx, name, active=False):
        if active == True:
            # idx_col = html.Th(str(idx), scope='row', style={'background-color': '#60893c'})
            name_col = html.Td(html.Strong(name.replace('.csv','')), style={'color': '#60893c', 'text-align':'left'})
        else:
            # idx_col = html.Th(str(idx), scope='row')
            name_col = html.Td(html.Strong(name.replace('.csv','')), style={'text-align':'left'})

        row = html.Tr([
                    # idx_col,
                    html.Button(children=name_col, className='btn btn-light-outline btn-sm rounded-0', title='Select File', id = {'type': 'ListFocusBtn', 'index': name}, style={'bg-color':'white'}),
                    html.Button(className='btn btn-light-outline btn-sm rounded-0', title='Delete Item',children=[html.Strong('X')], id = {'type': 'ListDeleteBtn', 'index': name})
                ])
        return row

    ListGroup = html.Div([
        html.Table( 
            children = [html.Thead([
                html.Tr([
                    html.Th(['Filename'],scope='col'),
                    
                ])
            ]),
            html.Tbody([
            get_row(idx, name, active=True) if name == active_file else get_row(idx, name) for idx, name in enumerate(list_of_names)])      
    ]),
    html.Button('clear', className='btn btn-dark-outline btn-sm', title='Clear List',id={'type': 'clearbtn', 'index': '0'})
    ],     style={
            'width': '99%',
            'height': '70px',
            'lineHeight': '30px',
            'textAlign': 'center',
            'margin': 'auto',
            'padding': '5px',
    'justify-content': 'center',
    'align-items': 'center',
    })
    return ListGroup
def run_server(host, port, debugmode=True):
    if debugmode == True:
        app.run(host=host, port=port, debug=True)
    # The reloader has not yet run - open the browser
    else:
        if not os.environ.get("WERKZEUG_RUN_MAIN"):
            webbrowser.open_new(f'http://{host}:{port}/')

        # Otherwise, continue as normal
        app.run(host=host, port=port)


header_navbar = dbc.Navbar(
    children=[
        dbc.NavbarToggler(id="navbar-toggler"),
        dbc.Collapse(
            dbc.Nav(
                [
                    dbc.NavItem(dbc.NavLink("Home", href="/", active="exact", style={"font-size":"1.5vw"})),
                    dbc.DropdownMenu(
                        [
                            dbc.DropdownMenuItem("One File Analysis", href="/onefile", style={"font-size":"1.5vw"}),
                            dbc.DropdownMenuItem("Comparison Tool", href="/comparison_tool", style={"font-size":"1.5vw"}),
                        ],
                        label="Analysis Tools",
                        style={"font-size":"1.5vw"},
                        nav=True,
                        in_navbar=True,
                    ),
                    dbc.NavItem(dbc.NavLink("Report", href="/report", active="exact", style={"font-size":"1.5vw"})),
                ],
                className="ml-auto",
                navbar=True,
            ),
            id="navbar-collapse",
            navbar=True,
        ),
    ],
    color="white",
    class_name="justify-content-center"
)
upload = dcc.Upload(
        children=html.Div([
            'Drag and drop or ',
                html.A('Select Files',href="#", style={'color': '#1d8fe5'})
        ]),
        style={
            'width': '99%',
            'height': '70px',
            'lineHeight': '30px',
            'border': '1px dashed #999',
            'borderRadius': '6px',
            'textAlign': 'center',
            'margin': 'auto',
            'padding': '5px',
            'display': 'flex',
    'justify-content': 'center',
    'align-items': 'center',
        },
        # Allow multiple files to be uploaded
        multiple=True,
        id ='upload' 
    )

server = app.server

app.layout = dbc.Container([
    dbc.Row([dbc.Col([html.Div([f'version({_AppSetup.version})'], style={'font-size': '12pt'})],width=2), dbc.Col([html.H1('Nolek Data Analysis', style={'color': _AppSetup.color['titlecolor'], 'text-align': 'center'})], width = 8), dbc.Col([html.Img(src=app.get_asset_url('nolek_logo.png'), alt='logo', style={'height': '2cm', 'float': 'right'})],width=2),
            header_navbar,
            html.Hr(style = {'color': _AppSetup.color['titlecolor']})]),
    
    dbc.Row([
            
    ]),
    dbc.Row(
        [
            dbc.Col(
                [
                    dcc.Location(id='url', refresh=False),
                    upload,
                    dcc.Store(data = {}, id='store-fig-configs', storage_type='session'),
                    dcc.Store(data = {}, id='store-raw-data', storage_type='session'), 
                    dcc.Store(data = {}, id='store-jsonformat', storage_type='session'), 
                    dcc.Store(data = {}, id='store-one-file', storage_type='session'), 
                    html.Div(id = 'file_info'),
                    html.Div(id = 'list_of_files', style={'width': '99%','margin': '10px','padding': '0px'}),
                    dcc.Store(data = '', id = 'active-file-holder')
                    
                ], xs=4, sm=4, md=2, lg=2, xl=2, xxl=2),
            dbc.Col(
                [
                    dash.page_container
                ], xs=8, sm=8, md=10, lg=10, xl=10, xxl=10)
        ]
    )
], fluid=True)

# @app.callback(
#     Output('active-file-holder', 'data'),
#     Output('store-raw-data', 'data'),
#     Output('list_of_files', 'children'),
#     Output('file_info', 'children'),
#     Input({'type': 'ListDeleteBtn', 'index': ALL}, 'n_clicks'), 
#     Input({'type': 'ListFocusBtn', 'index': ALL}, 'n_clicks'),
#     Input({'type': 'clearbtn', 'index': ALL}, 'n_clicks'),
#     Input('_pages_location', 'pathname'),
#     Input('upload', 'filename'),
#     State('upload', 'contents'),
#     State('active-file-holder', 'data'),
#     State('store-raw-data', 'data'),
# )
# def update_ListGroup_and_BtnsLog(DeleteListGroup, listGroup, clearBtn, page_location, filenames, contents, ActiveFile, FileStore):
#     trigger_id = ctx.triggered_id

#     if trigger_id == '_pages_location':
#         if len(FileStore.keys())!=0:
#             keys = list(FileStore.keys())
#             ActiveFile= keys[0]
#             list_group = get_ListGroup(keys, keys[0])
#         else:
#             ActiveFile= ''  
#             list_group = html.Div()

#     # New files uploaded
#     elif trigger_id == 'upload':
#         for file, content in zip(filenames, contents):
#             if ".csv" in file:
#                 if file not in FileStore:
#                     #Store raw content of file
#                     FileStore[file] = content.split(',')[-1]

#         #Update active file
#         if ActiveFile == '':
#             first_key = list(FileStore.keys())[0]
#             ActiveFile = first_key

#         #Make List Group
#         list_group = get_ListGroup(list(FileStore.keys()),  ActiveFile)
    
#     elif trigger_id['type'] == 'clearbtn':
#         FileStore = {}
#         ActiveFile = ''
#         list_group = html.Div()
    
#     elif trigger_id['type'] == 'ListFocusBtn':
#         #Active file is changed
#         ActiveFile = trigger_id['index']
#         list_group = get_ListGroup(list(FileStore.keys()), ActiveFile)
    
#     elif trigger_id['type']=='ListDeleteBtn':
#         del FileStore[trigger_id['index']]

#         #The deleted file was the active file
#         if ActiveFile == trigger_id['index']:  
            
#             # File store empty? Yes: reset active file to empty string, No: first file in file list is chosen
#             if len(FileStore.keys()) == 0:
#                 ActiveFile = ''
#                 list_group = html.Div()

#             else:
#                 first_key = list(FileStore.keys())[0]
#                 ActiveFile = first_key
        
#         list_group = get_ListGroup(list(FileStore.keys()), ActiveFile)

#     if ActiveFile == '':
#         info_from_filename = ''
#     else:
#         content_string = [value for name, value in FileStore.items() if name==ActiveFile][0]
#         decoded = base64.b64decode(content_string)
#         file_info = pd.read_csv(io.StringIO(decoded.decode('utf-8')),
#                     sep='[;]', 
#                     engine='python',
#                     decimal=',',
#                     nrows=0
#                     )
#         file_info = [name for name in file_info.columns if 'Unnamed' not in name]
#         info_from_filename = extract_info_from_filename(file_info)

#     return ActiveFile, FileStore, list_group, info_from_filename

# @app.callback(
#         Output('store-jsonformat', 'data'),
#         Output('store-one-file', 'data'),
#         Input('store-raw-data', 'data'),
#         State('active-file-holder', 'data'),
#         prevent_initial_call=True
# )
# def raw_to_jsonformat(FileStore, ActiveFile):

#     if len(FileStore)==0:
#         return dash.no_update

#     json_data = {}
#     one_file = {}
#     for name, content_string in FileStore.items():
#         decoded = base64.b64decode(content_string)
#         df = pd.read_csv(
#                 io.StringIO(decoded.decode('utf-8')),
#                 sep='[;]', 
#                 skiprows = [0,2],
#                 engine='python',
#                 decimal=','
#                     )
#         #ensure correct conversion of time axis can be done
#         if 'Log Time' in df.columns:
#             df['Log Time'] = convert_to_datetime(df['Log Time'])
                
#         json_data[name] = df.to_dict('records')
#         if name == ActiveFile: 
    
#             one_file = df.to_dict('records')
#     return json_data, one_file

if __name__ == "__main__":
   run_server(host=_AppSetup.host, port=_AppSetup.port)