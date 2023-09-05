import dash
from dash import dcc, Dash, html, Input, Output, dash_table, State, ctx, ALL, callback
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
        self.fig_style = dict(height = 400,
            title = None,
            plot_bgcolor = '#efefef', 
            legend = dict(title='', x=1.0, y=0.5),
            yaxis = dict(gridcolor = '#c4c4c4', title = None),
            xaxis = dict(gridcolor = '#c4c4c4', title = None),
            hovermode="x unified",
            margin = dict(b=10, t=5),
            )

_AppSetup = AppConstants('0.2.0')

dash.register_page(__name__,
                   path='/comparison_tool',  # represents the url text
                   name='Comparison Tool',  # name of page, commonly used as name of link
                   title='Comparison Tool'  # represents the title of browser's tab
)

def clean_data(df, columns, normalize=False):
    convert_to_datetime_object = pd.to_datetime(df['Log Time'], format='%Y-%m-%dT%H:%M:%S')
    reset_timeaxis = [time-convert_to_datetime_object[0] for time in convert_to_datetime_object]
    
    df['Log Time'] = reset_timeaxis

    if normalize == True:
        df_min_max_scaled = df.copy()

        for column in columns:
            df_min_max_scaled[f'{column}_raw'] = df_min_max_scaled[column]
            df_min_max_scaled[column] = (df_min_max_scaled[column]- df_min_max_scaled[column].min())/(df_min_max_scaled[column].max()-df_min_max_scaled[column].min())
        
        return df_min_max_scaled
    
    return df

def get_stepNo_lines(df):
    StepNo_labels = [0]+[i for i in range(1,len(df.StepNo)) if df.StepNo[i] != df.StepNo[i-1]]+[len(df['Log Time'].unique())-1]
    pos = [item for item in df['Log Time'][StepNo_labels]]
    color = ["rgba(255, 255, 255, 0.2)", "rgba(160, 160, 160, 0.2)"]*int(len(pos)*2)

    shapes = [dict(fillcolor=color[i],
            line={"width": 0},
            type="rect",
            x0=pos[i-1],
            x1=pos[i],
            xref="x",
            y0=0,
            y1=1,
            yref="paper")  for i in range(1,len(pos))]
    return shapes


def get_traces(df, xcol, ycols, group_tag, normalized = False):
    if normalized == True: 
        hovertemplates = {col: '<br>%{y:.3f}<br>Original Value = %{customdata[0]:.3f}<br>Duration = %{customdata[1]}<br>StepNo = %{customdata[2]}<br>' for col in ycols}
        custom_data = np.stack((np.array([[i] for i in df[f'{ycols[0]}_raw']]), np.array([[str(i)] for i in df['Log Time']]), np.array([[i] for i in df.StepNo]).astype(np.int64)), axis=1)
    else: 
        hovertemplates = {col: '<br>%{y:.3f}<br>Duration = %{customdata[0]}<br>StepNo = %{customdata[1]}<br>' for col in ycols}
        custom_data = np.stack((np.array([[str(i)] for i in df['Log Time']]), np.array([[i] for i in df.StepNo]).astype(np.int64)), axis=1)


    traces = [go.Scatter(
        x=df[xcol], 
        y=df[col], 
        name = col, 
        customdata=custom_data,  
        hovertemplate = hovertemplates[col], 
        legendgroup=group_tag,  # this can be any string, not just "group"
        legendgrouptitle_text=group_tag,
        ) for col in ycols]
    return traces

nav = dbc.Nav(
    [
        dbc.NavItem([dbc.NavLink("Comparison", active="exact", href="/comparison_tool", style = {'margin-left':'5px', 'color': 'black', 'font-size': '20px', 'border':'1px solid #bcbcbc', 'border-bottom': '1px solid white'})]),
        dbc.NavItem(dbc.NavLink("One File Analysis", active="exact", href="/onefile", style = {'margin-left':'3px','color': 'black', 'font-size': '20px'})),
    ],
    pills=True,
)

comparison_description = """
Features:
- Plot of the data
- Normalization of data to range from 0 to 1
    - Purpose: Usefull for visual comparison
"""

tooltip = html.Div(
    [
        html.P(
            [
                html.Strong("Meaning: "),
                html.Span(
                    "Ensemble averaging",
                    id="tooltip-target",
                    style={"textDecoration": "underline", "cursor": "pointer"},
                )
            ]
        ),
        dbc.Tooltip([
            dcc.Markdown(""" An average taken over the y-direction, using multiple processes, e.g. using 4 processes, an average is first computed using the first data point in each process and so on """)],
            target="tooltip-target",
            style = {'width': '10cm'}
        ),
    ]
)
main_container = [#html.Div([nav], style={'border-bottom': '1px solid #bcbcbc', 'margin-bottom':'-1px'}), 
                  #html.Br(),
                  html.H3(html.Strong('Comparison')),
                  dcc.Markdown(comparison_description),
                #   tooltip,
                  dbc.Row([html.H4('Choose the variables to show')]),
                  dbc.Row( dcc.Checklist(
    id = 'variable-boxes',
    options = ['Log Time', 'StepNo', 'CircuitName', 'TMP1', 'TMP2', 'B31', 'B32',
       'B21', 'B22', 'P101', 'RegulatorSP', 'RegulatorFB'],
    value = [],
    inline = True,
    inputStyle={"margin-right": "10px", 'margin-left': '20px'}
)),
                  html.Hr(),
                  html.H4('Plots'),
                  dbc.Row([dcc.Checklist(
    id = 'comparison-fig-boxes',
    # options = ['Show Normalized Data', 'Show Ensemble Average'],
    options = ['Show Normalized Data'],
    value = [],
    inline = False,
    inputStyle={"margin-right": "10px", 'margin-left': '20px'}
)]),
                  dbc.Row([html.Div(id='comparison-fig'),
        ])]

layout = dbc.Container(main_container, fluid=True)

@callback(
    Output('comparison-fig', 'children'),
    Input('variable-boxes', 'value'),
    Input('comparison-fig-boxes', 'value'),
    Input('store-jsonformat', 'data'),

    prevent_initial_call = True
)
def update_figs(var_val, checkbox_val, stored_dict):
    if var_val == []:
        return ''
        
    cols = var_val

    
    file_dict = {}
    for filename in stored_dict.keys():
        df = pd.DataFrame.from_dict(stored_dict[filename])
        dff = df.copy()

        if 'Show Normalized Data' in checkbox_val:
            dff = clean_data(dff, cols, normalize=True)
        else:
            dff = clean_data(dff, cols)
        file_dict[filename] = dff

    figs = []
    for col in cols:
        fig = go.Figure()
        for key in file_dict.keys():
            df = file_dict[key]
            if 'Show Normalized Data' in checkbox_val:
                for trace in get_traces(df, 'Log Time', [col], key, normalized=True):
                    fig.add_trace(trace)
            else: 
                for trace in get_traces(df, 'Log Time', [col], key):
                    fig.add_trace(trace)

        # if 'Show Ensemble Average' in checkbox_val:
        #     lengths = [len(df[col].values) for df in file_dict.values()]
        #     dfs = [df[col].values for df in file_dict.values()]
        #     print(lengths)

        #     mean = 0

        #     if 'Show Normalized Data' in checkbox_val: 
        #         hovertemplates = {col: '<br>%{y:.3f}<br>Original Value = %{customdata[0]:.3f}<br>StepNo = %{customdata[2]}<br>'}
        #         custom_data = np.stack((np.array([[i] for i in df[f'{col}_raw']]), np.array([[str(i)] for i in df['Log Time']]), np.array([[i] for i in df.StepNo]).astype(np.int64)), axis=1)
        #     else: 
        #         hovertemplates = {col: '<br>%{y:.3f}<br>StepNo = %{customdata[1]}<br>'}
        #         custom_data = np.stack((np.array([[str(i)] for i in df['Log Time']]), np.array([[i] for i in df.StepNo]).astype(np.int64)), axis=1)
        #     fig.add_trace(go.Scatter(
        #         x=df['Log Time'], 
        #         y=mean, 
        #         name = col, 
        #         customdata=custom_data,  
        #         hovertemplate = hovertemplates[col], 
        #         legendgroup='mean',  # this can be any string, not just "group"
        #         legendgrouptitle_text='Ensemble Average',))
    

        fig.update_layout(_AppSetup.fig_style, height=600)
        # figshapes = get_stepNo_lines(df)
        
        # fig.update_layout(shapes=figshapes )

        figs.append(html.P([html.Div([html.H5(f'Variable: {col}')], style={'margin-left':'2cm'}), dcc.Graph(id =key+col, figure = fig), html.Label('Note: x-axis is in time units')],style = _AppSetup.box_style))
    return figs