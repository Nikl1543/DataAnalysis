# from jupyter_dash import JupyterDash
import dash
from dash import dcc, Dash, html, Input, Output, dash_table, State, ctx, callback
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

dash.register_page(__name__,
                   path='/dataanalysis/comparison_tool',  # represents the url text
                   name='Comparison Tool',  # name of page, commonly used as name of link
                   title='Comparison Tool'  # represents the title of browser's tab
)

def convert_logtime(df):
    start = datetime.strptime(df['Log Time'][0].replace('.', ':').split(' ')[-1], '%H:%M:%S')
    drop_dates = [0]+[(datetime.strptime(logtime.replace('.', ':').split(' ')[-1], '%H:%M:%S')-start).seconds/60/60 for logtime in df['Log Time'][1:]]
    return drop_dates

def clean_data(df, columns, normalize=False):
    df['Log Time Raw'] = [time.replace('.', ':').split(' ')[-1] for time in df['Log Time']]
    df['Log Time'] = convert_logtime(df)


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
        hovertemplates = {col: '<br>%{y:.3f}<br>Original Value = %{customdata[0]:.3f}<br>Log Time = %{customdata[1]}<br>StepNo = %{customdata[2]}<br>' for col in ycols}
        custom_data = np.stack((np.array([[i] for i in df[f'{ycols[0]}_raw']]), np.array([[str(i)] for i in df['Log Time Raw']]), np.array([[i] for i in df.StepNo]).astype(np.int64)), axis=1)
    else: 
        hovertemplates = {col: '<br>%{y:.3f}<br>Log Time = %{customdata[0]}<br>StepNo = %{customdata[1]}<br>' for col in ycols}
        custom_data = np.stack((np.array([[str(i)] for i in df['Log Time Raw']]), np.array([[i] for i in df.StepNo]).astype(np.int64)), axis=1)


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

fig_style = dict(height = 720,
    title = None,
    plot_bgcolor = '#efefef', 
    legend = dict(title='', x=1.0, y=0.5, groupclick='toggleitem'),
    yaxis = dict(gridcolor = '#c4c4c4', title = None),
    xaxis = dict(gridcolor = '#c4c4c4', title = None),
    hovermode="x unified",
    margin = dict(b=10, t=5))

box_style = {'margin': '10px', 'padding': '5px','border': '3px solid #bcbcbc'}
color = {'titlecolor': '#60893c', 'plot_background': '#efefef', 'gridcolor': 'c4c4c4', 'plotcolor2': 'dadada'}

nav = dbc.Nav(
    [
        dbc.NavItem([dbc.NavLink("Comparison", active="exact", href="/dataanalysis/comparison_tool", style = {'margin-left':'5px', 'color': 'black', 'font-size': '20px', 'border':'1px solid #bcbcbc', 'border-bottom': '1px solid white'})]),
        dbc.NavItem(dbc.NavLink("One File Analysis", active="exact", href="/dataanalysis/onefile", style = {'margin-left':'3px','color': 'black', 'font-size': '20px'})),
    ],
    pills=True,
)
comparison_description = """
The "Comparison"-tool has the following features:
- Plot of the data
- Ensemble Averaging
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
main_container = [html.Div([nav], style={'border-bottom': '1px solid #bcbcbc', 'margin-bottom':'-1px'}), 
                  html.Br(),
                  html.H3(html.Strong('Comparison Tool')),
                  dcc.Markdown(comparison_description),
                  tooltip,
                  dbc.Row([html.H4('Choose the variables to show')]),
                  dbc.Row( dcc.Checklist(
    id = 'variable-boxes',
    options = ['Log Time', 'StepNo', 'CircuitName', 'TMP1', 'TMP2', 'B31', 'B32',
       'B21', 'B22', 'P101', 'RegulatorSP', 'RegulatorFB'],
    value = [],
    inline = False,
    inputStyle={"margin-right": "10px", 'margin-left': '20px'}
)),
                  html.Hr(),
                  html.H4('Plots'),
                  dbc.Row([dcc.Checklist(
    id = 'comparison-fig-boxes',
    options = ['Show Normalized Data', 'Show Ensemble Average'],
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
    Input('store-data', 'data'),
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

        if 'Show Ensemble Average' in checkbox_val:
            dfs = [df[col] for df in file_dict.values()]
            mean = pd.DataFrame(dfs).mean()

            if 'Show Normalized Data' in checkbox_val: 
                hovertemplates = {col: '<br>%{y:.3f}<br>Original Value = %{customdata[0]:.3f}<br>StepNo = %{customdata[2]}<br>'}
                custom_data = np.stack((np.array([[i] for i in df[f'{col}_raw']]), np.array([[str(i)] for i in df['Log Time Raw']]), np.array([[i] for i in df.StepNo]).astype(np.int64)), axis=1)
            else: 
                hovertemplates = {col: '<br>%{y:.3f}<br>StepNo = %{customdata[1]}<br>'}
                custom_data = np.stack((np.array([[str(i)] for i in df['Log Time Raw']]), np.array([[i] for i in df.StepNo]).astype(np.int64)), axis=1)
            fig.add_trace(go.Scatter(
                x=df['Log Time'], 
                y=mean, 
                name = col, 
                customdata=custom_data,  
                hovertemplate = hovertemplates[col], 
                legendgroup='mean',  # this can be any string, not just "group"
                legendgrouptitle_text='Ensemble Average',))
        
        fig.update_layout(fig_style, height=max(200*len(file_dict.keys()), 600))
        figshapes = get_stepNo_lines(df)
        
        fig.update_layout(shapes=figshapes )

        figs.append(html.P([html.Div([html.H5(f'Variable: {col}')], style={'margin-left':'2cm'}), dcc.Graph(id =key+col, figure = fig), html.Label('Note: x-axis is in hours from first timestamp')],style = box_style))
    return figs