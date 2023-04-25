# from jupyter_dash import JupyterDash
import dash
from dash import dcc, Dash, html, Input, Output, dash_table, State, ctx, callback
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose

dash.register_page(__name__,
                   path='/dataanalysis',  # represents the url text
                   name='Data Analysis',  # name of page, commonly used as name of link
                   title='Data Analysis'  # epresents the title of browser's tab
)
color = {'titlecolor': '#60893c', 'plot_background': '#efefef', 'gridcolor': 'c4c4c4', 'plotcolor2': 'dadada'}
nav = dbc.Nav(  className='greyNavpills',
    children = [
        dbc.NavItem([dbc.NavLink("Comparison", active="exact", href="dataanalysis/comparison_tool", style = {'margin-left':'5px', 'color': 'black', 'font-size': '20px', 'border':'1px solid #bcbcbc', 'border-bottom': '1px solid white'})]),
        dbc.NavItem(dbc.NavLink("One File Analysis", active="exact", href="dataanalysis/onefile", style = {'margin-left':'3px','color': 'black', 'font-size': '20px', 'border':'1px solid #bcbcbc', 'border-bottom': '1px solid white'})),
    ],
    pills=True,
)

one_file_description = """
The "One File Analysis"-tool has the following features:
- Information about a chosen StepNo
- Plot of the data, and plot of trend of the data
- Basic summary statistics
- Data table
- Correlation matrices
"""
comparison_description = """
The "Comparison"-tool has the following features:
- Plot of the data
- Ensemble Averaging
- Normalization of data to range from 0 to 1
"""
layout = dbc.Container([
        html.Div([nav], style={'border-bottom': '1px solid #bcbcbc', 'margin-bottom':'-1px'}),
        html.Br(),
        html.H6(html.Strong('Comparison Tool')),
        dcc.Markdown(comparison_description),
        html.Br(),
        html.H6(html.Strong('One File Analysis Tool')),
        dcc.Markdown(one_file_description)

], fluid=True)