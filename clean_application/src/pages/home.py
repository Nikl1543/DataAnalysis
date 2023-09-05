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

# To create meta tag for each page, define the title, image, and description.
dash.register_page(__name__,
                   path='/',  # '/' is home page and it represents the url
                   name='Home',  # name of page, commonly used as name of link
                   title='Home',  # title that appears on browser's tab
                   description='This is the home page, contains links to the other pages.'
)

# page 1 data

layout = html.Div([
    html.H2('About This Application'),
    dcc.Markdown("""
    The purpose of this application is fast data analysis. The application offers the option to upload multiple .csv files and the possibility to quickly switch between files, a one file analysis tool, a comparison tool and a PDF report generator.
    """)
])