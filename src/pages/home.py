import dash
from dash import dcc, html, callback, Output, Input
import plotly.express as px
import dash_bootstrap_components as dbc

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
    """),
    dcc.Markdown("""
Links to other pages:\n"""
), html.Div([dcc.Link('Data Analysis', href = '/dataanalysis')], style = {'margin-bottom': '5px'}), 
   html.Div([dcc.Link('Comparison Tool', href='/dataanalysis/comparison_tool')], style = {'margin-left': '1cm', 'margin-bottom': '2px'}), 
   html.Div([dcc.Link('One File Analysis Tool', href='/dataanalysis/comparison_tool')], style = {'margin-left': '1cm', 'margin-bottom': '5px'}), 
   html.Div([dcc.Link('Report', href = '/report')]),

])