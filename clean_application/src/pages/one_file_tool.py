import dash
from dash import dcc, Dash, html, Input, Output, dash_table, State, ctx, ALL, callback, Patch
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

#TO DO 
# Modify plot to have xaxis sync
# Redesign EDA section

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
        self.fig_style = dict(height = 800,
        title = None,
        plot_bgcolor = '#efefef', 
        legend = dict(title='', x=1.0, y=0.5),
        yaxis = dict(gridcolor = '#c4c4c4', title = None),
        xaxis = dict(gridcolor = '#c4c4c4', title = None),
        yaxis2 = dict(gridcolor = '#c4c4c4', title = None),
        xaxis2 = dict(gridcolor = '#c4c4c4', title = None),
        hovermode="x unified",
        margin = dict(b=10, t=5),
    )

_AppSetup = AppConstants('0.2.0')

dash.register_page(__name__,
                   path='/onefile',  # represents the url text
                   name='One file analysis tool',  # name of page, commonly used as name of link
                   title='One File Analysis'  # represents the title of browser's tab
)

def get_trend(df):
    dff = df.copy()
    for col in ['TMP1', 'TMP2', 'B31', 'B32','B21', 'B22', 'P101']:
        result=seasonal_decompose(dff[col], model='additive', period=6)
        for idx, item in enumerate(result.trend):
            if np.isnan(item):
                result.trend[idx] = df[col][idx]
        dff[col+'_trend'] = result.trend
    return dff

def get_traces(df, cols, group):
    hovertemplates = {col: '<br>Log Time=%{x}<br>value=%{y}<br>StepNo=%{customdata}<br>' for col in cols}
    custom_data = np.array([[i] for i in df.StepNo]).astype(np.int64)
    traces = [go.Scatter(x=df['Log Time'], y=df[col], name = col, customdata=custom_data,  hovertemplate = hovertemplates[col], legendgroup=group) for col in cols]
    return traces

def get_stepNo_lines(df):
    pos = [0]+[i for i in range(1,len(df.StepNo)) if df.StepNo[i] != df.StepNo[i-1]]+[len(df['Log Time'].unique())-1]
    dff = df.iloc[pos]
    color = ["rgba(255, 255, 255, 0.1)", "rgba(160, 160, 160, 0.1)"]*int(len(pos)*2)

    shapes = [dict(fillcolor=color[i],
            line={"width": 0},
            type="rect",
            x0=dff['Log Time'].iloc[i-1],
            x1=dff['Log Time'].iloc[i],
            xref="x",
            y0=0,
            y1=1,
            yref="paper")  for i in range(1,len(pos))]
    return shapes

def generate_id(fig_ids):
    id= ''
    id_list = np.random.choice(range(10), 10,replace=True)
    for i in id_list:
        id += str(i) 

    if id in fig_ids:
        while id in fig_ids:
            id = generate_id()
    return id

def make_histoboxplot(df, cols):
    figs = []
    for col in cols:
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.01, column_widths=[0.5, 0.5])
        fig.add_trace(go.Histogram(y=df[col], name=col),row=1,col=1)
        fig.add_trace(go.Box(y=df[col], name=col),row=1,col=2)
        fig.add_hline(df[col].mean(), line_color='black'
        )
        fig.add_annotation(
            xref="x domain",
            # yref="y domain",
            # The arrow head will be 25% along the x axis, starting from the left
            x=2,
            # The arrow head will be 40% along the y axis, starting from the bottom
            y=df[col].mean(),

            text=f"average: {df[col].mean():.2f}",
            # arrowhead=2,
            showarrow=False,
            yshift=10,
            xshift=10,
            row=1,

            col=1
        )
        fig.update_layout(dict(title=f'{col}, std:{df[col].std():.2f}\n'),showlegend=False)
        figs += [html.P(dcc.Graph(figure = fig), style = {'border': '1px solid #999', 'width': '50%', 'margin':'5px'})]
    return figs

digits, color, fig_style, box_style = _AppSetup.digits, _AppSetup.color, _AppSetup.fig_style, _AppSetup.box_style

one_file_description = """
Features:
- StepNo Information
- Trendfilter
- Plot of data
- Summary statistics
- Data table
"""
summary_stats_description = """
In Data Exploration one can visualise the distribution of a chosen variable in a chosen StepNo.\n
This is done through histograms(left) and boxplot(right)
    """
stepno_environment = dbc.Card([dbc.CardBody([
    dbc.Stack([
    dbc.Label(html.B('Click on StepNo to show more info')),
    html.Div(id='presdrop'),
    presdrop_info:=html.Div(id='presdrop_info')]),
    ])
    ], style={'border':'1px solid black', 'min-height': '13cm'})
fig_environment = html.Div([dbc.Row([ html.H1('Data Plots',style={'textAlign': 'center', 'color': color['titlecolor']})]),
    dbc.Row([dbc.Stack([
        html.B('Choose the variables in first figure', style={'text-align':'left'}),
        dcc.Checklist(style={'width': 'auto', 'font-size':'0.85em', 'margin-bottom':'1em'}, id = 'Fig1checkboxes', persistence = True, inline=True, persistence_type='session', inputStyle={"margin-right":"5px","margin-left": "5px"}),
        html.B('Choose the variables in second figure', style={'text-align':'left'}),
        dcc.Checklist(style={'width': 'auto', 'font-size':'0.85em', 'margin-bottom':'2em'}, id = 'Fig2checkboxes', persistence = True, inline=True, persistence_type='session', inputStyle={"margin-right":"5px","margin-left": "5px"})
        ])
        ]),
    dbc.Row([
    dcc.Checklist(id = 'checkbox',
    options = ['Show Only Trend In Data', 'Show Both'],
    value = [],
    inline = True,
    inputStyle={"margin-right": "10px", 'margin-left': '20px'})]),
    dcc.Graph(id='DataFig'),
    ])

#Maybe change to card
EDA_environment = html.P([
    dbc.Row([
        html.H1('Data Exploration',style={'textAlign': 'center', 'color': color['titlecolor']}),
        dcc.Markdown(children=[summary_stats_description], style={'width':'auto', 'text-align': 'left', 'lineHeight': '1.3em'}),
        html.B('Choose the variables in first figure', style={'text-align':'left'}),
        dcc.Checklist(style={'text-align': 'left', 'margin-bottom':'1em', 'width': 'auto', 'font-size':'0.85em'}, id = 'StatVarCheckboxes', persistence = True, inline=True, persistence_type='session', inputStyle={"margin-right":"5px","margin-left": "5px"}),
        html.B('Choose the variables in first figure', style={'text-align':'left'}),
        dcc.Checklist(id='state_drop', persistence = True, inline=True, style={'text-align': 'left', 'margin-bottom':'1em', 'width': 'auto', 'font-size':'0.85em'}, persistence_type='local', inputStyle={"margin-right":"5px","margin-left": "20px"}),
        html.Div(id='statfigs')
    ], style={'margin-bottom': '10px'})], style={'border': '1px solid #999'})

datatable_environment =  html.P([
    dbc.Row([html.H1('Data Table', style={'textAlign': 'center', 'color': color['titlecolor']})]),
    dbc.Stack([
    dbc.Row([
    html.B('Choose the columns to show', style={'text-align':'left'}),
    dbc.Stack([
    html.Button('Select All', className='btn btn-light btn-sm', title='Select All', id = {'type': 'SelectAllBtn', 'index': 'DataColumns'}),
    html.Button('Remove All', className='btn btn-light btn-sm', title='Remove All', id = {'type': 'RemoveAllBtn', 'index': 'DataColumns'}),
    ], direction='horizontal', gap=2,style={'margin-top':'-10px'}),
    dcc.Checklist(style={'width': 'auto','margin-top':'-10px', 'font-size':'0.85em'}, id = 'col_drop', persistence = True, inline=True, persistence_type='session', inputStyle={"margin-right":"5px","margin-left": "5px"}),
    ]),
    dbc.Row([
    html.B("Choose which StepNo's to show", style={'text-align':'left'}),
    dbc.Stack([
        html.Button('Select All', className='btn btn-light btn-sm', title='Select All', id = {'type': 'SelectAllBtn', 'index': 'StepNos'}),
        html.Button('Remove All', className='btn btn-light btn-sm', title='Remove All', id = {'type': 'RemoveAllBtn', 'index': 'StepNos'}),
    ], direction='horizontal', gap=2,style={'margin-top':'-10px'}),
    dbc.Col([dcc.Checklist(style={'width': 'auto','margin-top':'-10px', 'font-size':'0.8em', 'float': 'left'}, id='StepNo_drop', persistence = True, inline=True, persistence_type='session', inputStyle={"margin-right":"5px","margin-left": "5px"})               
                    ], width=9)]),
    dbc.Row([
        html.B('Show number of rows', style={'text-align':'left'}),
        dcc.Dropdown(value=10,clearable=False, options=[10,25,50,100], persistence=True, persistence_type='session',
                             style = {'width':'10em','margin-bottom': '10px'}, id='row_drop')])
    ], gap=1),
    
    dbc.Row([
    html.P([
    data_table := dash_table.DataTable(
        page_size= 10,
        style_cell=dict(textAlign='left'),
        style_header={'backgroundColor':'#c7dabf'},
        style_data={'overflow': 'hidden', 'textOverflow': 'ellipsis', 'maxWidth': 0, 'height': 'auto'},
        id = 'data_table'),
    ], style={'margin-top': '30px'}),
    ])], style = box_style)

#Layout  
main_container = [       
    stepno_environment,
    fig_environment,
    EDA_environment,
    datatable_environment,
]

layout = dbc.Container([html.H3(html.Strong('One File Analysis')),
    dcc.Markdown(one_file_description), html.Div(id='one_file_tool_page_view')], fluid=True)

@callback(
    Output('one_file_tool_page_view', 'children'),
    Input('store-raw-data', 'data')
)
def update_page(data):
    if len(data) == 0:
        return html.Div()
    return main_container

# @callback(
#         Output('col_drop', 'value'),
#         Output('col_drop', 'options'),
#         Output('StepNo_drop', 'value'),
#         Output('StepNo_drop', 'options'),
#         Output('state_drop', 'options'),
#         Output('state_drop', 'value'),
#         Output('StatVarCheckboxes', 'options'),
#         Output('StatVarCheckboxes', 'value'),
#         Output('presdrop', 'children'),
#         Output('Fig1checkboxes', 'options'),
#         Output('Fig2checkboxes', 'options'),
#         Output('Fig1checkboxes', 'value'),
#         Output('Fig2checkboxes', 'value'),
#         Input('store-one-file', 'data'),
#         Input('url', 'pathname'))
# def update_drops(json_dict, path, fig1_value = ['B21', 'B22', 'P101', 'RegulatorSP'], fig2_value = ['TMP1', 'TMP2', 'B31', 'B32']):

#     if json_dict == {}:
#         return dash.no_update

#     df = pd.DataFrame.from_dict(json_dict)
#     dff = df.copy()

#     step_options    = [i for i in dff.StepNo.unique()]
#     col_options     = [i for i in dff.columns]

#     StepNoInfo_drop = dbc.Tabs([dbc.Tab(label = str(StepNo), activeTabClassName="fw-bold", label_style={"color": "black"}, active_label_style={'font-size':'x-large'},  id = f'tab-{idx}') for idx, StepNo in enumerate(df.StepNo.unique())], active_tab='tab-0', id='presdroptabs')
    
#     variable_options = [col for col in col_options if col not in ['Log Time', 'CircuitName','StepNo']]

#     #StepNos in order of appearence
#     if len(df.query(f'StepNo=={df.StepNo[0]}'))>10:
#         stat_stepno_options = [df.StepNo[0] ]+[df.StepNo[i] for i in range(1,len(df.StepNo)) if df.StepNo[i] != df.StepNo[i-1] and len(df.query(f'StepNo=={df.StepNo[i]}'))>10]
#     else:
#         stat_stepno_options = [df.StepNo[i] for i in range(1,len(df.StepNo)) if df.StepNo[i] != df.StepNo[i-1] and len(df.query(f'StepNo=={df.StepNo[i]}'))>10]
#     #Filter out StepNos that appear more than once

#     #Remove StepNos that have small amount of datapoints

#     return col_options, col_options, step_options, step_options, stat_stepno_options, [], variable_options, [], StepNoInfo_drop, variable_options, variable_options, fig1_value, fig2_value

# @callback(
#     Output('presdrop_info', 'children'),
#     Input('presdroptabs', 'active_tab'),
#     State('store-one-file', 'data'),
#     prevent_initial_call=True
# )
# def pressure_drop_info(active, json_dict):
#     if json_dict == {}:
#         return html.Div()
    
#     df = pd.DataFrame.from_dict(json_dict)
#     dff = df.copy()

#     #StepNos
#     StepNo_chosen = [StepNo for idx, StepNo in enumerate(dff.StepNo.unique()) if f'tab-{idx}' == active][0]

#     StepNo_start= [0]+[i for i in range(1,len(dff.StepNo)) if dff.StepNo[i] != dff.StepNo[i-1]]
#     dff_filtered = dff.iloc[StepNo_start]
#     #Extract number of samples in each StepNo
#     nuniques = pd.Series([end-start for start, end in zip(dff_filtered.index[:-1], dff_filtered.index[1:])]+ [dff.index[-1]-dff_filtered.index[-1]], dff_filtered.index, name='Samples')
#     nuniques.iloc[-1] +=1

#     #Convert time axis to measure time duration of stepNos
#     convert_to_datetime_object = pd.to_datetime(list(dff_filtered['Log Time'].values)+[dff['Log Time'][df.index[-1]]], format='%Y-%m-%dT%H:%M:%S')
#     time_diffs = [(end-start).total_seconds() for start, end in zip(convert_to_datetime_object[:-1], convert_to_datetime_object[1:])]
    
#     #Convert to nice time format
#     time_converted = [list(divmod(divmod(time_diff,60)[0],60))+ [divmod(time_diff,60)[1]] for time_diff in time_diffs]
#     time_diff_text = pd.Series([f'{int(time[0])} h {int(time[1])} min {int(time[2])} s' for time in time_converted], index = dff_filtered.index, name= 'Duration')

#     #Dropout StepNo's with only one sample
#     dropout_idxs = [StepNo for StepNo, n in zip(dff_filtered.index, nuniques) if n==1]
#     idxs = np.array([(start,end-1) for start, end in zip(StepNo_start[:-1], StepNo_start[1:]) if start not in dropout_idxs]+ [[StepNo_start[-1], dff.index[-1]]])

#     cols = ['B22','B31', 'B32']
#     #starts|ends|diffs
#     starts = [pd.Series(dff['Log Time'][idxs[:,0]], name='Timestart')] + [pd.Series(dff[col][idxs[:,0]], name=col+'start').round(digits) for col in cols]
#     ends = [pd.Series([time for time in dff_filtered['Log Time'][idxs[:,1][:-1]+1]]+ [dff['Log Time'].iloc[-1]], index=dff.iloc[idxs[:,0]].index, name='Timeend')] + [pd.Series(dff[col][idxs[:,1]].values, name=col+'end', index=dff.iloc[idxs[:,0]].index).round(digits) for col in cols]

#     diffs = [pd.Series(time_diff_text[idxs[:,0]], name='Timechange')]+ [pd.Series((dff[col][idxs[:,1]].values - dff[col][idxs[:,0]].values), index=dff.iloc[idxs[:,0]].index, name=col+'change').round(digits) for col in cols]

#     #create as DataFrame
#     StepNoInfo = pd.DataFrame([nuniques]+starts+ends+diffs).rename(columns=dff.StepNo)

#     #convert to mbar
#     StepNoInfo.loc['B22change'] *=1000 

#     #Just some error handling
#     if type(StepNoInfo[StepNo_chosen].values[0]) ==np.ndarray:

#         StepNoInfo_vals = StepNoInfo[StepNo_chosen].values.T
#     else: 
#         StepNoInfo_vals = [StepNoInfo[StepNo_chosen].values.T]

#     the_text = []
#     for row in StepNoInfo_vals:
#         if int(row[0])==1:
#              the_text.append(dbc.Card([dbc.CardBody([dcc.Markdown(f""" 
# #### {StepNo_chosen}
# Samples: **{int(row[0])}**\n
# There is not more than one sample for this StepNo
# """)], className='card-text', style={'margin-right':'5px'})], style={'margin-right':'5px', 'margin-top': '5px'}))
#         else:
#             the_text.append(dbc.Card([dbc.CardBody([dcc.Markdown(f""" 
# #### {StepNo_chosen}
# Samples: **{int(row[0])}**\n
# Start: **{row[1]}, {row[2]:.2f} bar, B31 {row[3]:.2f}\u00B0C, B32 {row[4]:.2f}\u00B0C** \n
# Stop:  **{row[5]}, {row[6]:.2f} bar, B31 {row[7]:.2f}\u00B0C, B32 {row[8]:.2f}\u00B0C** \n
# Total time: **{row[9]}**\n
# Pressure difference: **{row[10]:.2f} mbar**  \n
# Temp B31 difference: **{row[11]:.2f}\u00B0C**\n
# Temp B32 difference: **{row[12]:.2f}\u00B0C**\n""")], className='card-text', style={'margin-right':'5px'})], style={'margin-right':'5px', 'margin-top': '5px'}))
#     return dbc.Stack(the_text, direction='horizontal')

# @callback(
#         Output('col_drop', 'value', allow_duplicate=True),
#         Output('StepNo_drop', 'value', allow_duplicate=True),
#         Input({'type': 'SelectAllBtn', 'index': ALL}, 'n_clicks'),
#         Input({'type': 'RemoveAllBtn', 'index': ALL}, 'n_clicks'),
#         State('col_drop', 'options'),
#         State('StepNo_drop','options'),
#         prevent_initial_call='initial_duplicate',
# )
# def select_remove_all(SelecAll, RemoveAll, col_options, stepno_options):
#     trigger_id = ctx.triggered_id
    
#     if trigger_id == None:
#         return dash.no_update
    
#     col_vals, stepno_vals = col_options, stepno_options
    
#     trigger_index, trigger_type = trigger_id.values()
#     if trigger_index == 'DataColumns':
#         if trigger_type == 'RemoveAllBtn':
#             col_vals = []
#         elif trigger_type == 'SelectAllBtn':
#             col_vals = col_options
#     elif trigger_index == 'StepNos':
#         if trigger_type == 'RemoveAllBtn':
#             stepno_vals = []
#         elif trigger_type == 'SelectAllBtn':
#             stepno_vals = stepno_options
#     return col_vals, stepno_vals

# @callback(
#     Output('data_table', 'data'),
#     Output('data_table', 'columns'),
#     Output('data_table', 'page_size'),
#     Input('StepNo_drop', 'value'),
#     Input('row_drop','value'),
#     Input('col_drop', 'value'),
#     Input('store-one-file', 'data'),
#     prevent_initial_call=True
#     )
# def update_datatable(StepNo_vs ,row_v, col_vs, json_dict):
#     if json_dict == {}:
#         return pd.DataFrame().to_dict('records'), [], 10

#     df = pd.DataFrame.from_dict(json_dict)
#     dff = df.copy()

#     chosen_cols = [{"name": i, "id": i} for i in dff.columns[dff.columns.isin(col_vs)]]

#     if StepNo_vs or col_vs:
#         dff = dff[dff.StepNo.isin(StepNo_vs)][col_vs].round(digits)
        

#     return dff.to_dict('records'), chosen_cols, row_v

# @callback( 
#     Output('DataFig', 'figure'),
#     Input('Fig1checkboxes', 'value'),
#     Input('Fig2checkboxes', 'value'),
#     Input('checkbox', 'value'),
#     Input('store-one-file', 'data'),
#     ) 
# def figs_from_store(fig1cols, fig2cols, show_trend, json_dict):

#     df = pd.DataFrame.from_dict(json_dict)
#     dff =df.copy()
#     dff = get_trend(dff)

#     cols1 = fig1cols
#     cols2 = fig2cols
#     fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
#     if 'Show Only Trend In Data' in show_trend:

#         for trace in get_traces(dff, [col + '_trend' if col + '_trend' in dff.columns else col  for col in cols1],'group1'):
#             fig.add_trace(trace, row=1, col=1)
#         for trace in get_traces(dff, [col + '_trend' if col + '_trend' in dff.columns else col for col in cols2], 'group2'):
#             fig.add_trace(trace, row=2, col=1)
#     elif 'Show Both' in show_trend:   
#         for trace in get_traces(dff, cols1 + [col + '_trend' if col + '_trend' in dff.columns else col  for col in cols1], 'group1'):
#             fig.add_trace(trace, row=1, col=1)
#         for trace in get_traces(dff, cols2 + [col + '_trend' if col + '_trend' in dff.columns else col for col in cols2], 'group2'):
#             fig.add_trace(trace, row=2, col=1)
#     else:        
#         for trace in get_traces(dff, cols1, 'group1'):
#             fig.add_trace(trace, row=1, col=1)
#         for trace in get_traces(dff, cols2, 'group2'):
#             fig.add_trace(trace, row=2, col=1)
#     fig.update_layout(_AppSetup.fig_style)
#     fig.update_layout(legend=dict(groupclick="toggleitem", y=0.5))

#     figshapes = get_stepNo_lines(dff) 
#     fig.update_layout(shapes=figshapes )

#     return fig

# @callback(
#     Output('statfigs', 'children'),
#     Input('StatVarCheckboxes', 'value'),
#     Input('state_drop', 'value'),
#     State('store-one-file', 'data'),
# )
# def statisticFigures(variables, states, json_dict):
#     if len(variables)== 0:
#         return dash.no_update
#     elif len(states) == 0:
#         return dash.no_update

#     df = pd.DataFrame.from_dict(json_dict)

#     figs_collection = []
#     for state in states:
#         df_filtered = df.query(f'StepNo=={state}')
#         figs = make_histoboxplot(df_filtered, variables)

#         grid = [html.B('StepNo: '+str(state), style={'font-size': '2em', 'text-align':'center'})]
#         for i in range(0, int(len(variables)*len(states)),2):
#             if len(variables)-i == 1:
#                 row = dbc.Stack(children=[figs[i], html.Div()], direction="horizontal")
#             else: 
#                 row = dbc.Stack(figs[i:i+2], direction="horizontal")
#             grid += [row]
#         figs_collection += grid
        
#     return figs_collection

