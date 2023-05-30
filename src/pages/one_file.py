# from jupyter_dash import JupyterDash
import dash
from dash import dcc, Dash, html, Input, Output, dash_table, State, ctx, callback, ALL
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose

dash.register_page(__name__,
                   path='/dataanalysis/onefile',  # represents the url text
                   name='One file analysis tool',  # name of page, commonly used as name of link
                   title='One File Analysis'  # represents the title of browser's tab
)

# page 2 data
digits = 2
color = {'titlecolor': '#60893c', 'plot_background': '#efefef', 'gridcolor': 'c4c4c4', 'plotcolor2': 'dadada'}
fig_style = dict(height = 1080,
        title = None,
        plot_bgcolor = '#efefef', 
        legend = dict(title='', x=1.0, y=0.5),
        # yaxis = dict(gridcolor = '#c4c4c4', title = None),
        # xaxis = dict(gridcolor = '#c4c4c4', title = None),
        hovermode="x unified",
        margin = dict(b=10, t=5),
    )
box_style = {'margin': '10px', 'padding': '5px','border-bottom': '3px solid #bcbcbc'}

def get_trend(df):
    dff = df.copy()
    for col in ['TMP1', 'TMP2', 'B31', 'B32','B21', 'B22', 'P101']:
        result=seasonal_decompose(dff[col], model='additive', period=5)
        for idx, item in enumerate(result.trend):
            if np.isnan(item):
                result.trend[idx] = df[col][idx]
        dff[col+'_trend'] = result.trend
    return dff

def get_yaxis(ctx_input_and_id):

    if 'yaxis.range[0]' in ctx_input_and_id:
        yaxis = [ctx_input_and_id['yaxis.range[0]'], ctx_input_and_id['yaxis.range[1]']]
    else: 
        yaxis = 'auto'
    return yaxis

def compute_time_diff(t1, t2, tf):
    time_diff = (datetime.strptime(t2, tf)- datetime.strptime(t1, tf)).total_seconds()
        
    return time_diff

def get_traces(df, cols):
    hovertemplates = {col: '<br>Log Time=%{x}<br>value=%{y}<br>StepNo=%{customdata}<br>' for col in cols}
    custom_data = np.array([[i] for i in df.StepNo]).astype(np.int64)
    traces = [go.Scatter(x=df['Log Time'], y=df[col], name = col, customdata=custom_data,  hovertemplate = hovertemplates[col]) for col in cols]
    return traces

def get_stepNo_lines(df):
    StepNo_labels = [0]+[i for i in range(1,len(df.StepNo)) if df.StepNo[i] != df.StepNo[i-1]]+[len(df['Log Time'].unique())-1]
    dff = df.iloc[StepNo_labels]
    pos = dff.index
    color = ["rgba(255, 255, 255, 0.2)", "rgba(160, 160, 160, 0.2)"]*int(len(dff)*2)

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

def extract_info_from_filename(file):
    if len(file) >1:
        filename, *file_info = file
        file_info = html.Table([html.Tr([html.Th(html.Strong(i), style = {'padding': '10px'}) for i in file_info])], style={'padding': '10px', 'overflow': 'hidden', 'textOverflow': 'ellipsis', 'maxWidth': 0, 'height': 'auto'})

        return html.P([
            dbc.Row([dcc.Markdown(f'Filename: **{filename}** \n\n Details:'), 
            dbc.Row([file_info])])], 
            style = box_style)
    else:
        return html.P([dbc.Row([dcc.Markdown(f'**Filename**: {file[0]}')])
        ], style = box_style)

nav = dbc.Nav(
    [
        dbc.NavItem([dbc.NavLink("Comparison", active="exact", href="/dataanalysis/comparison_tool", style = {'margin-left':'3px', 'color': 'black', 'font-size': '20px'})]),
        dbc.NavItem(dbc.NavLink("One File Analysis", active="exact", href="/dataanalysis/onefile", style = {'margin-left':'3px', 'color': 'black', 'font-size': '20px', 'border':'1px solid #bcbcbc', 'border-bottom': '1px solid white'})),
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
main_container = [
    html.Div([nav], style={'border-bottom': '1px solid #bcbcbc', 'margin-bottom':'-1px'}),
    html.Br(),
    html.H6(html.Strong('One File Analysis Tool')),
    dcc.Markdown(one_file_description),        
    dbc.Card([dbc.CardBody([
    dbc.Stack([
    dbc.Label(html.B('Click on StepNo to show more info')),
    html.Div(id='presdrop'),
    presdrop_info:=html.Div(id='presdrop_info')]),
    ])
    ], style={'border':'1px solid black', 'min-height': '13cm'}),
    
    dbc.Row([ html.H1('Data Plots',style={'textAlign': 'center', 'color': color['titlecolor']})]),
    dbc.Row([dcc.Checklist(id = 'checkbox',
    options = ['Show Only Trend In Data', 'Show Both'],
    value = [],
    inline = True,
    inputStyle={"margin-right": "10px", 'margin-left': '20px'}
)]),
    dbc.Row([
        dcc.Graph(
            id='DataFig',
    )]),

    html.P([
    dbc.Row([
        html.P([ html.H1('Data Exploration',style={'textAlign': 'center', 'color': color['titlecolor']}),
        describe_table := dcc.Markdown("""

    List of statistics included in table:
    - number of samples
    - mean value
    - standard deviation 
    - minimum value
    - the 25% median
    - the 50% median
    - the 75% median
    - maximum value

    The options are the different StepNo's, and a table of the statistics for the specific StepNo will be included.
        """),
        stat_drop := dcc.Checklist(id='stat_drop', persistence = True, inline=True, persistence_type='local', inputStyle={"margin-right":"5px","margin-left": "20px"}),
    ], style={'margin-bottom': '10px'})]),
    dbc.Row([stat_table := html.P(id='stat_table')]),
    ], style = box_style),
    html.P([
    dbc.Row([html.H1('Data Table', style={'textAlign': 'center', 'color': color['titlecolor']})]),
    dbc.Stack([
    dbc.Row([
    dbc.Label(html.B('Choose the columns to show')),
    col_drop := dcc.Checklist(style={'width': '100%', 'color': '#999'}, id = 'col_drop', persistence = True, inline=True, persistence_type='session', inputStyle={"margin-right":"5px","margin-left": "5px"}),
    ]),
    html.Br(),
    dbc.Row([
    StepNo_droptext := dbc.Label(html.B("Choose which StepNo's to show")),
    StepNo_drop := dcc.Checklist(style={'width': '100%'}, id='StepNo_drop', persistence = True, inline=True, persistence_type='session', inputStyle={"margin-right":"5px","margin-left": "5px"})               
                    ]),
    html.Br(),
    dbc.Row([dbc.Label(html.B('Show number of rows')),
    row_drop := dcc.Dropdown(value=10,clearable=False, options=[10,25,50,100], persistence=True, persistence_type='session',
                             style = {'width': '50%', 'margin-bottom': '5px', 'direction': 'up'}, id='row_drop')])
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
    ])], style = box_style),
    dbc.Row(html.Div(style={'height': '60px',}))
]

layout = dbc.Container(main_container, fluid=True)

@callback(
        Output('col_drop', 'value'),
        Output('col_drop', 'options'),
        Output('StepNo_drop', 'value'),
        Output('StepNo_drop', 'options'),
        Output('stat_drop', 'options'),
        Output('stat_drop', 'value'),
        Output('presdrop', 'children'),
        Input('store-one_file', 'data'),
        Input('url', 'pathname'))
def update_drops(stored_data, path):

    df = pd.DataFrame.from_dict(stored_data)
    dff = df.copy()

    step_options    = [i for i in dff.StepNo.unique()]
    col_options     = [i for i in dff.columns]

    StepNoInfo_drop = dbc.Tabs([dbc.Tab(label = str(StepNo), activeTabClassName="fw-bold", label_style={"color": "black"}, active_label_style={'font-size':'x-large'},  id = f'tab-{idx}') for idx, StepNo in enumerate(df.StepNo.unique())], active_tab='tab-0', id='presdroptabs')
    return (col_options, col_options, step_options, step_options,
            step_options, [], StepNoInfo_drop)



@callback(
    Output('presdrop_info', 'children'),
    Input('presdroptabs', 'active_tab'),
    State('store-one_file', 'data'),
    prevent_initial_call=True
)
def pressure_drop_info(active, stored_data):
    if active == None:
        return dash.no_update
    
    df = pd.DataFrame.from_dict(stored_data)
    dff = df.copy()

    StepNo_chosen = [StepNo for idx, StepNo in enumerate(dff.StepNo.unique()) if f'tab-{idx}' == active][0]

    StepNo_start= [0]+[i for i in range(1,len(dff.StepNo)-1) if dff.StepNo[i] != dff.StepNo[i-1]]
    dff_filtered = dff.iloc[StepNo_start]
    nuniques = pd.Series([end-start for start, end in zip(dff_filtered.index[:-1], dff_filtered.index[1:])]+ [dff_filtered.index[-1]-StepNo_start[-1]], dff_filtered.index, name='Samples')
    nuniques.iloc[-1] +=1

    drop_dates = [time.replace('.', ':').split(' ')[-1] for time in dff_filtered['Log Time']]+ [dff['Log Time'].iloc[-1].replace('.', ':').split(' ')[-1]]
    time_axis = [0]+[(datetime.strptime(time, '%H:%M:%S')-datetime.strptime(drop_dates[0], '%H:%M:%S')).seconds for time in drop_dates[1:]]
    time_diffs = [end-start for start, end in zip(time_axis[:-1], time_axis[1:])]
    time_converted = [list(divmod(divmod(time_diff,60)[0],60))+ [divmod(time_diff,60)[1]] for time_diff in time_diffs]
    time_diff_text = pd.Series([f'{int(time[0])} h {int(time[1])} min {int(time[2])} s' for time in time_converted], index = dff_filtered.index, name= 'Duration')

    dropout_idxs = [StepNo for StepNo, n in zip(dff_filtered.index, nuniques) if n==1]
    idxs = np.array([(start,end-1) for start, end in zip(StepNo_start[:-1], StepNo_start[1:]) if start not in dropout_idxs]+ [[StepNo_start[-1], dff.index[-1]]])

    cols = ['B22','B31', 'B32']
    #starts|ends|diffs
    starts = [pd.Series(dff['Log Time'][idxs[:,0]], name='Timestart')] + [pd.Series(dff[col][idxs[:,0]], name=col+'start').round(digits) for col in cols]
    ends = [pd.Series([time for time in dff_filtered['Log Time'][idxs[:,1][:-1]+1]]+ [dff['Log Time'].iloc[-1]], index=dff.iloc[idxs[:,0]].index, name='Timeend')] + [pd.Series(dff[col][idxs[:,1]].values, name=col+'end', index=dff.iloc[idxs[:,0]].index).round(digits) for col in cols]

    diffs = [pd.Series(time_diff_text[idxs[:,0]], name='Timechange')]+ [pd.Series((dff[col][idxs[:,1]].values - dff[col][idxs[:,0]].values), index=dff.iloc[idxs[:,0]].index, name=col+'change').round(digits) for col in cols]


    StepNoInfo = pd.DataFrame([nuniques]+starts+ends+diffs).rename(columns=dff.StepNo)
    StepNoInfo.loc['B22change'] *=1000 

    if type(StepNoInfo[StepNo_chosen].values[0]) ==np.ndarray:

        StepNoInfo_vals = StepNoInfo[StepNo_chosen].values.T
    else: 
        StepNoInfo_vals = [StepNoInfo[StepNo_chosen].values.T]

    the_text = []
    for row in StepNoInfo_vals:
        if int(row[0])==1:
             the_text.append(dbc.Card([dbc.CardBody([dcc.Markdown(f""" 
#### {StepNo_chosen}
Samples: **{int(row[0])}**\n
There is not more than one sample for this StepNo
""")], className='card-text', style={'margin-right':'5px'})], style={'margin-right':'5px', 'margin-top': '5px'}))
        else:
            the_text.append(dbc.Card([dbc.CardBody([dcc.Markdown(f""" 
#### {StepNo_chosen}
Samples: **{int(row[0])}**\n
Start: **{row[1]}, {row[2]:.2f} bar, B31 {row[3]:.2f}\u00B0C, B32 {row[4]:.2f}\u00B0C** \n
Stop:  **{row[5]}, {row[6]:.2f} bar, B31 {row[7]:.2f}\u00B0C, B32 {row[8]:.2f}\u00B0C** \n
Total time: **{row[9]}**\n
Pressure change: **{row[10]:.2f} mbar**  \n
Temp B31 change: **{row[11]:.2f}\u00B0C**\n
Temp B32 change: **{row[12]:.2f}\u00B0C**\n""")], className='card-text', style={'margin-right':'5px'})], style={'margin-right':'5px', 'margin-top': '5px'}))
    return dbc.Stack(the_text, direction='horizontal')

@callback(
    Output('data_table', 'data'),
    Output('data_table', 'columns'),
    Output('data_table', 'page_size'),
    Input('StepNo_drop', 'value'),
    Input('row_drop','value'),
    Input('col_drop', 'value'),
    Input('store-one_file', 'data'),
    prevent_initial_call=True
    )
def update_datatable(StepNo_vs ,row_v, col_vs, stored_data):

    df = pd.DataFrame.from_dict(stored_data)
    dff = df.copy()

    chosen_cols = [{"name": i, "id": i} for i in dff.columns[dff.columns.isin(col_vs)]]

    if StepNo_vs or col_vs:
        dff = dff[dff.StepNo.isin(StepNo_vs)][col_vs].round(digits)
        

    return dff.to_dict('records'), chosen_cols, row_v


@callback( 
    Output('DataFig', 'figure'),
    # Output('relayout-data', 'children'),
    Input('checkbox', 'value'),
    Input('store-one_file', 'data'),
    prevent_initial_call=True
    ) 
def figs_from_store(show_trend, stored_data):

    df = pd.DataFrame.from_dict(stored_data)
    df['Log Time'] = [i.split(' ')[1] for i in df['Log Time']]
    df = get_trend(df)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    cols1 = ['B21', 'B22', 'P101', 'RegulatorSP']
    cols2 = ['TMP1', 'TMP2', 'B31', 'B32']
    if 'Show Only Trend In Data' in show_trend:

        for trace in get_traces(df, [col + '_trend' if col + '_trend' in df.columns else col  for col in cols1]):
            fig.add_trace(trace, row=1,col=1)
        for trace in get_traces(df, [col + '_trend' if col + '_trend' in df.columns else col for col in cols2]):
            fig.add_trace(trace, row=2,col=1)
    elif 'Show Both' in show_trend:   
        for trace in get_traces(df, cols1 + [col + '_trend' if col + '_trend' in df.columns else col  for col in cols1]):
            fig.add_trace(trace, row=1,col=1)
        for trace in get_traces(df, cols2 + [col + '_trend' if col + '_trend' in df.columns else col for col in cols2]):
            fig.add_trace(trace, row=2,col=1)

    else:        
        for trace in get_traces(df, cols1):
            fig.add_trace(trace, row=1,col=1)
        for trace in get_traces(df, cols2):
            fig.add_trace(trace, row=2,col=1)

    fig.update_layout(fig_style)
    figshapes = get_stepNo_lines(df)
    
    fig.update_layout(shapes=figshapes )

    return fig

@callback(
    Output('stat_table', 'children'),
    Input('store-one_file', 'data'),
    Input('stat_drop', 'value'),
    prevent_initial_call=True
)
def description_table(json_dict, stat_vs):

    df = pd.DataFrame.from_dict(json_dict)

    tables = []
    for table in stat_vs:
        data_df = df.query(f'StepNo =={table}').drop(['StepNo'], axis=1).describe().round(digits)
        data_df = pd.concat([pd.Series([i for i in data_df.index], index=data_df.index, name=f'StepNo: {table}'), data_df], axis=1)
        tables.append(dash_table.DataTable(
        columns=[{"name": i, "id": i} 
                    for i in data_df.columns],
        data=data_df.to_dict('records'),
        style_cell=dict(textAlign='left'),
        style_header={'backgroundColor':'#c7dabf'},
        style_data={'overflow': 'hidden', 'textOverflow': 'ellipsis', 'maxWidth': 0, 'height': 'auto'}
        ))

        
        tables.append(html.Hr())

    return tables