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
fig_style = dict(height = 560,
        title = None,
        plot_bgcolor = '#efefef', 
        legend = dict(title='', x=1.0, y=0.5),
        yaxis = dict(gridcolor = '#c4c4c4', title = None),
        xaxis = dict(gridcolor = '#c4c4c4', title = None),
        hovermode="x unified",
        margin = dict(b=10, t=5),
    )
box_style = {'margin': '10px', 'padding': '5px','border-bottom': '3px solid #bcbcbc'}

def get_trend(df):
    dff = df.copy()
    for col in ['TMP1', 'TMP2', 'B31', 'B32','B21', 'B22']:
        result=seasonal_decompose(dff[col], model='additive', period=6)
        for idx, item in enumerate(result.trend):
            if np.isnan(item):
                result.trend[idx] = df[col][idx]
        dff[col+' trend'] = result.trend
    return dff

def corr_matrix(df, columns, title):
       df_corr = df[columns].corr().round(2)

       fig = px.imshow(df_corr, text_auto=True, title = title,color_continuous_scale=[[0, '#006600'], [0.35, '#FFFFFF'], [0.5, 'white'], [0.65, '#FFFFFF'], [1.0, '#006600']], zmin =-1, zmax=1, aspect="auto")
       fig.update_traces(hovertemplate='%{x} - %{y}<br>Coefficient %{z}<extra></extra>')
       fig.update_xaxes(side='top')

       return fig

def get_yaxis(ctx_input_and_id):

    if 'yaxis.range[0]' in ctx_input_and_id:
        yaxis = [ctx_input_and_id['yaxis.range[0]'], ctx_input_and_id['yaxis.range[1]']]
    else: 
        yaxis = 'auto'
    return yaxis

def sync_figs_and_slider(inputs, trigger):
    inputs, trigger = ctx.inputs, [key for key in ctx.triggered_prop_ids.keys()]

    slider_vals, slider_min, slider_max = inputs['plot_rangeSlider.value'], inputs['plot_rangeSlider.min'], inputs['plot_rangeSlider.max']

    if 'plot_rangeSlider.min' in trigger:
        update_slider = [slider_min, slider_max]

    elif 'fig1.relayoutData' in trigger:
        if 'xaxis.range[0]' in inputs['fig1.relayoutData']:
            update_slider = [inputs['fig1.relayoutData']['xaxis.range[0]'], inputs['fig1.relayoutData']['xaxis.range[1]']]
        else:
            update_slider = [slider_min, slider_max]
    elif 'fig2.relayoutData' in trigger:
        if 'xaxis.range[0]' in inputs['fig2.relayoutData']:
            update_slider = [inputs['fig2.relayoutData']['xaxis.range[0]'], inputs['fig2.relayoutData']['xaxis.range[1]']]
        else:
            update_slider = [slider_min, slider_max]
    else: 
        update_slider = slider_vals

    xaxis = update_slider
    yaxes = [get_yaxis(inputs['fig1.relayoutData']), get_yaxis(inputs['fig2.relayoutData'])]
    return xaxis, yaxes, update_slider

def compute_time_diff(t1, t2, tf):
    time_diff = (datetime.strptime(t2, tf)- datetime.strptime(t1, tf)).total_seconds()
        
    return time_diff

def get_legend_info(info_legend, show_trend, inputs, id):
    legend_info = info_legend[id]

    if 'Show Both' in show_trend:
        figstate = 'both'
    elif 'Show Only Trend In Data' in show_trend: 
        figstate = 'trend'
    else:
        figstate = 'data'

    legend_info = legend_info[figstate]
    
    if inputs[id] != None:
        for idx, i in enumerate(inputs[id][1]):
            legend_info['legend_state'][i] = inputs[id][0]['visible'][idx]

    visible_state = {col: state for col, state in zip(legend_info['cols'],legend_info['legend_state'])}

    info_legend[id][figstate]['legend_state'] = legend_info['legend_state']
    for key in info_legend.keys():
        if key != id:
            for state in info_legend[key].keys():
                if state != figstate:
                    info_legend[key][state]['legend_state'] = [True for i in range(len(info_legend[key][state]['legend_state']))]

    return visible_state, legend_info['cols'], info_legend

def get_traces(df, cols, visible_state):
    hovertemplates = {col: '<br>Log Time=%{x}<br>value=%{y}<br>StepNo=%{customdata}<br>' for col in cols}
    custom_data = np.array([[i] for i in df.StepNo]).astype(np.int64)
    traces = [go.Scatter(x=df['Log Time'], y=df[col], name = col, customdata=custom_data,  hovertemplate = hovertemplates[col], visible =visible_state[col]) for col in cols]
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
    
    html.P([
    dbc.Row([ html.H1('Data Plots',style={'textAlign': 'center', 'color': color['titlecolor']})]),
    dbc.Row([dcc.Checklist(id = 'checkbox',
    options = ['Show Only Trend In Data', 'Show Both'],
    value = [],
    inline = False,
    inputStyle={"margin-right": "10px", 'margin-left': '20px'}
)]),
    dbc.Row([
        dcc.Graph(
            id='fig2',
        ),
    dbc.Row([
    dbc.Col([ ], width=1),
    dbc.Col([
        html.P([
            plot_rangeSlider := dcc.RangeSlider(0,100, value= [0,64.2194], updatemode='mouseup', id= 'plot_rangeSlider', pushable=20, persistence = True, persistence_type='local')], style ={'margin-top':'34px', 'margin-bottom':'34px'}),
    ], width=9),
    ]),
        dcc.Graph(
            id='fig1',
        ),
    ])], style = box_style),
    html.Label(str({ "fig2.restyleData" :{
        'data': {'legend_state': [True]*4, 
                 'cols': ['B21', 'B22', 'RegulatorSP', 'RegulatorFB']}, 
        'trend': {'legend_state': [True]*4, 
                 'cols': ['B21 trend', 'B22 trend', 'RegulatorSP', 'RegulatorFB']},  
        'both': {'legend_state': [True]*6, 
                 'cols': ['B21', 'B22', 'B21 trend', 'B22 trend', 'RegulatorSP', 'RegulatorFB']}
    },
       "fig1.restyleData" :{
        'data': {'legend_state': [True]*4, 
                 'cols': ['TMP1', 'TMP2', 'B31', 'B32']}, 
        'trend': {'legend_state': [True]*4, 
                 'cols': ['TMP1 trend', 'TMP2 trend', 'B31 trend', 'B32 trend']},  
        'both': {'legend_state': [True]*8, 
                 'cols': ['TMP1', 'TMP2', 'B31', 'B32', 'TMP1 trend',
       'TMP2 trend', 'B31 trend', 'B32 trend']}
       }
       }),
        style = {'color': '#Ffffff', 'font-size': '2pt'}, id = 'legend_placeholder'),
    # html.Pre(id='relayout-data'),
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

    In the dropdown above the table the options are the different StepNo's, and a table of the statistics for the specific StepNo will be included.
    There is also the option to choose "All" which will include a table of the global statistics.

        """),
        stat_drop := dcc.Checklist(id='stat_drop', persistence = True, persistence_type='local', inputStyle={"margin-right":"5px","margin-left": "20px"}),
    ], style={'margin-bottom': '10px'})]),
    dbc.Row([stat_table := html.P(id='stat_table')]),
    ], style = box_style),
    html.P([
    dbc.Row([html.H1('Data Table', style={'textAlign': 'center', 'color': color['titlecolor']}),
    dbc.Col([
    dbc.Label(html.B('Choose the columns to show')),
    col_drop := dcc.Checklist(style={'width': '100%', 'color': '#999'}, id = 'col_drop', persistence = True, persistence_type='session', inputStyle={"margin-right":"5px","margin-left": "5px"}),
    html.Br(),],width=3),
    dbc.Col([
    StepNo_droptext := dbc.Label(html.B("Choose which StepNo's to show")),
    StepNo_drop := dcc.Checklist(style={'width': '100%'}, id='StepNo_drop', persistence = True, persistence_type='session', inputStyle={"margin-right":"5px","margin-left": "5px"})               
                    ],width=3),
    dbc.Col([dbc.Label(html.B('Show number of rows')),
    row_drop := dcc.Dropdown(value=10,clearable=False, options=[10,25,50,100], persistence=True, persistence_type='session',
                             style = {'width': '50%', 'margin-bottom': '5px', 'direction': 'up'}, id='row_drop')], width=3)
    ]),
    
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

    html.P([
    dbc.Row([html.H1('A Word On Correlation',style={'textAlign': 'center', 'color': color['titlecolor']})]),
    dbc.Row([dcc.Markdown('''
    Definition of sample correlation coefficient: $\\text{coef}= \\frac{cov(X,Y)}{\sigma_X\sigma_Y}$
    Correlation is a statistic describing the dependency between parameters. Consider B31 and B22, if the correlation coefficient
    is $0.6$ then knowing B31, one can say that $0.6^2 \\times 100 = 36\%$ of the variance in B22 can be "explained" by B31. \n\n

    The tables presenting the Pearson correlation coefficient between the variables are computed with different data cleanings:
    In order of appearence from left to right, the correlation coefficient is computed using:
    - No data cleaning
    - Using the trends in the data
    - Sorting out data points with large residuals

    This was done in order to maybe get different "pictures" of the dependcies between variables
    ''', mathjax=True
    )]),
    dbc.Row(id='corr_table')] ,style = box_style),
    dbc.Row(html.Div(style={'height': '60px',}))
]

layout = dbc.Container(main_container, fluid=True)

@callback(
        Output('col_drop', 'value'),
        Output('col_drop', 'options'),
        Output('StepNo_drop', 'value'),
        Output('StepNo_drop', 'options'),
        Output('plot_rangeSlider', 'min'),
        Output('plot_rangeSlider', 'max'),
        Output('plot_rangeSlider', 'marks'),
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
    slider_min      = 0
    slider_max      = len(dff['Log Time'].unique())
    StepNo_start = [0]+[i for i in range(1,len(dff.StepNo)) if dff.StepNo[i] != dff.StepNo[i-1]]
    StepNo_names = [str(i) for i in df.StepNo[StepNo_start]]
    adjust_pos = [0, -30, 12, -42, 24, -54, 36, -66]
    mark_pos = [f'{adjust_pos[i%8]}px' for i in range(len(StepNo_start)+1)]
    slider_marks = {idx: {
        "label": name,
        "style": {"margin-top": markpos, "white-space": "nowrap"},
    } for name, idx, markpos in zip(StepNo_names, StepNo_start, mark_pos)}
    StepNoInfo_drop = dbc.Tabs([dbc.Tab(label = str(StepNo), activeTabClassName="fw-bold", label_style={"color": "black"}, active_label_style={'font-size':'x-large'},  id = f'tab-{idx}') for idx, StepNo in enumerate(df.StepNo.unique())], active_tab='tab-0', id='presdroptabs')
    return (col_options, col_options, step_options, step_options, slider_min, slider_max, slider_marks, 
            ['All']+step_options, ['All'], StepNoInfo_drop)



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
    Output('fig1', 'figure'),
    Output('fig2', 'figure'),
    Output('plot_rangeSlider', 'value'),
    Output('legend_placeholder', 'children'),
    # Output('relayout-data', 'children'),
    Input('fig1', 'relayoutData'),
    Input('fig2', 'relayoutData'),
    Input('fig1', 'restyleData'),
    Input('fig2', 'restyleData'),
    Input('plot_rangeSlider', 'value'),
    Input('plot_rangeSlider', 'min'),
    Input('plot_rangeSlider', 'max'),
    Input('checkbox', 'value'),
    Input('legend_placeholder', 'children'),
    Input('store-one_file', 'data'),
    Input('upload', 'filename'),
    prevent_initial_call=True
    ) 
def figs_from_store(relayoutData1, relayoutData2, restyle1, restyle2, slider_vals, slider_min, slider_max, show_trend, legend_info, stored_data, update_state):

    if update_state == None:
        raise dash.exceptions.PreventUpdate

    legend_info = eval(legend_info)

    df = pd.DataFrame.from_dict(stored_data)
    df['Log Time'] = [i.split(' ')[1] for i in df['Log Time']]
    df = get_trend(df)

    xaxis, yaxes, update_slider = sync_figs_and_slider(ctx.inputs, [key for key in ctx.triggered_prop_ids.keys()])

    visible_state2, cols2, legend_info = get_legend_info(legend_info, show_trend, ctx.inputs, "fig2.restyleData")
    visible_state1, cols1, legend_info = get_legend_info(legend_info, show_trend, ctx.inputs, "fig1.restyleData")
    fig2 = go.Figure()
    for trace in get_traces(df, cols2, visible_state2):
        fig2.add_trace(trace)

    fig1 = go.Figure()
    for trace in get_traces(df, cols1, visible_state1):
        fig1.add_trace(trace)
    
    fig2.update_layout(fig_style)
    if 'Show Only Trend In Data' in show_trend or 'Show Both' in show_trend:
        fig1.update_layout(fig_style
        )
    else:
        fig1.update_layout(fig_style, legend = dict(title='', x=1.018, y=0.5)
        )

    if 'auto' == xaxis:
        fig1.update_xaxes(
        autorange = True
        )
        fig2.update_xaxes(
        autorange = True
        )
    else:
        fig1.update_xaxes(
        range = xaxis
        )
        fig2.update_xaxes(
        range = xaxis
        )

    if yaxes[1] == 'auto':
        fig2.update_yaxes(autorange = True)
    else:
        fig2.update_yaxes(
        range = yaxes[1]
        )
    if yaxes[0] == 'auto':
        fig1.update_yaxes(autorange = True)
    else:
        fig1.update_yaxes(
        range = yaxes[0]
        )

    fig2.update_xaxes(
        side = 'top',
    )

    figshapes = get_stepNo_lines(df)
    
    fig1.update_layout(shapes=figshapes )
    fig2.update_layout(shapes=figshapes)

    return fig1, fig2, update_slider, str(legend_info)
    # , [json.dumps({
    #     'states': ctx.states,
    #     'triggered': ctx.triggered_prop_ids,
    #     'inputs': ctx.inputs,
    # }, indent=2)]

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
        if table == 'All':
            data_df = df.drop(['StepNo'], axis=1).describe().round(digits)
            data_df = pd.concat([pd.Series([i for i in data_df.index], index=data_df.index, name=f'StepNo: {table}'), data_df], axis=1)
            tables.append(dash_table.DataTable(
            columns=[{"name": i, "id": i} 
                        for i in data_df.columns],
            data=data_df.to_dict('records'),
            style_cell=dict(textAlign='left'),
            style_header={'backgroundColor':"#c7dabf"},
            style_data={'overflow': 'hidden', 'textOverflow': 'ellipsis', 'maxWidth': 0, 'height': 'auto'}
            ),
            )
            tables.append(html.Hr())
        else: 

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

@callback(
        Output('corr_table', 'children'),
        Input('store-one_file', 'data'),
        Input('upload', 'filename'),
)
def global_corr_coeff(json_dict, update_state):
    
    if update_state == None:
        raise dash.exceptions.PreventUpdate

    df = pd.DataFrame.from_dict(json_dict)
    dff = df.copy()

    for col in ['TMP1', 'TMP2', 'B31', 'B32','B21', 'B22']:
        result=seasonal_decompose(dff[col], model='additive', period=6)
        for idx, item in enumerate(result.trend):
            if np.isnan(item):
                result.trend[idx] = df[col][idx]
                result.resid[idx] = 10
        dff[col+' trend'] = result.trend
        dff[col+' resid'] = result.resid

    def drop_rows(dff):
        index = []
        for col in [ 'TMP1 resid', 'TMP2 resid', 'B31 resid', 'B32 resid', 'B21 resid', 'B22 resid']:
            for idx, val in enumerate(dff[col]):
                if abs(val) > 2:
                    index.append(idx)

        return dff.drop(index)

    drop_row_df = drop_rows(dff)

    table_cols = [['TMP1', 'TMP2', 'B31', 'B32', 'B21', 'B22'], ['TMP1 trend', 'TMP2 trend', 'B31 trend', 'B32 trend', 'B21 trend', 'B22 trend'], 
                    ['TMP1', 'TMP2', 'B31', 'B32', 'B21', 'B22']]
    dfs = [dff,dff,drop_row_df]
    titles = ['Correlation Coefficients for Raw Data', 'Correlation Coefficients Using Data Trend', 'Correlation Coefficients When Large Residuals Are Dropped']

    the_row = [dbc.Col([dcc.Graph(figure = corr_matrix(data,cols, title))]) for cols, data, title in zip(table_cols, dfs, titles)]
    return the_row