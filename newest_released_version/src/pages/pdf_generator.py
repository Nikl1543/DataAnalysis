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

def get_hist_and_box(data, col):
        df = data

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
        return fig
def get_fig(stored_data, cols1, cols2, query_data = False):
    dff = stored_data.copy()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    for trace in get_traces(dff, cols1, 'group1'):
        fig.add_trace(trace, row=1, col=1)
    for trace in get_traces(dff, cols2, 'group2'):
        fig.add_trace(trace, row=2, col=1)
    fig.update_layout(_AppSetup.fig_style)
    fig.update_layout(legend=dict(groupclick="toggleitem", y=0.5))

    if query_data==False:
        figshapes = get_stepNo_lines(dff)
        fig.update_layout(shapes=figshapes )
    return fig
def plotlyfig_to_png(fig, path, filename):
    fig.write_image(path + filename+'.png', format="png", width=1300, height=560, scale=5, engine= 'kaleido')
def compress_png(path, filename):
    img_path = path+filename+ '.png'
    img = Image.open(img_path)
    img = img.convert("P", colors=256)
    img.save(path + filename+ '.png', optimize=True)
def plot_to_file(fig, path, filename):
    plotlyfig_to_png(fig, path, filename)
    compress_png(path,filename)

def create_custom_info_table(df):
    StepNo_start= [0]+[i for i in range(1,len(df.StepNo)-1) if df.StepNo[i] != df.StepNo[i-1]]+ [df.index[-1]]
    StepNos = [val for val in df.StepNo[StepNo_start]]
    nunique = [end-start for start, end in zip(df.StepNo[StepNo_start[:-1]].index, df.StepNo[StepNo_start[1:]].index)]
    nunique[-1] += 1
    locs = []
    for n, StepNo, idx in zip(nunique, StepNos, range(1,len(StepNo_start))):
        if n >1:
            locs += [StepNo_start[idx-1],  StepNo_start[idx]-1]

    dff_filtered = df.loc[locs].reset_index()

    time_formats = ['%d-%m-%Y %H:%M:%S', '%d/%m/%Y %H.%M', '%d/%m/%Y %H.%M.%S']

    for tf in time_formats:
        for i in dff_filtered.index:
            try:
                datetime.strptime(dff_filtered['Log Time'][i], tf)
                time_format = tf
            except:
                pass

    idxs = [i for i in dff_filtered.index]
    p_diff = [(dff_filtered.B22[idxs[i+1]]-dff_filtered.B22[idxs[i]])*1000 for i in range(0, len(idxs)-1, 2)]
    B31_diff = [dff_filtered.B31[idxs[i+1]]-dff_filtered.B31[idxs[i]] for i in range(0, len(idxs)-1, 2)]
    B32_diff = [dff_filtered.B32[idxs[i+1]]-dff_filtered.B32[idxs[i]] for i in range(0, len(idxs)-1, 2)]

    convert_to_datetime_object = pd.to_datetime(list(dff_filtered['Log Time'].values)+[df['Log Time'][df.index[-1]]], format='%Y-%m-%dT%H:%M:%S')
    time_diffs = [(end-start).total_seconds() for start, end in zip(convert_to_datetime_object[:-1], convert_to_datetime_object[1:])]
    #Convert to nice time format
    time_converted = [list(divmod(divmod(time_diff,60)[0],60))+ [divmod(time_diff,60)[1]] for time_diff in time_diffs]
    time_diff_text = pd.Series([f'{int(time[0])}h{int(time[1])}m{int(time[2])}s' for time in time_converted], index = dff_filtered.index, name= 'Duration')

    dff_filtered['Log Time'] = dff_filtered['Log Time'].str.replace('-','').replace('.', ':').str.split('T')
    T_start = [dff_filtered['Log Time'][i][-1] for i in range(0, len(idxs)-1, 2)]
    T_end = [dff_filtered['Log Time'][i][-1] for i in range(1, len(idxs), 2)]
    P_start = [dff_filtered.B22[i] for i in range(0, len(idxs)-1, 2)]
    P_end = [dff_filtered.B22[i] for i in range(1, len(idxs), 2)]
    B31_start = [dff_filtered.B31[i] for i in range(0, len(idxs)-1, 2)]
    B31_end = [dff_filtered.B31[i] for i in range(1, len(idxs), 2)]
    B32_start = [dff_filtered.B32[i] for i in range(0, len(idxs)-1, 2)]
    B32_end = [dff_filtered.B32[i] for i in range(1, len(idxs), 2)]


    data_rows = [i for i in zip(T_start, T_end, time_diff_text, B31_start, B31_end, B31_diff, B32_start, B32_end, B32_diff, P_start, P_end, p_diff)]
    StepNos_filtered = [StepNo for StepNo, n in zip(StepNos, nunique) if n !=1]

    index = [(StepNo) for StepNo in StepNos_filtered]
    header = [' Time ']*3+ ['B31 [\u00B0C]']*3 + ['B32 [\u00B0C]']*3 + [' B22 [bar] ']*3
    column_names = [' START ', ' END ', ' DIFF ']*3 + ['START', 'END', 'DIFF[mbar]']
    columns = [i for i in zip(header, column_names)]

    data = {'index': index,
            'columns': columns,
            'data': data_rows,
            'index_names': ['StepNo'],
            'column_names': ['', '']}
    df_custom = pd.DataFrame.from_dict(data, orient='tight').round(2)
    return df_custom.reset_index()

class PDF(FPDF, HTMLMixin):
    def setup(self, font, margin, page_width = 190):
        self.normal_font = font
        self.footer_margin = margin
        self.set_auto_page_break(True, margin)
        self.page_width = page_width

    def header(self):
        self.y= self.y-6
        self.write_html('<img src="https://www.nolek.com/wp-content/uploads/2015/11/Nolek-logo.png" height="40">')
        # Arial bold 15
        self.set_font(self.normal_font, 'B', 8)
        # Title
        self.y= self.y-6
        self.cell(0, 10, self.title,align = 'R')
        self.set_y(20)
        bottom = self.y
        self.line(10, bottom, 200, bottom)
        self.ln(4)

    # Page footer
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-self.footer_margin)

        footer_dict = {
            0: ['Address:', 'Nolek A/S', 'Birkedam 10V', 'DK-6000 Kolding'],
            1: ['Telephone:', '(+45) 72 20 66 30'],
            2: ['Web:', 'www.nolek.dk'],
            3: ['e-mail:', 'info@nolek.dk']
        }
        top = self.y
        self.line(10, top, 200, top)
        keys = footer_dict.keys()
        for col, xoffset in zip(keys, np.linspace(10,170, len(keys)+1)):
            self.y=top
            self.x = xoffset
            self.set_font(self.normal_font, 'B', 8)
            self.cell(40,5,footer_dict[col][0],align = 'L')
            self.set_font(self.normal_font, '', 8)
            for item, yoffset in zip(footer_dict[col][1:], np.linspace(5,20, len(footer_dict[0])+1)):
                self.y = top + yoffset
                self.x = xoffset
                self.cell(40,5,item, align ='L')

        self.set_font(self.normal_font, 'B', 8)
        self.y =top
        self.cell(0, 4, 'Page ' + str(self.page_no()) + '/{nb}',align = 'R')
    def section_title(self, title, fontsize = 16):
        self.set_font(self.normal_font, 'B', fontsize)
        self.cell(40,10, title)
        self.ln(6)
    def text_box(self,text, h=5):
        self.set_font(self.normal_font, '', 12)
        self.write(h, text)
        self.ln(1)

    def output_df_to_pdf(self, df, title=None, table_cell_width=25, table_cell_height=6, multiIndex = False, relative_cell_width=False):

        numeric_cols = df.select_dtypes(include='float').columns
        df[numeric_cols] = df[numeric_cols].round(digits)
        # Select a font as Arial, bold, 8
        self.set_font(self.normal_font, 'B', 8)
        
        #determine column width of each column:
        colwidth = np.zeros(len(df.columns))
        if multiIndex == True:
            col_rows = {i: [] for i in range(len(df.columns[0]))}
            keys = [key for key in col_rows.keys()]

            for key in keys:
                if key == 0:
                    for name in df.columns:
                        name = name[key]
                        if name not in col_rows[key]:
                            col_rows[key].append(name)
                        else:
                            col_rows[key].append('')
                else:
                    col_rows[key] = [name[key] for name in df.columns]
            
            for value in col_rows.values():
                for idx, item in enumerate(value):
                    
                    string_width = self.get_string_width(item)
                    if string_width > colwidth[idx]:
                        colwidth[idx] = string_width
            for row in range(len(df)):
                for idx,item in enumerate(df.loc[row]):
                    
                    string_width = self.get_string_width(str(item))
                    if string_width > colwidth[idx]:
                        colwidth[idx] = string_width
            
        else:
            for idx, item in enumerate(df.columns):
                    string_width = self.get_string_width(item)
                    if string_width > colwidth[idx]:
                        colwidth[idx] = string_width
            
            for row in range(len(df)):
                for idx,item in enumerate(df.loc[row]):
                    string_width = self.get_string_width(str(item))
                    if string_width > colwidth[idx]:
                        colwidth[idx] = string_width
        if relative_cell_width == True:
            colwidth = [width/sum(colwidth)*self.page_width for width in colwidth]
            self.colwidth = colwidth
        else:
            colwidth = [table_cell_width for i in colwidth]

        # Loop over to print column names
        if multiIndex == True:
 
            #Determine size of table
            nr_rows, nr_cols = len(df)+len(keys), len(df.columns)
            total_width = table_cell_width*nr_cols
            total_height, page_pos = table_cell_height*(nr_rows), self.y

            if total_width > self.page_width:
                table_cell_width = int(self.page_width/nr_cols)
            if total_height+page_pos > self.h-30:
                self.add_page()
            counter = {}  
            count = 1 
            for idx, item in enumerate(col_rows[0]):
                if item != '':
                    name = item
                    count = 1
                else:
                    count+=1
                counter[name] = count
            self.counter = counter
            start = 0
            for key in counter.keys():
                spaces = counter[key]
                counter[key] = sum(colwidth[start:start+spaces])
                start += spaces

            if title != None: 
                self.cell(40,10, title)
                self.ln(5)
                self.set_font(self.normal_font, 'B', 8)
            for key in keys:
                offset = self.x
                y_pos  = self.y
                if key == 0:
                    for name, width in counter.items():
                        table_cell_width = width
                        self.x = offset
                        self.y = y_pos
                        self.cell(table_cell_width, table_cell_height, name, align='C', border=1)
                        offset += table_cell_width
                        self.ln(table_cell_height)

                else:

                    for idx, item in enumerate(col_rows[key]):
                        table_cell_width = colwidth[idx]
                        self.x = offset
                        self.y = y_pos
                        self.cell(table_cell_width, table_cell_height, item, align='R', border=1)
                        offset += table_cell_width
                        if key!= keys[-1]:
                            self.ln(table_cell_height)

        else:
            cols = df.columns
            #Determine size of table
            nr_rows, nr_cols = len(df)+1, len(cols)
            total_width = table_cell_width*nr_cols
            total_height, page_pos = table_cell_height*nr_rows, self.y

            if total_width > self.page_width:
                table_cell_width = int(self.page_width/nr_cols)
            if total_height+page_pos > self.h-60:
                self.add_page()

            if title != None: 
                self.section_title(title, fontsize=12)
                self.ln(3)
                self.set_font(self.normal_font, 'B', 8)

            for idx, col in enumerate(cols):
                table_cell_width = colwidth[idx]
                self.cell(table_cell_width, table_cell_height, col, align='C', border=1)
        
        # Line break
        self.ln(table_cell_height)
        # Select a font as Arial, regular, 10
        self.set_font(self.normal_font, '', 10)
        # Loop over to print each data in the table
        for row in range(len(df)):
            offset = self.x
            y_pos  = self.y
            for idx,item in enumerate(df.loc[row]):
                table_cell_width = colwidth[idx]
                self.x = offset
                self.y = y_pos
                self.cell(table_cell_width, table_cell_height, str(item), align='R', border=1)
                offset += table_cell_width
                if row != len(df)-1:
                     self.ln(table_cell_height)
        self.ln(5)

_AppSetup = AppConstants('0.2.0')
digits, color, fig_style, box_style = _AppSetup.digits, _AppSetup.color, _AppSetup.fig_style, _AppSetup.box_style

dash.register_page(__name__,
                   path='/report',
                   name='Report',
                   title='Report',
                   description='Construct Report.'
)

button_style =  {
        'height': '30px',
        'lineHeight': '30px',
        'border': '1px solid #999',
        'borderRadius': '5px',
        'textAlign': 'center',
        'margin': '10px',
        'padding': '5px'
    }

layout = html.Div(
    [   dbc.Row([
            dbc.Col([html.H2('Report', style={'text-align': 'left'})], width='auto', align='start'), 
            dbc.Col([dbc.Stack([
                html.Div(id="cls-output", className='output-loading'),
                dbc.Button("Download PDF Report", id = 'create_pdf', color='secondary', className="me-1")], direction='horizontal')],
                width = 'auto', align='end')],justify="between"),
        dbc.Row([
            dcc.Markdown("""
**PDF Report Generator**\n
Automatically includes:
- Unit information
- Table of information for each StepNo, such as pressure change, temperature change and time duration
- Data plot of entire data set
\n
Further includes(optional):
- Boxplot and histograms for chosen variables and StepNo's
- 'Zoom in' data plot for each chosen StepNo
- A comments section in the end of the report
\n
Note: All checkboxes can be left empty, if zoom in, histgram and boxplot should not be included.
        """)]),

        dcc.Markdown("**Choose the StepNo's to include histogram and boxplot for:**"),
        dcc.Markdown("(StepNo's consisting of less than 10 data points are filtered out)"),
        dcc.Checklist(options = [' ', '  '], value = [], inputStyle={'margin-right':'3px', 'margin-left':'5px'}, id='checklist-StepNos', inline=True, persistence=True, persistence_type='session'),
        dbc.Row([dbc.Stack([
        html.B('Choose the variables in first figure', style={'text-align':'left'}),
        dcc.Checklist(style={'width': 'auto', 'font-size':'0.85em', 'margin-bottom':'1em'}, id = 'Fig1checkboxesREPORT', persistence = True, inline=True, persistence_type='session', inputStyle={"margin-right":"5px","margin-left": "5px"}),
        html.B('Choose the variables in second figure', style={'text-align':'left'}),
        dcc.Checklist(style={'width': 'auto', 'font-size':'0.85em', 'margin-bottom':'2em'}, id = 'Fig2checkboxesREPORT', persistence = True, inline=True, persistence_type='session', inputStyle={"margin-right":"5px","margin-left": "5px"})
        ])
        ]),
        html.Br(),
        dcc.Markdown('**Add comment section to the report** (The comments section is not included if the textarea below is left empty)'),
        dcc.Textarea(id = 'textarea',style={'width': '100%', 'min-height': '5cm'}),
        html.Div(id='textarea-output', style={'whiteSpace': 'pre-line'}),
        dcc.Download(id="download-pdf"),

            ])

@callback(
    Output('checklist-StepNos', 'options'),
    Output('Fig1checkboxesREPORT', 'options'),
    Output('Fig2checkboxesREPORT', 'options'),
    Output('Fig1checkboxesREPORT', 'value'),
    Output('Fig2checkboxesREPORT', 'value'),
    Input('store-one-file', 'data')
)
def update_checklists(json_dict, fig1_value = ['B21', 'B22', 'P101', 'RegulatorSP'], fig2_value = ['TMP1', 'TMP2', 'B31', 'B32']):
    df = pd.DataFrame.from_dict(json_dict)
    StepNo_info = df.StepNo.value_counts()
    dff = df[df.StepNo.isin([key for key in StepNo_info.keys() if StepNo_info[key]>10])]

    StepNo_options = [option for option in dff.StepNo.unique()] 

    col_options     = [i for i in dff.columns]
    variable_options = [col for col in col_options if col not in ['Log Time', 'CircuitName','StepNo']]

    return StepNo_options, variable_options, variable_options, fig1_value, fig2_value

@callback(
    Output('download-pdf', 'data'),
    Output('cls-output', 'children'),
    Input( 'create_pdf', 'n_clicks'),
    State('store-one-file', 'data'),
    State('active-file-holder', 'data'),
    State('store-raw-data', 'data'),
    State('checklist-StepNos', 'value'),
    State('Fig1checkboxesREPORT', 'value'),
    State('Fig2checkboxesREPORT', 'value'),
    State('textarea', 'value'),
    prevent_initial_call = True
)
def create_pdf_report(n, json_dict, active_file, FileStore, StepNo_vals, fig1cols, fig2cols, comment_text):  
    if len(FileStore.keys()) == 0:
        return dash.no_update, ''

    content_string = FileStore[active_file]
    decoded = base64.b64decode(content_string)
    file_info = pd.read_csv(io.StringIO(decoded.decode('utf-8')),
                sep='[;]', 
                engine='python',
                decimal=',',
                nrows=0
                )

    file_info = [name for name in file_info.columns if 'Unnamed' not in name]
    file_info[0] = file_info[0].split('\\')[-1]
    filename = file_info[0].replace('.csv', '')

    df = pd.DataFrame.from_dict(json_dict)
    dff = df.copy()

    order = {idx: item for idx, item in enumerate(dff.StepNo.unique())}
    StepNo_vals = [order[key] for key in sorted(order) if order[key] in StepNo_vals]
    # Instantiation of inherited class
    pdf = PDF()
    pdf.setup(font='Times', margin = 30)
    pdf.set_title(filename)
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.section_title('Unit Information', fontsize=14)
    pdf.ln(3)
    offsets = [int(pdf.get_string_width(info))+5 for info in file_info]
    offset = 10
    pdf.set_font(pdf.normal_font, '', 12)
    for info, xoffset in zip(file_info, offsets):
        pdf.x = offset
        offset += xoffset
        pdf.cell(xoffset,6, info, align='L')
    pdf.ln(10)
    pdf.section_title('Statistical Summary')

    pdf.text_box( """
In this section the extracted statistics are included. This includes histogram and boxplot, together with information such as pressure drop over some StepNo's.
                        """)
    pdf.text_box("Pressure change, temperatrue change and start and stop datetime is shown for each Stepno consisting of more than one sample and not appearing more once.")
    pdf.ln(5)
    df_custom = create_custom_info_table(dff)
    pdf.output_df_to_pdf(df_custom, multiIndex=True, relative_cell_width=True)
    pdf.ln(5)
    pdf.add_page()
    pdf.text_box("""
In Data Exploration one can visualise the distribution of a chosen variable in a chosen StepNo.\n
This is done through histograms(left) and boxplot(right)
    """)
    with tempfile.TemporaryDirectory() as workdir:
        #add files to folder:
        cols = [col for col in dff.columns if col not in ['Log Time', 'StepNo', 'CircuitName']]
        if StepNo_vals != []:
            count = 0
            for val in StepNo_vals:
                for col in cols:
                    stat_fig =get_hist_and_box(dff.query('StepNo == '+str(val)), col)
                    if count%2 == 0 and count != 0:
                        pdf.add_page()
                    pdf.section_title(f'Following plot is StepNo {val}, column {col}', fontsize=13)
                    pdf.ln(3)
                    plot_to_file(stat_fig, workdir + '/', f'plot_of_statfig{str(val)}{str(col)}')
                    pdf.image(workdir+f'/plot_of_statfig{str(val)}{str(col)}.png', w=pdf.page_width)
                    count+=1
    pdf.ln(5)

    pdf.add_page()
    pdf.section_title( 'Data Plots')
    pdf.text_box( """ 
In this section plots of the data are included
                        """)
    with tempfile.TemporaryDirectory() as workdir:
        #add files to folder:

        data_fig =get_fig(dff, fig1cols, fig2cols)
        pdf.section_title('Following plot is all data', fontsize=13)
        pdf.ln(3)

        plot_to_file(data_fig, workdir + '/', 'plot_of_fig')

        pdf.image(workdir+'/plot_of_fig.png', w=pdf.page_width)

        count = 0
        if StepNo_vals != []:
            for val in StepNo_vals:
                data_fig =get_fig(dff.query('StepNo == '+str(val)), fig1cols, fig2cols, query_data=True)
                
                if count%2==0 and count != 0:
                    pdf.add_page()
                pdf.section_title(f'Following plot is StepNo {val}', fontsize=13)
                pdf.ln(3)
                plot_to_file(data_fig, workdir + '/', f'plot_of_fig'+str(val))

                pdf.image(workdir+f'/plot_of_fig{str(val)}.png', w=pdf.page_width)
                count+=1
    pdf.ln(5)
    if comment_text != None:
        if comment_text!='':
            pdf.add_page()
            pdf.section_title( 'Comments')
            pdf.ln(2)
            pdf.text_box(f"""{comment_text}""")

    with tempfile.TemporaryDirectory() as workdir:
        pdf.output(workdir+ '/' + filename+'.pdf', 'F')

        return dcc.send_file(workdir+ '/'+filename+ '.pdf'), ''