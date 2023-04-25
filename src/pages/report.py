# from jupyter_dash import JupyterDash
import dash
from dash import dcc, Dash, html, Input, Output, dash_table, State, ctx, callback
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
# from plotly.subplots import make_subplots
import base64
from datetime import datetime
import io
import json
from fpdf import FPDF, HTMLMixin
from statsmodels.tsa.seasonal import seasonal_decompose
import tempfile
import os
from PIL import Image
import urllib.request
import ssl

def main():
    ssl._create_default_https_context = ssl._create_unverified_context
    r = urllib.request.urlopen('https://google.com')
main()

dash.register_page(__name__,
                   path='/report',
                   name='Report',
                   title='Report',
                   description='Construct Report.'
)
box_style = {'width': '99%',
        'height': '235mm',
        'lineHeight': '60px',
        'border': '1px solid #999',
        'textAlign': 'center',
        'margin': '10px',
        'padding': '5px'}
color = {'titlecolor': '#60893c', 'plot_background': '#efefef', 'gridcolor': 'c4c4c4', 'plotcolor2': 'dadada'}
digits = 3
button_style =  {
        'height': '30px',
        'lineHeight': '30px',
        'border': '1px solid #999',
        'borderRadius': '5px',
        'textAlign': 'center',
        'margin': '10px',
        'padding': '5px'
    }

upload = dcc.Upload(
    children=html.Div([
        'Drag and Drop or ',
            html.A('Select File',href="#", style={'color': '#1d8fe5'})
    ]),
    style={
        'width': '99%',
        'height': '60px',
        'lineHeight': '60px',
        'border': '1px dashed #999',
        'borderRadius': '5px',
        'textAlign': 'center',
        'margin': '10px',
        'padding': '5px'
    },
    # Allow multiple files to be uploaded
    multiple=True,
    id ='upload_snapshots'
    
)
fig_style = dict(height = 560,
        title = None,
        plot_bgcolor = '#efefef', 
        legend = dict(title='', x=1.02, y=0.5),
        yaxis = dict(gridcolor = '#c4c4c4', title = None),
        xaxis = dict(gridcolor = '#c4c4c4', title = None),
        hovermode="x unified",
        margin = dict(b=10, t=5),
    )
box_style = {'margin': '10px', 'padding': '5px','border': '1px solid #999'}

def corr_matrix(df, columns, title):
       df_corr = df[columns].corr().round(2)

       fig = px.imshow(df_corr, text_auto=True, title = title,color_continuous_scale=[[0, '#006600'], [0.35, '#FFFFFF'], [0.5, 'white'], [0.65, '#FFFFFF'], [1.0, '#006600']], zmin =-1, zmax=1, aspect="auto")
       fig.update_traces(hovertemplate='%{x} - %{y}<br>Coefficient %{z}<extra></extra>')
       fig.update_xaxes(side='top')

       return fig

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

def get_traces(df, cols):
    hovertemplates = {col: '<br>Log Time=%{x}<br>value=%{y}<br>StepNo=%{customdata}<br>' for col in cols}
    custom_data = np.array([[i] for i in df.StepNo]).astype(np.int64)
    traces = [go.Scatter(x=df['Log Time'], y=df[col], name = col, customdata=custom_data,  hovertemplate = hovertemplates[col]) for col in cols]
    return traces

def get_trend(df):
    dff = df.copy()
    for col in ['TMP1', 'TMP2', 'B31', 'B32','B21', 'B22']:
        result=seasonal_decompose(dff[col], model='additive', period=6)
        for idx, item in enumerate(result.trend):
            if np.isnan(item):
                result.trend[idx] = df[col][idx]
        dff[col+' trend'] = result.trend
    return dff

def get_figs(stored_data, query_data = False):
    dff = stored_data.copy()
    dff['Log Time'] = [i.split(' ')[1] for i in dff['Log Time']]

    fig2 = go.Figure()
    for trace in get_traces(dff, ['B21', 'B22', 'RegulatorSP', 'RegulatorFB']):
            fig2.add_trace(trace)

    fig1 = go.Figure()
    for trace in get_traces(dff, ['TMP1', 'TMP2', 'B31', 'B32']):
        fig1.add_trace(trace)
    fig1.update_layout(fig_style, legend = dict(title='', x=1.04, y=0.5))

    fig2.update_layout(fig_style, legend = dict(title='', x=1.0, y=0.5)
    )

    fig2.update_xaxes(
        side = 'top',
    )
    if query_data==False:
        figshapes = get_stepNo_lines(dff)

        fig1.update_layout(shapes=figshapes )
        fig2.update_layout(shapes=figshapes)
    return fig1, fig2

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
    time_diffs = [compute_time_diff(dff_filtered['Log Time'][idxs[i]], dff_filtered['Log Time'][idxs[i+1]], time_format) for i in range(0, len(idxs)-1, 2)]
    time_converted = [list(divmod(divmod(time_diff,60)[0],60))+ [divmod(time_diff,60)[1]] for time_diff in time_diffs]
    time_diff_text = [f'{int(time[0])} h {int(time[1])} min {int(time[2])} s' for time in time_converted]

    T_start = [dff_filtered['Log Time'][i].replace('.', ':').split(' ')[-1] for i in range(0, len(idxs)-1, 2)]
    T_end = [dff_filtered['Log Time'][i].replace('.', ':').split(' ')[-1]  for i in range(1, len(idxs), 2)]
    P_start = [dff_filtered.B22[i] for i in range(0, len(idxs)-1, 2)]
    P_end = [dff_filtered.B22[i] for i in range(1, len(idxs), 2)]
    B31_start = [dff_filtered.B31[i] for i in range(0, len(idxs)-1, 2)]
    B31_end = [dff_filtered.B31[i] for i in range(1, len(idxs), 2)]
    B32_start = [dff_filtered.B32[i] for i in range(0, len(idxs)-1, 2)]
    B32_end = [dff_filtered.B32[i] for i in range(1, len(idxs), 2)]


    data_rows = [i for i in zip(T_start, T_end, time_diff_text, B31_start, B31_end, B31_diff, B32_start, B32_end, B32_diff, P_start, P_end, p_diff)]
    StepNos_filtered = [StepNo for StepNo, n in zip(StepNos, nunique) if n !=1]

    index = [(StepNo) for StepNo in StepNos_filtered]
    header = [' Time ']*3+ ['B31 [\u00B0C]']*3 + ['B32 [\u00B0C]']*3 + [' B22 ']*3
    column_names = [' START ', ' END ', ' DIFF ']*3 + ['START [bar]', 'END [bar]', 'DIFF [mbar]']
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
    def get_statistics_table(self, df, title):
        data_df = df.describe().round(digits)
        data_df = pd.concat([pd.Series([i for i in data_df.index], index=data_df.index, name=f' '), data_df], axis=1).reset_index()
        data_df = data_df[data_df.columns[1:]]
        self.output_df_to_pdf(data_df, title=title, relative_cell_width=True)
        self.ln(5)

                
def compute_time_diff(t1, t2, tf):
    time_diff = (datetime.strptime(t2, tf)- datetime.strptime(t1, tf)).total_seconds()
        
    return time_diff


def correlation_table_to_pdf(df, columns):
    cols = []
    corrs = {}
    for col1 in columns:
        for col2 in columns:
            if col2 != col1 and col2 not in cols:
                corrs[col1+'|' +col2] = np.round(df[col1].corr(df[col2]),digits)
        cols.append(col1)
    table_corrs = pd.DataFrame(corrs.items(), columns = ['Variables', 'correlation'])
    return table_corrs

layout = html.Div(
    [   dbc.Row([dbc.Col([html.H2('Report', style={'text-align': 'left'})], width='auto', align='start'), dbc.Col([dbc.Stack([html.Div(id="cls-output", className='output-loading'),dbc.Button("Download PDF Report", id = 'create_pdf', color='secondary', className="me-1")], direction='horizontal')], width = 'auto', align='end')],justify="between",),
        dbc.Row([dcc.Markdown("""
**PDF Report Generator**\n
Automatically includes:
- Unit information
- Table of information for each StepNo, such as pressure change, temperature change and time duration
- Global extract of summary statistics
- Correlation coefficient table using all data
- Data plot of entire data set
\n
Further includes(optional):
- An extract of summary statistics for chosen StepNo's
- Correlation table for chosen StepNo's
- 'Zoom in' data plot for each chosen StepNo
- A comments section in the end of the report
\n
Note: All checkboxes can be left empty, if preferred.
        """)]),
        # dcc.Markdown('**Upload the snapshots to include in the report** (must be .jpeg or .png)**:**'),
        # upload,
        # html.Div(id='upload-state', children = ' '),
        # dcc.Store('snapshots'),
        dcc.Markdown("**Choose the StepNo's to include statistics for:**"),
        dcc.Checklist(options = [' ', '  '], value = [], inputStyle={'margin-right':'3px', 'margin-left':'5px'}, id='checklist-StepNos', persistence=True, persistence_type='local'),
        # dcc.Markdown("**Add comments for the snapshot:**"),
        # html.Div(id='tabs-for-snapshots'),
        html.Br(),
        dcc.Markdown('**Add comment section to the report** (The comments section is not included if the textarea below is left empty)'),
        dcc.Textarea(id = 'textarea',style={'width': '100%', 'min-height': '5cm'}),
        html.Div(id='textarea-output', style={'whiteSpace': 'pre-line'}),
        dcc.Download(id="download-pdf"),

        # dbc.Row([
        #         html.ObjectEl(data="assets/reportbuild2.pdf", type="application/pdf", style={"width": "250mm", "height": "270mm"})]),
            ])
# @callback(
#     Output('snapshots', 'data'),
#     Output('upload-state', 'children'),
#     Input('upload_snapshots', 'contents'),
#     State('upload_snapshots', 'filename'),
#     State('upload_snapshots', 'last_modified'),
#     prevent_initial_call = True
# )
# def store_snapshots(contents, list_of_names, list_of_dates):
#     if [contents, list_of_names, list_of_dates] == [None, None, None]:
#         return dash.no_update
    
#     content_type_list = [content.split(',')[0] for content in contents]
#     content_string_list = [content.split(',')[1] for content in contents]
#     print(content_type_list)
    
#     not_filetype = []
#     for idx, content_type in enumerate(content_type_list):
#         if 'png' not in content_type:
#             if 'jpeg' not in content_type:
#                 not_filetype.append(idx)

#     upload_state = ' '
#     if not_filetype != []:
#         text = ''
#         for item in not_filetype:
#             text += f'{item +1}, '
#         upload_state = 'uploaded file number ' + text +'is not of type .jpeg or .png'

#     content_strings_saved = [content_string for idx, content_string in enumerate(content_string_list) if idx not in not_filetype]
#     list_of_names_saved = [name for idx, name in enumerate(list_of_names) if idx not in not_filetype]


#     return content_strings_saved, upload_state


# @callback(
#     Output('tabs-for-snapshots', 'children'),
#     Input('snapshots', 'data'),
#     prevent_initial_call = True
# )
# def create_snapshot_tabs(snapshots):
#     snapshots = snapshots[1]
#     if snapshots == None:
#         return 'No snapshots uploaded yet'

#     # for content in snapshots:
#     #     decoded = base64.b64decode(content)
#     #     im = Image.open(io.BytesIO(decoded))
#     #     im.save('image.png', 'PNG')
#     card = [dbc.Card(
#         [
#             dbc.CardHeader(
#                 dbc.Tabs(
#                     [
#                         dbc.Tab(label=name, tab_id = f'tab-{i+1}') for i, name in enumerate(snapshots)
#                         # dbc.Tab(label="Tab 1", tab_id="tab-1"),
#                         # dbc.Tab(label="Tab 2", tab_id="tab-2"),
#                     ],
#                     id="card-tabs",
#                     active_tab="tab-1",
#                 )
#             ),
#             dbc.CardBody(html.P(id="card-content", className="card-text")),

#         ]
#     )]
#     placeholder = [dcc.Store('comment-content')]
#     component = card + placeholder
#     return component

# @callback(
#     Output("card-content", "children"), [Input("card-tabs", "active_tab")]
# )
# def get_textarea(active_tab):
#     nr = active_tab.split('-')[-1]
#     return [dcc.Markdown('**Write comment for this snapshot**'), html.Div([
#     dcc.Textarea(
#         id=f'textarea-{nr}',
#         value='',
#         style={'width': '100%', 'height': 200},
#     ),
#     html.Button('Submit', id=f'textarea-{nr}-button', n_clicks=0),
#     html.Div(id='textarea-state-example-output', style={'whiteSpace': 'pre-line'})
# ])]

@callback(
    Output('checklist-StepNos', 'options'),
    Input('store-one_file', 'data')
)
def create_checklist(json_dict):
    df = pd.DataFrame.from_dict(json_dict)
    StepNo_info = df.groupby(by = 'StepNo').size()
    dff = df[df.StepNo.isin([key for key in StepNo_info.keys() if StepNo_info[key]>1])]

    StepNo_options = [option for option in dff.StepNo.unique()] 

    return StepNo_options

# @callback(
#     Output('textarea-output', 'children'),
#     Input('textarea', 'value')
# )
# def get_text(text):
#     return(dcc.Markdown(text))

@callback(
    Output('download-pdf', 'data'),
    Output('cls-output', 'children'),
    Input( 'create_pdf', 'n_clicks'),
    State('store-one_file', 'data'),
    State('log_of_btns', 'data'),
    State('file-store', 'data'),
    State('checklist-StepNos', 'value'),
    State('textarea', 'value'),
    prevent_initial_call = True
)
def create_pdf_report(n, json_dict, BtnsLog, FileStore, StepNo_vals, comment_text):  
    if len(FileStore.keys()) == 0:
        return dash.no_update, ''

    active_file = BtnsLog['active_file']
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
In this section the extracted statistics are included. This includes a table for the global statistics, together with information such as pressure drop over some StepNo's.
                        """)
    pdf.text_box("Pressure change, temperatrue change and start and stop datetime is shown for each Stepno consisting of more than one sample and not appearing more once.")
    pdf.ln(5)
    df_custom = create_custom_info_table(dff)
    pdf.output_df_to_pdf(df_custom, multiIndex=True, relative_cell_width=True)
    pdf.ln(5)
    pdf.add_page()
    pdf.text_box("""
The statistics computed:
    - number of samples
    - mean value
    - standard deviation 
    - minimum value
    - the 25% median
    - the 50% median
    - the 75% median
    - maximum value
    """)
    pdf.get_statistics_table(dff.drop(['StepNo'], axis=1), 'Statistics for the full data set')
    if StepNo_vals != []:
        for val in StepNo_vals:
            dff_filtered = dff.query(f'StepNo=={val}')
            pdf.get_statistics_table(dff_filtered.drop(['StepNo'], axis=1), f'Statistics for StepNo {val}')

    pdf.add_page()
    pdf.section_title('Correlation coefficients', fontsize=14)
    pdf.ln(3)
    pdf.text_box('The correlation coefficient computed is the Pearson coefficient')
    pdf.ln(5)
    pdf.ln(5)

    corr_fig_global = corr_matrix(dff, ['TMP1', 'TMP2', 'B31', 'B32', 'B21', 'B22'], title='Global correlation coefficients')

    corr_fig2 = corr_matrix(dff.query('StepNo == 44000'), ['TMP1', 'TMP2', 'B31', 'B32', 'B21', 'B22'], title = 'Correlation coefficients for StepNo 44000')
    
    with tempfile.TemporaryDirectory() as workdir:

        corr_fig = corr_matrix(dff, ['TMP1', 'TMP2', 'B31', 'B32', 'B21', 'B22'], title='Global correlation coefficients')
        if StepNo_vals != []:
            corr_figs = {f'corr_fig{i}': corr_matrix(dff.query('StepNo == '+str(val)), ['TMP1', 'TMP2', 'B31', 'B32', 'B21', 'B22'], 
                        title = f'Correlation coefficients for StepNo {val}') for i, val in enumerate(StepNo_vals)}
            corr_figs = {**{'corr_fig_global':corr_fig}, **corr_figs}

            for key, val in corr_figs.items():
                plot_to_file(val, workdir + '/', key)
                if key == 'corr_fig_global':
                    pdf.section_title('Following table is the global correlation coefficients', fontsize=13)
                    pdf.ln(5)
                    pdf.image(workdir+f'/{key}.png', w=pdf.page_width)
                    pdf.ln(5)
                    pdf.section_title('Following tables are correlation coefficients only computed for a specific StepNo', fontsize=13)
                    pdf.ln(5)
                else:
                    pdf.image(workdir+f'/{key}.png', w=pdf.page_width)
                    pdf.ln(5)

        else:
            corr_fig = corr_matrix(dff, ['TMP1', 'TMP2', 'B31', 'B32', 'B21', 'B22'], title='Global correlation coefficients')
            plot_to_file(corr_fig, workdir + '/', 'plot_of_corrfig')
            pdf.text_box('Following table is the global correlation coefficients')
            pdf.ln(5)
            pdf.image(workdir+'/plot_of_corrfig.png', w=pdf.page_width)


    pdf.add_page()
    pdf.section_title( 'Data Plots')
    pdf.text_box( """ 
In this section plots of the data are included
                        """)
    pdf.ln(5)
    
    with tempfile.TemporaryDirectory() as workdir:
        #add files to folder:
        data_fig1, data_fig2 =get_figs(dff)
        pdf.section_title('Following plot is all data', fontsize=13)
        pdf.ln(3)

        plot_to_file(data_fig2, workdir + '/', 'plot_of_fig2')
        plot_to_file(data_fig1, workdir + '/', 'plot_of_fig1')

        pdf.image(workdir+'/plot_of_fig2.png', w=pdf.page_width)
        pdf.image(workdir+'/plot_of_fig1.png', w=pdf.page_width)


        if StepNo_vals != []:
            for val in StepNo_vals:
                data_fig_1, data_fig_2 =get_figs(dff.query('StepNo == '+str(val)), query_data=True)
                pdf.add_page()
                pdf.section_title(f'Following plot is StepNo {val}', fontsize=13)
                pdf.ln(3)
                plot_to_file(data_fig_2, workdir + '/', f'plot_of_fig2'+str(val))
                plot_to_file(data_fig_1, workdir + '/', f'plot_of_fig1'+str(val))

                pdf.image(workdir+f'/plot_of_fig2{str(val)}.png', w=pdf.page_width)
                pdf.image(workdir+f'/plot_of_fig1{str(val)}.png', w=pdf.page_width)

    if comment_text != None:
        if comment_text!='':
            pdf.add_page()
            pdf.section_title( 'Comments')
            pdf.ln(2)
            pdf.text_box(f"""{comment_text}""")

    with tempfile.TemporaryDirectory() as workdir:
        pdf.output(workdir+ '/' + filename+'.pdf', 'F')

        return dcc.send_file(workdir+ '/'+filename+ '.pdf'), ''