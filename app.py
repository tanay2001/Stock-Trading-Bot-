import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import datetime
import numpy as np
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
df = pd.read_csv('/home/tanay/Documents/Stonks/stonks_data.csv')
available_indicators = df['index'].unique()
tech= df.columns[7:]
df['year']= df['Date'].apply(lambda x : x[:4])
df['year'] = df['year'].astype(np.int64)


colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}


app.layout = html.Div([
        html.H1(
        children='Stonks',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    html.Div([

        html.Div([
            dcc.Dropdown(
                id='xaxis-column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='M&M'
            )],style={'width': '48%', 'display': 'inline-block'}),
            
        html.Div([
            dcc.Dropdown(
                id='indicator',
                options=[{'label': i, 'value': i} for i in tech],
                value='ma21'
            )
        ],style={'width': '48%', 'float': 'right', 'display': 'inline-block'}) 
            ]),

    dcc.Graph(id='indicator-graphic'),

    dcc.Slider(
        id='year--slider',
        min=df['year'].min(),
        max=df['year'].max(),
        value=df['year'].min(),
        marks={str(date) : str(date) for date in df['year'].unique()},
        step=None
    )
])

@app.callback(
    Output('indicator-graphic', 'figure'),
    [Input('year--slider', 'value'),
    Input('xaxis-column', 'value'),
     Input('indicator', 'value')
     ])

def update_graph(yr, ticker,ta):
    data =df[df['index']==ticker][df['year']>=yr]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x= data['Date'], y=data['Close'], name ='closing price'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data[ta], name ='technical indicator {}'.format(ta)))
    fig.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text']
)
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)