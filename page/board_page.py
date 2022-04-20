import dash
from dash import Dash, dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px


fig = px.scatter(y=[10, 10, 10], x=[100, 200, 300], size=[40, 50, 10])
selector = dcc.Slider(
    id='first_figure_selector',
    min=0,
    max=100,
    marks={0: '0', 100: '100'},
    step=1,
    value=0
)


@callback(Output('first_figure', component_property='figure'),
          Input('first_figure_selector', component_property='value'),
          suppress_callback_exceptions=True)
def figure(size1):
    size = [40*size1, 50*size1, 10+size1]
    _fig = px.scatter(y=[10, 10, 10], x=[100, 200, 300], size=size)
    return _fig


ID = 'board_page'
layout = html.Div(id=ID, children=[
    html.Div(className='row', children=[
        html.Div(className='col', children=[
            html.Div('PREDICTIVE ANALYSIS', className='title'),
            html.Div('Thanks to modern algorithms, our team can predict various time series, for example, '
                     'sales of your products.', className='first_text'),
            html.Div('For a qualitative forecast, we study the influence of various factors such as: competitors, '
                     'various marketing activities and others.', className='second_text')
        ]),
        html.Div(className='col', children=[
            html.Div(dcc.Graph(id='first_figure', figure=fig)),
            html.Div(selector)
        ])
    ]),
])



