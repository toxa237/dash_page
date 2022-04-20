import dash
from dash import Dash, dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px

df = pd.DataFrame(data={'val': [10, 20, 30, 40],
                        'names': ['1', '2', '3', '4']})

fig = px.pie(df, values='val', names='names')

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}
fig.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text']
)

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
    df.loc[3, 'val'] = 40 + size1
    _fig = px.pie(df, values='val', names='names')
    _fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text']
    )
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



