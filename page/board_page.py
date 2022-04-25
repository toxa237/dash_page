import dash
import numpy as np
from dash import Dash, dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from page.element.model_for_ts import ONLY_PREDICTION_WITH_TRP

df1 = pd.read_csv('assets/board_pade/figure1.csv')
df1.index = df1['area']


@callback(Output('first_figure', component_property='figure'),
          Input('first_figure_selector', component_property='value'),
          suppress_callback_exceptions=True)
def figure1(size1):
    global df1
    df = df1.copy()
    _fig = px.scatter(df.loc[size1], x='positive', y='negative', size='count', size_max=50,
                      color='color', hover_name='area',
                      hover_data={i: False for i in df.columns})
    _fig.update_layout(
        plot_bgcolor='#151515',
        paper_bgcolor='black',
        font_color='white',
        xaxis={'showgrid': False,
               'zeroline': False},
        yaxis={'showgrid': False,
               'zeroline': False},
        showlegend=False,
    )
    return _fig


df2 = pd.read_csv('assets/board_pade/figure2.csv')


@callback(Output('second_figure', component_property='figure'),
          Input('second_figure_selector', component_property='value'),
          suppress_callback_exceptions=True)
def figure2(size1):
    global df2
    df: pd.DataFrame = df2[df2['area'].isin(size1)].copy()
    df = df.groupby(by=['network']).sum()
    df['size'] = df.sum(axis=1)
    for i in df.columns[:-1]:
        df[i] = (df[i] / df['size']) * 100
    df.reset_index(inplace=True)
    _fig = px.scatter(df, x='positive', y='negative', size='size', size_max=50, hover_name='network',
                      hover_data={i: False for i in df.columns}, color='network')
    _fig.update_layout(
        plot_bgcolor='#151515',
        paper_bgcolor='black',
        font_color='white',
        xaxis={'showgrid': False,
               'zeroline': False},
        yaxis={'showgrid': False,
               'zeroline': False},
        showlegend=False,
    )
    return _fig


@callback(Output('third_figure', component_property='figure'),
          Input('third_figure_selector', component_property='value'),
          suppress_callback_exceptions=True)
def figure3(size1):
    global df1
    df = df1.copy()
    _fig = px.scatter(x=[0, 1], y=[0, 1])
    _fig.update_layout(
        plot_bgcolor='#151515',
        paper_bgcolor='black',
        font_color='white',
        xaxis={'showgrid': False,
               'zeroline': False},
        yaxis={'showgrid': False,
               'zeroline': False},
        showlegend=False,
    )
    return _fig


model = ONLY_PREDICTION_WITH_TRP('assets/board_pade/model_for_figure4.pkl')


@callback(Output('fourth_figure', component_property='figure'),
          Input('fourth_figure_selector1', component_property='value'),
          Input('fourth_figure_selector2', component_property='value'),
          Input('fourth_figure_selector3', component_property='value'),
          Input('fourth_figure_selector4', component_property='value'),
          suppress_callback_exceptions=True)
def figure4(trp0, trp1, trp2, trp3):
    global model
    df = pd.DataFrame({'pcs': model.pcs,
                       'x': list(range(model.pcs.shape[0])),
                       'color': 'real_data'})
    df = df.append(pd.DataFrame({'pcs': model.prediction(np.zeros(52)),
                                 'x': list(range(model.pcs.shape[0] + 1, model.pcs.shape[0] + 53)),
                                 'color': 'base_predict'}), ignore_index=True)
    trp_val = np.array([])
    trp = [trp0, trp1, trp2, trp3]
    for i in range(4):
        trp_val = np.append(trp_val, [trp[i] / 13] * 13)
    df = df.append(pd.DataFrame({'pcs': model.prediction(trp_val),
                                 'x': list(range(model.pcs.shape[0] + 1, model.pcs.shape[0] + 53)),
                                 'color': 'predict_with_trp'}), ignore_index=True)
    df.to_excel('check.xlsx')
    _fig = px.line(df, x='x', y='pcs', color='color')
    _fig.update_layout(
        plot_bgcolor='#151515',
        paper_bgcolor='black',
        font_color='white',
        xaxis={'showgrid': False,
               'zeroline': False},
        yaxis={'showgrid': False,
               'zeroline': False}
    )
    return _fig


ID = 'board_page'
layout = html.Div(id=ID, children=[
    html.Div(className='container-fluid page-content-element', children=[
        html.Div(className='row', children=[
            html.Div(className='col', children=[
                html.Div('PREDICTIVE ANALYSIS', className='title'),
                html.Div('Thanks to modern algorithms, our team can predict various time series, for example, '
                         'sales of your products.', className='first_text'),
                html.Div('For a qualitative forecast, we study the influence of various factors such as: competitors, '
                         'various marketing activities and others.', className='second_text')
            ]),
            html.Div(className='col', children=[
                html.Img(src=f'assets/{ID}/banner.png', className='img_to_basic_stile_div')
            ])
        ])
    ]),

    html.Div(className='row', children=[
        html.Div(className='col', children=[
            html.Div(className='shadow', children=[
                html.Div(className='shape-outer figure', children=[
                    html.Div(className='shape-inner figure', children=[
                        html.Img(src=f'assets/{ID}/icon1.png', className='img_prod')
                    ])
                ])
            ]),
            html.Div(className='text_prod', children='Трекинг отношения клиентов к компании в реальном времени')
        ]),

        html.Div(className='col', children=[
            html.Div(className='shadow', children=[
                html.Div(className='shape-outer figure', children=[
                    html.Div(className='shape-inner figure', children=[
                        html.Img(src=f'assets/{ID}/icon2.png', className='img_prod')
                    ])
                ])
            ]),
            html.Div(className='text_prod', children='Определение слабых мест продукта')
        ]),

        html.Div(className='col', children=[
            html.Div(className='shadow', children=[
                html.Div(className='shape-outer figure', children=[
                    html.Div(className='shape-inner figure', children=[
                        html.Img(src=f'assets/{ID}/icon3.png', className='img_prod')
                    ])
                ])
            ]),
            html.Div(className='text_prod', children='Возможность быстрой реакции')
        ]),
    ]),

    #  -------------------------------  first figure  --------------------------------

    html.Div(className='container', children=[
        html.Div(className='row', children=[
            html.Div(className='col', children=[
                'Исследования репутации бренда проводят в разных отрослях. '
                'Больше всего негатива в отрасли табачных изделий, а позитива – в службе такси. '
                '……….. '
            ]),
            html.Div(className='col', children=[
                html.Div(dcc.Graph(id='first_figure')),
                html.Div(dcc.Dropdown(df1.index,
                                      df1.index,
                                      multi=True, id='first_figure_selector'))
            ]),
        ])
    ]),

    #  -------------------------------  second figure  --------------------------------

    html.Div(className='container', children=[
        html.Div(className='row', children=[
            html.Div(className='col', children=[
                html.Div(dcc.Graph(id='second_figure')),
                html.Div(dcc.Dropdown(df2['area'].unique(),
                                      df2['area'].unique(),
                                      multi=True, id='second_figure_selector'))
            ]),
            html.Div(className='col', children=[
                'Основным каналом комуникации являються социальные сети, особенно Фейсбук. Поскольку '
                'этот канал позволяет напрямую общаться бренду с его пользователями. +из през добавить'
            ]),
        ])
    ]),

    #  -------------------------------  third figure  --------------------------------

    html.Div(className='container', children=[
        html.Div(className='row', children=[
            html.Div(className='col', children=[
                'На основании проведённых исследований составляется репутационный профиль бренда. '
                'Основными блоками являются: Для автомобилей основным слабым местом являлись - , '
                'для табачки - , а для бытовой техники - +++++++++++++'
            ]),

            html.Div(className='col', children=[
                html.Div(dcc.Graph(id='third_figure')),
                html.Div(dcc.Dropdown(df2['area'].unique(),
                                      multi=True, id='third_figure_selector'))
            ]),
        ])
    ]),

    #  -------------------------------  fourth figure  --------------------------------

    html.Div(className='container', children=[
        html.Div(className='row', children=[
            html.Div(className='col', children=[
                html.Div(dcc.Graph(id='fourth_figure')),
                html.Div(className='col', children=[
                    dcc.Slider(id='fourth_figure_selector1', vertical=True, min=0, max=500, step=1, value=0,
                               marks={i: f'{i}' for i in range(0, 600, 100)}),
                    dcc.Slider(id='fourth_figure_selector2', vertical=True, min=0, max=500, step=1, value=0,
                               marks={i: f'{i}' for i in range(0, 600, 100)}),
                    dcc.Slider(id='fourth_figure_selector3', vertical=True, min=0, max=500, step=1, value=0,
                               marks={i: f'{i}' for i in range(0, 600, 100)}),
                    dcc.Slider(id='fourth_figure_selector4', vertical=True, min=0, max=500, step=1, value=0,
                               marks={i: f'{i}' for i in range(0, 600, 100)}),
                ]),
            ]),
            html.Div(className='col', children=[
                'Большое влияние на продажи каждого бренда имеют рекламные активности в медиа. Основными каналами '
                'по-прежнему остаются ТВ и Интернет. Благодаря проведённым исследованиям в фармацептической отрасли, '
                'можем сказать, что наибольшое влияние имеет на аудиторию комуникация через ТВ.'
            ]),
        ])
    ]),
])
