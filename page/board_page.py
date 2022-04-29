import dash
import numpy as np
from dash import Dash, dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import pypyodbc
from page.element.model_for_ts import ONLY_PREDICTION_WITH_TRP


conn = pypyodbc.connect('Driver={SQL SERVER};'
'Server=UAIEV1SQLM01.emea.media.global.loc;'
'Database=AmplifiDataDashboards;'
'Trusted_Connection=yes;'
'UID=AmplifiDataRobot;'
'PWD=Straight6probably#;')


qwery1 = 'select * from industry'
df1 = pd.read_sql(qwery1, conn)
qwery2 = 'select * from category'
df2 = pd.read_sql(qwery2, conn)
qwery3 = 'select * from social_network'
df3 = pd.read_sql(qwery3, conn)
print('close')
conn.close()


@callback(Output('first_figure', component_property='figure'),
          Input('first_figure_selector', component_property='value'),
          suppress_callback_exceptions=True)
def figure1(size1):
    global df1
    df = df1[df1['area'].isin(size1)].copy()
    df['size'] = df[df.columns[1:-1]].sum(axis=1)
    for i in df.columns[1:-2]:
        df[i] = (df[i] / df['size']) * 100
    _fig = px.scatter(df, x='positive', y='negative', size='size', size_max=50,
                      color='area', hover_name='area', color_discrete_sequence=df['color'].values,
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


@callback(Output('second_figure', component_property='figure'),
          Input('second_figure_selector', component_property='value'),
          suppress_callback_exceptions=True)
def figure2(size1):
    global df2
    df = df2[df2['area'] == size1].copy()
    _fig = px.bar(df, y='categor', x=['negative', 'neutral', 'positive'], orientation='h',
                  color_discrete_sequence=['#ED7D31', '#D9D9D9', '#27AB6C'])
    _fig.update_layout(
        plot_bgcolor='#151515',
        paper_bgcolor='black',
        font_color='white',
        xaxis={'showgrid': False,
               'zeroline': False},
        yaxis={'showgrid': False,
               'zeroline': False},
        showlegend=False
    )
    return _fig


@callback(Output('third_figure', component_property='figure'),
          Input('third_figure_selector', component_property='value'),
          suppress_callback_exceptions=True)
def figure3(size1):
    global df3
    df: pd.DataFrame = df3[df3['area'].isin(size1)].copy()
    df = df.groupby(by=['network']).sum()
    df.reset_index(inplace=True)
    df['size'] = df[df.columns[1:]].sum(axis=1)
    for i in df.columns[1:-1]:
        df[i] = (df[i] / df['size']) * 100
    _fig = px.scatter(df, x='positive', y='negative', size='size', size_max=70, hover_name='network',
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


@callback(Output('fourth_figure', component_property='figure'),
          Input('fourth_figure_selector', component_property='value'),
          suppress_callback_exceptions=True)
def figure4(size1):
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


@callback(Output('fifth_figure', component_property='figure'),
          Input('fifth_figure_selector1', component_property='value'),
          Input('fifth_figure_selector2', component_property='value'),
          Input('fifth_figure_selector3', component_property='value'),
          Input('fifth_figure_selector4', component_property='value'),
          suppress_callback_exceptions=True)
def figure5(trp0, trp1, trp2, trp3):
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
                html.Div('BRAND REPUTATION RESEARCH', className='title'),
                html.Div('Инструмент, который позволяет в быстро меняющейся среде вовремя среагировать на возникающие '
                         'угрозы в веб-пространстве. Благодаря мониторингу бренда в социальных сетях и медиа стало '
                         'возможным отслежить, что говорят о продукте, выявить его слабые места и найти возможность '
                         'улучшить продукт и мнение о нём.', className='first_text'),
                html.Div('Люди легко переносят отношение к конкретному продукту на весь бренд, вовремя отрабатывать '
                         'негатив - ключевой момент в работе над репутацией Тематика проблем меняется постоянно '
                         'Non-stop анализ данных позволяет быть в курсе проблем и вовремя на них реагировать'
                         , className='second_text')
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

    #  -------------------------------  first figure  --------------------------------  #

    html.Div(className='container', children=[
        html.Div(className='row', children=[
            html.Div(className='col', children=[
                'Исследования репутации бренда проводят в разных отрослях. '
                'Самый большой уровень негатива у службы такси и у доставки еды, а самый низкий у крупной бытовой '
                'технике и у самокатов. Чаще всего пользователи упоминают проблемы с тормозами и конфликты между '
                'пешеходами и водителями самокатов. Основные всплески активности обсуждений – травматизм пешеходов '
                'при участии самокатов. Среди исследованных категорий автомобилей доставка автомобилей имеет наибольшую '
                'долю положительных упоминаний, а больше всего негатива — качество двигателей. Проблемы с двигателем '
                'обсуждались в 3% всех упоминаний о неисправности, не так часто, как предполагалось. Основное количество '
                'негативных отзывов связано с расходом топлива и масла, а также проблемами с клапанами и цилиндрами.'
            ]),
            html.Div(className='col', children=[
                html.Div(dcc.Graph(id='first_figure')),
                html.Div(dcc.Dropdown(df1['area'],
                                      df1['area'],
                                      multi=True, id='first_figure_selector'))
            ]),
        ])
    ]),

    #  -------------------------------  second figure  --------------------------------  #

    html.Div(className='container', children=[
        html.Div(className='row', children=[
            html.Div(className='col', children=[
                html.Div(dcc.Graph(id='second_figure')),
                html.Div(dcc.Dropdown(df2['area'].unique(),
                                      df2['area'].unique()[1],
                                      id='second_figure_selector'))
            ]),
            html.Div(className='col', children=[
                'Для каждого бренда были выделины основные тематики обсуждений бренда в медиа. На основании '
                'проведённых исследований было выявлено, что наиболее болезненными темами в автомобильной отрасли '
                'являются проблемы с цилиндрами, расход масла, проблемы с клапанами, перегрев двигателя, проблемы с '
                'рулевой рейкой, проблемы с коленвалом, проблемы с тормозами, проблемы с системой климат-контроля, '
                'проблемы с ремнями безопасности, проблемы с кнопками управления. Негативные упоминания о телевизорах '
                'обычно касаются качества экрана, программных ошибок и плохого комплекта оборудования. Покупателей '
                'раздражает, когда сторонние продавцы часто меняют цены, особенно в период Черной пятницы. Также среди '
                'наиболее обсуждаемых проблем крупной бытовой техники — отсутствие некоторых деталей / руководств '
                'пользователя, программные ошибки и проблемная очистка'
            ]),
        ])
    ]),

    #  -------------------------------  third figure  --------------------------------  #

    html.Div(className='container', children=[
        html.Div(className='row', children=[
            html.Div(className='col', children=[
                'FACEBOOK - ИНИЦИАТОР ПОВОДОВ, СРАБАТЫВАЕТ КАК ТРИГГЕР ДЛЯ ИНФОРМАЦИОННОЙ ВОЛНЫ',
                html.P(),
                'Основным каналом комуникации являються социальные сети, особенно Фейсбук. Поскольку этот канал '
                'позволяет напрямую общаться бренду с его пользователями. Социальные сети выступают в роли сервисных '
                'центров, где пользователи сообщают о проблемах продукта. Также социальные сети часто имеют наибольшую '
                'долю негатива. Для крупной бытовой техники онлайн-магазины также являются среди главных платформ. '
                'Они, как правило, выступают в качестве места основного обсуждения преимуществ и недостатков продуктов, '
                'опыта использования и т. д. Для автомобилей важным каналом коммуникаций также является youtube '
                '(обзоры автомобилей, сравнение автомобилей, краш-тесты, руководства по техническому обслуживанию).'
            ]),

            html.Div(className='col', children=[
                html.Div(dcc.Graph(id='third_figure')),
                html.Div(dcc.Dropdown(df3['area'].unique(),
                                      df3['area'].unique(),
                                      multi=True, id='third_figure_selector'))
            ]),
        ])
    ]),

    #  -------------------------------  fourth figure  --------------------------------  #
    html.Div(className='container', children=[
        html.Div(className='row', children=[
            html.Div(className='col', children=[
                html.Div(dcc.Graph(id='fourth_figure')),
                html.Div(dcc.Dropdown(df1.index,
                                      df1.index,
                                      multi=True, id='fourth_figure_selector'))
            ]),
            html.Div(className='col', children=[
                'На основании проведённых исследований составляется репутационный профиль бренда. Репутационный '
                'профиль формируется на основе самых обсуждаемых тем в сети. Каждый имеет свой вес в общей оценке '
                'уровня репутации бренда. Для крупной бытовой техники Performance имеет самый высокий рейтинг из-за '
                'в основном положительных и нейтральных отзывов о продуктах на платформах электронной коммерции. '
                'Категории «Инновации» и «Лидерство» самые низкие из-за большой доли нейтральных комментариев в '
                'обзорах. Для телевизоров ситуация похожа с Performance, но самой низкой по своему характеру '
                'является категория «Товары/Услуги» (обсуждения ремонтов, программных багов и т. д.). Для автомобилей '
                'самый низкий уровень у категории «Инновации» из-за высокой конкуренции среди производителей '
                'автомобилей, поэтому достаточно сложно выделиться инновационностю продукта. Для службы такси '
                'слабым местом является «Место работы», поскольку тема кадровой политики имеет самый высокий '
                'уровень негатива. '
            ]),
        ])
    ]),

    #  -------------------------------  fifth figure  --------------------------------  #

    html.Div(className='container', children=[
        html.Div(className='row', children=[
            html.Div(className='col', children=[
                'Большое влияние на продажи каждого бренда имеют рекламные активности в медиа. Основными каналами '
                'по-прежнему остаются ТВ и Интернет. Благодаря проведённым исследованиям в фармацевтической отрасли, '
                'можем сказать, что наибольшее влияние имеет на аудиторию коммуникация через ТВ. Так вы можете видеть '
                'как ПРИМЕРНО может меняться уровень продаж от рекламной активности бренда на ТВ и в Интернете. Важно '
                'отметить, что на практике для каждого бренда модель подбирается индивидуально в зависимости от доли '
                'его на рынке, сезонности продукта, активности конкурентов и других имеющихся факторов.'
            ]),
            html.Div(className='col', children=[
                html.Div(dcc.Graph(id='fifth_figure')),
                html.Div(className='slider_group', children=[
                    dcc.Slider(id='fifth_figure_selector1', vertical=True, min=0, max=500, step=1, value=0,
                               className='slider4', marks={i: f'{i}' for i in range(0, 600, 100)}),
                    dcc.Slider(id='fifth_figure_selector2', vertical=True, min=0, max=500, step=1, value=0,
                               className='slider4', marks={i: f'{i}' for i in range(0, 600, 100)}),
                    dcc.Slider(id='fifth_figure_selector3', vertical=True, min=0, max=500, step=1, value=0,
                               className='slider4', marks={i: f'{i}' for i in range(0, 600, 100)}),
                    dcc.Slider(id='fifth_figure_selector4', vertical=True, min=0, max=500, step=1, value=0,
                               className='slider4', marks={i: f'{i}' for i in range(0, 600, 100)}),
                ]),
            ]),
        ])
    ]),
])
