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
df1.drop(columns='id', inplace=True)
qwery2 = 'select * from category'
df2 = pd.read_sql(qwery2, conn)
df2.drop(columns='id', inplace=True)
qwery3 = 'select * from social_network'
df3 = pd.read_sql(qwery3, conn)
df3.drop(columns='id', inplace=True)
qwery4 = 'select * from rep_profile'
df4 = pd.read_sql(qwery4, conn)
df4.drop(columns='id', inplace=True)
conn.close()
print('close')


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
                  color_discrete_sequence=['#dd1a3a', '#638b8a', '#00cabe'])
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
    global df4
    df = df4[df4['area'] == size1].copy()

    _fig = go.Figure(go.Sunburst(
        labels=df['category'],
        parents=[""]+['TOTAL']*(df.shape[0]-1),
        values=df['value']
    ))
    _fig.update_layout(
        margin=dict(t=10, l=10, r=10, b=10),
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
                html.Div('The tool allows you to react in time to emerging threats in the fast-changing web '
                         'environment. By monitoring the brand on social media and social networks, it is possible to '
                         'track what is being said about a product, identify its weaknesses and find opportunities to '
                         'improve the product and opinions about it. People simply transfer their attitudes to a '
                         'certain product to the whole brand, so dealing with negativity in a timely manner is a '
                         'key moment in your reputational work. NON-STOP DATA ANALYSIS allows you to be aware of '
                         'problems and react to them instantly, as the subject of problems changes all the time.',
                         className='first_text')
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
                html.Div(className='smol_title', children='BRANCHES'),
                'Brand reputation is the consumers’ perception of a particular brand, product, or service. Therefore, '
                'its research is important in all industries without exception. Based on results from industries such '
                'as tobacco, gaming, taxi service, automotive, large home appliances, and logistics, we can conclude '
                'that taxi service and food delivery have the highest level of negativity, while large home '
                'appliances and scooters have the lowest. The analysis shows that the main accidental discussions '
                'about scooters are pedestrian injuries when scooters are involved. Users say they are most concerned '
                'about the brakes and conflicts between pedestrians and scooter drivers. Regarding cars, it’s '
                'important to highlight that among the categories surveyed, delivery has the highest rate of '
                'positive mentions, and engine quality is the negative one. Meanwhile, engine problems were '
                'discussed in 3% of all faulty mentions — not as often as basically thought. The fuel and oil '
                'consumption, as well as valve and cylinder problems, described the main number of negative mentions.'
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
                html.Div(className='smol_title', children='CATEGORIES'),
                'In brand research, the important step is to identify the key themes consumers are talking about. It '
                'allows the problem areas of a product or service to be explored in more detail. For example, based '
                'on the research conducted, we found that the most troublesome topics in the automotive industry are '
                'cylinder problems, high oil consumption, valves, engine overheating, steering rack, crankshaft, '
                'brakes, climate control, seat belts and control buttons. As for TVs, the top negative mentions '
                'frequently relate to screen quality, programming errors and bad set of equipment. Customers annoy '
                'when third-party sellers change prices again and again, especially during Black Friday. The lack of '
                'some parts / user manuals, software bugs and problematic cleaning are as well among the most '
                'discussed problems of large home appliances.'
            ]),
        ])
    ]),

    #  -------------------------------  third figure  --------------------------------  #

    html.Div(className='container', children=[
        html.Div(className='row', children=[
            html.Div(className='col', children=[
                html.Div(className='smol_title', children='SOCIAL MEDIA'),
                'FACEBOOK IS THE INITIATOR OF THE OCCASIONS, TRIGGER OF THE INFORMATION WAVE.',
                html.P(),
                'The primary communication channel is social media, and Facebook in particular. This channel allows a '
                'brand to communicate with their users personally. Social networks are the service centers where users '
                'signal product problems. It often has the largest share of negativity. One of the main platforms for '
                'large home appliances is also online retailers, for example. They go to act as a venue for a basic '
                'discussion of the advantages and disadvantages of products, user experiences, etc. As for cars, '
                'YouTube is another important communication channel, where car reviews, comparisons, crash tests, and '
                'maintenance manuals are published mainly. '
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
                html.Div(dcc.Dropdown(df4['area'].unique(),
                                      df4.loc[0, 'area'],
                                      id='fourth_figure_selector'))
            ]),
            html.Div(className='col', children=[
                html.Div(className='smol_title', children='REPUTATION PROFILE'),
                'A brand reputation profile based on the research — what is being said about the brand, what are the '
                'most vulnerable areas of the product and identified its key problems. In other words, we form a '
                'reputation profile based on the most discussed topics online, and each topic has its own weight in '
                'the overall assessment of the brand’s reputation. For example, as for large appliances and TVs, '
                '“Performance” has the highest score, mainly because of the positive and neutral product reviews on '
                'e-commerce platforms. The lowest category for TVs is “Products/Services” (discussions of repairs, '
                'software bugs, etc.). The lowest score for cars is found in the Innovation category because of top '
                'competition among manufacturers. Thus, it is quite difficult to stand out with product innovation. '
                'As for the taxi service, “Place of work” is the weakest point, as personnel policy has the highest '
                'level of negativity.'
            ]),
        ])
    ]),

    #  -------------------------------  fifth figure  --------------------------------  #

    html.Div(className='container', children=[
        html.Div(className='row', children=[
            html.Div(className='col', children=[
                html.Div(className='smol_title', children='SALES'),
                'The key indicator of a user’s attitude to a product is the level of its sales. Media advertising '
                'activity has a major impact on sales for each brand. TV and the Internet are still the main channels. '
                'Research in the pharmaceutical industry shows that communication via TV has the greatest influence '
                'on the audience. So, you can see how the change in sales levels difference between the brand’s '
                'advertising activity on TV and the Internet. It’s important to highlight that we customize the model '
                'for each brand depending on its market share, product seasonality, competitor activity and other '
                'factors.'
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
