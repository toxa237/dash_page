df = pd.DataFrame(data={'val': [10, 20, 30, 40],
                        'names': ['1', '2', '3', '4'],
                        'pull': [0, 0, 0.2, 0]})
fig1 = go.Figure(data=[go.Pie(values=df['val'], labels=df['names'], pull=df['pull'],
                              marker_colors=['rgb(146, 123, 21)', 'rgb(177, 180, 34)', 'rgb(206, 206, 40)',
                                            'rgb(175, 51, 21)']
                              )])
colors1 = {
    'background': '#111111',
    'text': '#7FDBFF'
}
selector1 = dcc.Slider(
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
def figure1(size1):
    df.loc[3, 'val'] = 40 + size1
    _fig = go.Figure(data=[go.Pie(values=df['val'], labels=df['names'], pull=df['pull'],
                                  marker_colors=['rgb(146, 123, 21)', 'rgb(177, 180, 34)', 'rgb(206, 206, 40)',
                                                 'rgb(175, 51, 21)']
                                  )
                           ])

    _fig.update_layout(
        plot_bgcolor=colors1['background'],
        paper_bgcolor=colors1['background'],
        font_color=colors1['text']
    )
    return _fig