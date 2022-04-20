import dash
from dash import Dash, dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
from page import board_page


app = Dash(__name__,
           external_stylesheets=[dbc.themes.CYBORG],
           meta_tags=[
               {
                   'name': 'description',
                   'content': 'My description'
               },
               {
                   'name': 'viewport',
                   'content': 'width=device-width, initial-scale=1.0'
               },
               {
                   'http-equiv': 'X-UA-Compatible',
                   'content': 'IE=edge'
               },
           ])

app.title = 'bi_board'

app.layout = html.Div(className='body',
                      children=[
                          dcc.Location(id='url', refresh=False),
                          html.Div(id='page-content'),
                         ])


@callback(Output('page-content', 'children'),
          Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/':
        return board_page.layout
    else:
        return '404'


if __name__ == '__main__':
    app.run_server(debug=True)  # False, host='0.0.0.0'
