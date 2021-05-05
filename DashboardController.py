import dash as dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
from RandomForest import *
from PIL import Image
from LRAlgo import * #import

class DashboardController:

    def __init__(self, data_set):
        self.data_set = data_set
        self.linear_regression = LRAlgo(self.data_set)
        #self.linear_regression.calculate_lr('duration','avg_vote')
        #self.multiple_linear_regression = MultipleLR(self.data_set)
        self.random_forest = RandomForest(self.data_set)
        self.random_forest.random_forest('avg_vote')
        self.random_forest.generate_rf_pngs()
        print(self.random_forest.get_pred())

    def new_row_figure(self, figure):
        return dbc.Row([dbc.Col(dcc.Graph(figure=figure))])

    def new_row_imgs(self,i):
        try:
            image = Image.open(fr"assets\tree{i}.png")
            image = image.resize((3000, 900), Image.ANTIALIAS)
            image.save(fp=fr'assets\tree{i}.png')
            return html.Div([html.Img(src=f'/assets/tree{i}.png')],
                            className='text-center')
        except:
            print('probably not ready for input yet :)')

    def dash_application(self):
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col(html.H1('Movie Dashboard',
                                className='text-center'))
            ]),

                html.Div(id='picture_rows', children=[self.new_row_imgs(i) for i in range(0, 5)])
        ])

        if __name__ == 'DashboardController':
            app.run_server(debug=True,port=8061)