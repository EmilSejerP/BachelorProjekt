import plotly.express as px
from sklearn.linear_model import LinearRegression

class MultipleLR:

    def __init__(self, data_set):
        self.data_set = data_set

    def return_fig(self, predict_column):

        X = self.data_set.drop(columns=[predict_column])
        y = self.data_set[predict_column]

        model = LinearRegression()
        model.fit(X, y)

        colors = ['Positive' if c > 0 else 'Negative' for c in model.coef_]

        fig = px.bar(
            x=X.columns, y=model.coef_, color=colors,
            color_discrete_sequence=['red', 'blue'],
            labels=dict(x='Feature', y='Linear coefficient'),
            title='Weight of each feature for predicting petal width'
        )

        def get_zip_values():
            return list(zip(X.columns,model.coef_))

        return fig