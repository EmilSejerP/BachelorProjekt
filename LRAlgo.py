from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

class LRAlgo:

    #Data_set is the data frame you pass the model
    #the col's are the two specific Columns you wanna pass (must be string)
    def __init__(self,data_set):
        self.data_set = data_set

    def calculate_lr(self, col_1, col_2):

        X = np.reshape(self.data_set[col_1].tolist(),(-1,1))
        y = self.data_set[col_2].tolist()

        model = LinearRegression()
        model.fit(X,y)

        x_range = np.linspace(X.min(), X.max(), 100)
        y_range = model.predict(x_range.reshape(-1, 1))

        def return_fig():
            fig = px.scatter(self.data_set, x=col_1, y=col_2, opacity=0.65)
            fig.add_traces(go.Scatter(x=x_range, y=y_range, name='Regression Fit'))
            return fig.show()

        return return_fig()