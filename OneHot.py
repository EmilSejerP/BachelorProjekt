import pandas as pd

class OneHot:
    def __init__(self, data_set, encode_column):

        self.data_set = data_set
        self.encode_column = encode_column
        self.data_split = self.data_set[self.encode_column].str.split(", ").apply(pd.Series)

    def encode(self):

        df = (pd.get_dummies(self.data_split[0])
            .add(pd.get_dummies(self.data_split[1]), fill_value=0)
            .astype(int))

        return df
