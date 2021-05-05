import pandas as pd


class DataClean:

    def __init__(self, data_set):                                       # Init dataclean
        self.data_set = data_set

    def combine_data_sets(self, merge_file):                            # Combines dat a set from init with a parsed dataset - finds key itself
        self.data_set = pd.merge(self.data_set, merge_file)
        return self.data_set

    def keep_columns(self, col):  # Removes column from dataset - takes list input pass array of strings
        for i in self.data_set.head():
            if i not in col:
                del self.data_set[i]
        return self.data_set

    def drop_na_rows(self):                                             #Drops all rows with N/A (Wont delete oneHotEncodes because we havent split by the time we use this function
        self.data_set = self.data_set.dropna()
        return self.data_set

    def drop_non_numeric(self):                                         #Currently drops all years that are non-numeric
        self.data_set = self.data_set[self.data_set.year.apply(lambda x: x.isnumeric())]
        return self.data_set

    def full_clean(self, merge_file, col):                              #Runs all of the above functionsD
        self.combine_data_sets(merge_file)
        self.keep_columns(col)
        self.drop_non_numeric()
        self.drop_na_rows()
        return self.data_set