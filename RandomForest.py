from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.tree import export_graphviz
import os
from sklearn.model_selection import train_test_split
from scipy import sparse
import json
from graphviz import Digraph
from sklearn.metrics import accuracy_score


class RandomForest:

    def __init__(self, data_set):
        self.data_set = data_set
        self.train, self.test = train_test_split(data_set, test_size=0.2)

    def random_forest(self, predict_column):

        self.y_train = np.array(self.train[predict_column]).tolist()
        self.x_train = np.array(self.train.drop([predict_column], axis=1)).tolist()

        self.x_names = self.train.drop([predict_column], axis=1).columns.tolist()

        self.x_test = np.array(self.test.drop([predict_column], axis=1))
        # self.y_test = np.array(self.test[predict_column]).tolist()

        self.model = RandomForestRegressor(n_estimators=50, bootstrap=True, max_depth=5, max_samples=400)

        self.model = self.model.fit(self.x_train, self.y_train)
        # y_pred = self.model.predict(self.x_test)        #træner modellen
        # print(accuracy_score(self.y_test,np.array(y_pred).tolist))

    def generate_rf_pngs(self):

        length_estimators = range(len(self.model.estimators_))
        for i in length_estimators[-5:]:
            estimator = self.model.estimators_[i]
            export_graphviz(estimator,
                            out_file=fr'dotfiles\tree{i}.dot',
                            feature_names=self.x_names,
                            filled=True,
                            rounded=True)

            os.system(fr'dot -Txdot_json -ojsonfiles\tree{i}.json dotfiles\tree{i}.dot')

            a = self.model.estimators_[i].decision_path(self.x_test[0].reshape(1, -1))
            b = sparse.csr_matrix(a).toarray()
            for j in range(len(b[0])):
                    with open(fr'jsonfiles\tree{i}.json', 'r+') as f:
                        data = json.load(f)
                        if b[0][j] == 1:
                            data['objects'][j]['fillcolor'] = 'red'
                        f.seek(0)
                        json.dump(data, f, indent=4)
                        f.truncate()

            digraph = Digraph()

            for a in range(len(data['objects'])):
                digraph.node(str(data['objects'][a]['_gvid']), str(data['objects'][a]['label']),
                             style='filled',
                             fillcolor=str(data['objects'][a]['fillcolor']))

            for b in range(len(data['edges'])):
                digraph.edge(str(data['edges'][b]['tail']), str(data['edges'][b]['head']))

            digraph.render(filename=f'digraphfiles\digraph{i}.dot')

            os.system(fr'dot -Tpng digraphfiles\digraph{i}.dot -o assets\tree{i}.png')

    def get_pred(self):
        predictions_all = np.array([tree.predict(self.x_test[0].reshape(1, -1)) for tree in self.model.estimators_])
        overall_prediction = self.model.predict(self.x_test[0].reshape(1, -1))
        return predictions_all, overall_prediction