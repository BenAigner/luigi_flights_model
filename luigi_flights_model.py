import luigi
import numpy as np
import pandas as pd
import sklearn
import configparser

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


def load_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

config = load_config('config.ini')


flights_data_path = config.get('Paths', 'flights_data_path')
# test_size = config.getfloat('DataProcessingParameters', 'test_size') funktioniert nicht
sample_size = config.getint('DataProcessingParameters', 'sample_size')
logistic_regression_max_iter = config.getint('TrainModelParameters', 'logistic_regression_max_iter')

def add_season(features):
    def label_months_to_season(month):

        if month in [12, 1, 2]:
            return 'WINTER'
        elif month in [3, 4, 5]:
            return 'SPRING'
        elif month in [6, 7, 8]:
            return 'SUMMER'
        else:
            return ('AUTUMN')

    features['SEASON'] = features['MONTH'].apply(label_months_to_season)
    features.drop('MONTH', axis=1, inplace=True)


class LoadData(luigi.Task):

    def run(self):
        data = pd.read_csv(flights_data_path,index_col=0)
        features = data.drop(["ARRIVAL_DELAY"], axis=1)
        label = data[["ARRIVAL_DELAY"]]
        features.to_csv(self.output()["feature"].path, index=False)
        label.to_csv(self.output()["label"].path, index=False)

    def output(self):
        return {
            'feature': luigi.LocalTarget('features.csv'),
            'label': luigi.LocalTarget('label.csv')
        }

class DropDuplicatesDropNA(luigi.Task):

    def requires(self):
        return LoadData()
    def run(self):

        features = pd.read_csv(self.input()["feature"].path)
        label = pd.read_csv(self.input()["label"].path)
        features = features.drop_duplicates().dropna()
        label = label.loc[features.index]
        features.to_csv(self.output()["feature"].path, index=False)
        label.to_csv(self.output()["label"].path, index=False)

    def output(self):
        return {
            'feature': luigi.LocalTarget('cleaned_features.csv'),
            'label': luigi.LocalTarget('cleaned_label.csv')
        }


class BalancingData(luigi.Task):

    def requires(self):
        return(DropDuplicatesDropNA())

    def run(self):

        features = pd.read_csv(self.input()["feature"].path)
        label = pd.read_csv(self.input()["label"].path)

        delayed_flights = label[label['ARRIVAL_DELAY'] == 'delayed'].sample(sample_size, replace=False)
        timely_flights = label[label['ARRIVAL_DELAY'] == 'timely'].sample(sample_size, replace=False)
        label = pd.concat([delayed_flights, timely_flights])
        features = features.loc[label.index]

        label.to_csv(self.output()["label"].path, index=False)
        features.to_csv(self.output()["feature"].path, index=False)

    def output(self):
        return {
            'feature': luigi.LocalTarget('balanced_features.csv'),
            'label': luigi.LocalTarget('balanced_label.csv')
        }
class FeatureEncoding(luigi.Task):
    def requires(self):
        return(BalancingData())

    def run(self):
        features = pd.read_csv(self.input()["feature"].path)

        #features['SEASON'] = features['MONTH'].apply(lambda x: 'WINTER' if x in [12, 1, 2] else ('SPRING' if x in [3, 4, 5] else ('SUMMER' if x in [6, 7, 8] else 'AUTUMN')))
        #features.drop('MONTH', axis=1, inplace=True)

        add_season(features)

        one_hot_encoder = OneHotEncoder()

        categorical_features = ['SEASON', 'AIRLINE']
        encoded_features = one_hot_encoder.fit_transform(features[categorical_features])
        encoded_features = pd.DataFrame(encoded_features.toarray(), columns=one_hot_encoder.get_feature_names_out(categorical_features))
        features = pd.concat([features.drop(columns=categorical_features), encoded_features], axis=1)

        features.to_csv(self.output()["feature"].path, index=False)

    def output(self):
        return {
            'feature': luigi.LocalTarget('encoded_features.csv')
        }

class LabelEncoding(luigi.Task):

    def requires(self):
        return(BalancingData())

    def run(self):
        label = pd.read_csv(self.input()["label"].path)

        encoder = LabelEncoder()
        label['ARRIVAL_DELAY'] = encoder.fit_transform(label['ARRIVAL_DELAY'])
        label.to_csv(self.output()["label"].path, index=False)

    def output(self):
        return {
            'label': luigi.LocalTarget('encoded_label.csv')
        }


class CorrelationAnalysis(luigi.Task):

    def requires(self):
        return(FeatureEncoding())

    def run(self):
        pass    #Although 'SCHEDULED_TIME' and 'DISTANCE' are highly correlated,
                # they dont get dropped from the dataframe for this task bc we expect
                #to need as much information as we can get to accuratly preict the label
    def output(self):
        pass


class TrainLogisticRegressionModel(luigi.Task):

    def requires(self):
        return [FeatureEncoding(), LabelEncoding()]

    def run(self):
        features = pd.read_csv(self.input()[0]["feature"].path)
        label = pd.read_csv(self.input()[1]["label"].path)

        x = features
        y = label['ARRIVAL_DELAY']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


        logistic_regression = LogisticRegression(max_iter=logistic_regression_max_iter)
        logistic_regression.fit(x_train, y_train)


        actual_value = y_test.values
        predicted_value = logistic_regression.predict(x_test)

        label = pd.DataFrame({'ACTUAL': actual_value, 'PREDICTED': predicted_value})

        label.to_csv(self.output()["label"].path, index=False)

        accuracy = accuracy_score(actual_value, predicted_value)
        print("Accuracy of Logistic Regression:", accuracy*100, " %")

    def output(self):
        return{
            'label': luigi.LocalTarget('resultsLogisticRegression.csv')
        }

class TrainDecisionTreeModel(luigi.Task):
    def requires(self):
        return [FeatureEncoding(), LabelEncoding()]

    def run(self):
        features = pd.read_csv(self.input()[0]["feature"].path)
        label = pd.read_csv(self.input()[1]["label"].path)                              #redundant immer in jeder task zu formatieren
                                                                                        # kann ich das auch als funktion schreiben?
        x = features
        y = label['ARRIVAL_DELAY']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        decision_tree = DecisionTreeClassifier()
        decision_tree.fit(x_train, y_train)

        actual_value = y_test.values
        predicted_value = decision_tree.predict(x_test)

        label = pd.DataFrame({'ACTUAL': actual_value, 'PREDICTED': predicted_value})

        label.to_csv(self.output()["label"].path, index=False)

        accuracy = accuracy_score(actual_value, predicted_value)
        print("Accuracy of Decison Tree:", accuracy * 100, " %")

    def output(self):
        return{
            'label': luigi.LocalTarget('resultsDecisionTree.csv')
        }

class TrainKNeighborsModel(luigi.Task):
    def requires(self):
        return[FeatureEncoding(), LabelEncoding()]
    def run(self):

        features = pd.read_csv(self.input()[0]["feature"].path)
        label = pd.read_csv(self.input()[1]["label"].path)

        x = features
        y = label['ARRIVAL_DELAY']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        decision_tree = DecisionTreeClassifier()
        decision_tree.fit(x_train, y_train)

        actual_value = y_test.values
        predicted_value = decision_tree.predict(x_test)

        label = pd.DataFrame({'ACTUAL': actual_value, 'PREDICTED': predicted_value})

        label.to_csv(self.output()["label"].path, index=False)

        accuracy = accuracy_score(actual_value, predicted_value)
        print("Accuracy of K Nearest Neighbor:", accuracy * 100, " %")

    def output(self):
        return {
            'label': luigi.LocalTarget('resultsKNearestNeighbor.csv')
        }

class ExecutePipeline(luigi.Task):
    def requires(self):
        return[TrainLogisticRegressionModel(), TrainDecisionTreeModel(), TrainKNeighborsModel()]

    def run(self):
        pass

    def output(self):
        pass



if __name__ == '__main__':
    luigi.build([ExecutePipeline()], local_scheduler=True)