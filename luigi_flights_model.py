import luigi
import numpy as np
import pandas as pd
import sklearn

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Load_Data(luigi.Task):

    def run(self):
        data = pd.read_csv(r"D:\Arbeit\HIWI\Jupyter Notebooks\asca_part_IV\5_sklearn\data\2_flights_reduced.csv",
                           index_col=0
                           )
        features = data.drop(["ARRIVAL_DELAY"], axis=1)
        label = data[["ARRIVAL_DELAY"]]
        features.to_csv(self.output()["feature"].path, index=False)
        label.to_csv(self.output()["label"].path, index=False)

    def output(self):
        return {
            'feature': luigi.LocalTarget('features.csv'),
            'label': luigi.LocalTarget('label.csv')
        }

class Clean_Data(luigi.Task):

    def requires(self):
        return Load_Data()
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


class Balancing_Data(luigi.Task):

    def requires(self):
        return(Clean_Data())

    def run(self):

        features = pd.read_csv(self.input()["feature"].path)
        label = pd.read_csv(self.input()["label"].path)

        delayed_flights = label[label['ARRIVAL_DELAY'] == 'delayed'].sample(10000, replace=False)
        timely_flights = label[label['ARRIVAL_DELAY'] == 'timely'].sample(10000, replace=False)
        label = pd.concat([delayed_flights, timely_flights])
        features = features.loc[label.index]

        label.to_csv(self.output()["label"].path, index=False)
        features.to_csv(self.output()["feature"].path, index=False)

    def output(self):
        return {
            'feature': luigi.LocalTarget('balanced_features.csv'),
            'label': luigi.LocalTarget('balanced_label.csv')
        }
class Feature_Encoding(luigi.Task):
    def requires(self):
        return(Balancing_Data())

    def run(self):
        features = pd.read_csv(self.input()["feature"].path)

        features['SEASON'] = features['MONTH'].apply(lambda x: 'WINTER' if x in [12, 1, 2] else ('SPRING' if x in [3, 4, 5] else ('SUMMER' if x in [6, 7, 8] else 'AUTUMN')))
        features.drop('MONTH', axis=1, inplace=True) #wieso brauche ich hier inplace?

        one_hot_encoder = OneHotEncoder()
        ordinal_encoder = OrdinalEncoder()

        categorical_features = ['SEASON', 'AIRLINE']
        encoded_features = one_hot_encoder.fit_transform(features[categorical_features])
        encoded_features = pd.DataFrame(encoded_features.toarray(), columns=one_hot_encoder.get_feature_names_out(categorical_features))
        features = pd.concat([features.drop(columns=categorical_features), encoded_features], axis=1)

        features.to_csv(self.output()["feature"].path, index=False)

    def output(self):
        return {
            'feature': luigi.LocalTarget('encoded_features.csv')
        }

class Label_Encoding(luigi.Task):

    def requires(self):
        return(Balancing_Data())

    def run(self):
        label = pd.read_csv(self.input()["label"].path)

        encoder = LabelEncoder()
        label['ARRIVAL_DELAY'] = encoder.fit_transform(label['ARRIVAL_DELAY'])
        label.to_csv(self.output()["label"].path, index=False)

    def output(self):
        return {
            'label': luigi.LocalTarget('encoded_label.csv')
        }


class Correlation_Analysis(luigi.Task):

    def requires(self):
        return(Feature_Encoding())

    def run(self):
        pass    #Although 'SCHEDULED_TIME' and 'DISTANCE' are highly correlated,
                # they dont get dropped from the dataframe for this task bc we expect
                #to need as much information as we can get to accuratly preict the label
    def output(self):
        pass

class Train_Data(luigi.Task):

    def requires(self):
        return [Feature_Encoding(), Label_Encoding()]

    def run(self):
        features = pd.read_csv(self.input()[0]["feature"].path)
        label = pd.read_csv(self.input()[1]["label"].path)

        x = features
        y = label['ARRIVAL_DELAY']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


        logistic_regression = LogisticRegression(max_iter=10000)
        logistic_regression.fit(x_train, y_train)

        test_indices = x_test.index
        actual_value = y_test.values
        predicted_value = logistic_regression.predict(x_test)

        label = pd.DataFrame({'ACTUAL': actual_value, 'PREDICTED': predicted_value})

        label.to_csv(self.output()["label"].path, index=False)

        accuracy = accuracy_score(actual_value, predicted_value)
        print("Accuracy:", accuracy*100, " %")

    def output(self):
        return{
            'label': luigi.LocalTarget('results.csv')
        }


if __name__ == '__main__':
    luigi.build([Train_Data()], local_scheduler=True)