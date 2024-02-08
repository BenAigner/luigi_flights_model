import luigi
import numpy as np
import pandas as pd

class Clean_Data(luigi.Task):

    def run(self):
        data = pd.read_csv(r"D:\Arbeit\HIWI\Jupyter Notebooks\asca_part_IV\5_sklearn\data\2_flights_reduced.csv", index_col=0
                        ).dropna().drop_duplicates()

        features = data.drop(["ARRIVAL_DELAY"], axis=1)
        label = data[["ARRIVAL_DELAY"]  ]
        features.to_csv(self.output()["feature"].path, index=False)
        label.to_csv(self.output()["label"].path, index=False)

    def output(self):
        return {
            'feature': luigi.LocalTarget('features.csv'),
            'label': luigi.LocalTarget('label.csv')
        }


class Labal_Balancing_Data(luigi.Task):

    def requires(self):
        return(Clean_Data())

    def run(self):

        features = pd.read_csv(self.input()["feature"].path)
        label = pd.read_csv(self.input()["label"].path)

        #delayed_flights = data[data['ARRIVAL DELAY'] == 'delayed'].sample(10000, replace=False)
        #timely_flights = data[data['ARRIVAL DELAY'] == 'timely'].sample(10000, replace=False)
        pass
    def output(self):
        pass

class Feature_Encoding(luigi.Task):
    def requires(self):
        pass

    def run(self):
        pass

    def output(self):
        pass

class Label_Encoding(luigi.Task):

    def requires(self):
        pass

    def run(self):
        pass

    def output(self):
        pass

class Correlation_Analysis(luigi.Task):

    def requires(self):
        pass

    def run(self):
        pass

    def output(self):
        pass




if __name__ == '__main__':
    luigi.build([Clean_Data()], local_scheduler=True)