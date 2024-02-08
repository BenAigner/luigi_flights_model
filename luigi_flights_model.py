import luigi
import numpy as np
import pandas as pd

class Clean_Data(luigi.Task):

    def run(self):
        data = pd.read_csv(r"C:\Users\49157\HIWI_Gitlab\asca_part_IV\asca_part_IV\5_sklearn\data\2_flights_reduced.csv", index_col=0
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
        #delayed_flights = data['ARRIVAL DELAY'] == 'delayed'].sample(n_samples, replace=False)
        #timely_flights = data['ARRIVAL DELAY'] == 'delayed'].sample(n_samples, replace=False)
        pass
    def output(self):
        pass


if __name__ == '__main__':
    luigi.build([Clean_Data()], local_scheduler=True)