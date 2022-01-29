
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from lightgbm import LGBMClassifier
import catboost as catb
import dask.dataframe as dd
from pathlib import Path
import luigi
import pickle
from current import estimator_test

class LuigiPipeline(luigi.Task):
    TRAIN_FILENAME = luigi.Parameter()
    FEATURES_FILENAME = luigi.Parameter()
    TEST_FILENAME = luigi.Parameter()

    def run(self):
        data_train = pd.read_csv(self.TRAIN_FILENAME)
        features = dd.read_csv(self.FEATURES_FILENAME, blocksize=25e6, sep='\t')
        features_train = features.loc[features['id'].isin(data_train['id'])].compute()

        data_test = pd.read_csv(self.TEST_FILENAME)
        features_test = features.loc[features['id'].isin(data_test['id'])].compute()

        datapreprocessing = estimator_test.DataPreprocessing()
        df_all = datapreprocessing.transform(data_train, features_train)
        df_all = estimator_test.reduce_mem_usage(df_all)
        df_all_test = datapreprocessing.transform(data_test, features_train)
        df_all_test = estimator_test.reduce_mem_usage(df_all_test)
        data_train = None
        data_test = None
        features = None
        features_train = None

        featuregenerator = estimator_test.Featuregenerator()
        train_df = featuregenerator.fit(df_all, 316, 337)
        test_df = featuregenerator.transform(df_all_test)

        estimator = estimator_test.Estimator()
        estimator.fit(train_df)
        predicted = estimator.predict(test_df)
        print(predicted.head())
        predicted.to_csv('data/answers_test.csv', index=False)

    def output(self):
        return luigi.LocalTarget('data/answers_test.csv')


if __name__ == '__main__':

    #TRAIN_FILENAME = "current/data/data_train.csv"
    #FEATURES_FILENAME = "current/data/features.csv"
    #TEST_FILENAME = "current/data_test.csv"

    luigi.build([LuigiPipeline("current/data/data_train.csv", "current/data/features.csv", "current/data_test.csv")])