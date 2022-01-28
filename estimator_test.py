import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from lightgbm import LGBMClassifier
import catboost as catb
import dask.dataframe as dd


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2  # подсчитываем память потребляемую изначальным датасетом
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:  # проходимся по всем колонкам
        col_type = df[col].dtype  # узнаем тип колонки

        if col_type != object:
            c_min = df[col].min()  # смотрим минимальное значение признака
            c_max = df[col].max()  # смотрим максимальное значение признака
            if str(col_type)[:3] == 'int':  # if int
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:  # сравниваем с int8
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:  # сравниваем с int16 и.т.д.
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:  # если был тип object, то меняем его тип на пандасовский тип 'category', на нем разные агрегации данных работают в разы быстрее
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2  # считаем сколько теперь у нас занято памяти
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))  # и выводим статистику
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


class DataPreprocessing:

    def fit(self, data_train, features_train):
        return data_train, features_train

    def transform(self, data_train, features_train):
        """Трансформация данных"""

        dublicated = features_train[features_train['id'].duplicated(keep=False)].sort_values(by='id')[
            'id'].value_counts()
        data_merged_train = pd.merge(data_train, features_train, on='id')
        tmp = data_merged_train.loc[data_merged_train['id'].isin(dublicated.index)]
        tmp['delta'] = abs(tmp['buy_time_x'] - tmp['buy_time_y'])
        tmp.sort_values(by=['Unnamed: 0_x', 'delta'], inplace=True)
        duplicates = tmp['Unnamed: 0_x'].duplicated()
        duplicates_to_delete = duplicates[duplicates.values == True]
        data_merged_train.drop(duplicates_to_delete.index, axis=0, inplace=True)
        data_merged_train['time_delta'] = data_merged_train['buy_time_x'] - data_merged_train['buy_time_y']
        data_merged_train.drop(['Unnamed: 0_x', 'Unnamed: 0_y', 'buy_time_y'], axis=1, inplace=True)

        return data_merged_train

class Featuregenerator():

    def __init__(self):
        self.log_df = None
        self.action_model = None

    def fit(self, df_all, start, stop):
        from datetime import datetime, date, time
        from datetime import timedelta
        import holidays
        from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

        X = df_all.drop(columns=['target'])
        y = df_all['target']

        df = X[['id', 'vas_id']]
        df['vas_id_01'] = 0
        df['vas_id_02'] = 0
        df['vas_id_04'] = 0
        df['vas_id_05'] = 0
        df['vas_id_06'] = 0
        df['vas_id_07'] = 0
        df['vas_id_08'] = 0
        df['vas_id_09'] = 0

        df.loc[df['vas_id'] == 1.0, 'vas_id_01'] = 1
        df.loc[df['vas_id'] == 2.0, 'vas_id_02'] = 1
        df.loc[df['vas_id'] == 4.0, 'vas_id_04'] = 1
        df.loc[df['vas_id'] == 5.0, 'vas_id_05'] = 1
        df.loc[df['vas_id'] == 6.0, 'vas_id_06'] = 1
        df.loc[df['vas_id'] == 7.0, 'vas_id_07'] = 1
        df.loc[df['vas_id'] == 8.0, 'vas_id_08'] = 1
        df.loc[df['vas_id'] == 9.0, 'vas_id_09'] = 1

        self.log_df = df.groupby(['id', 'vas_id'], as_index=False)['vas_id_01', 'vas_id_02', 'vas_id_04', 'vas_id_05',
                                                                   'vas_id_06', 'vas_id_07', 'vas_id_08', 'vas_id_09'].sum()
        X = X.reset_index().merge(self.log_df, on=['id', 'vas_id'], how='left').set_index('index').fillna(0)
        list_id = list(X.id.unique())
        for i in list_id:
            ix = X.loc[X['id'] == i, 'vas_id'].value_counts()
            for k in ix.index:
                X.loc[X['id'] == i, 'vas_id_0' + str(int(k))] += ix[k]

        # Добываю максимум информации из 'buy_time_x', остальные признаки, включая выходные и праздники
        # к увеличению метрик не привели
        X['date'] = list(map(datetime.fromtimestamp, X['buy_time_x']))
        X['month'] = X['date'].apply(lambda x: x.timetuple()[1])
        X['day'] = X['date'].apply(lambda x: x.timetuple()[7])
        X['weekofyear'] = X['buy_time_x'].apply(lambda x: pd.to_datetime(date.fromtimestamp(x)).weekofyear)

        X['time_max'] = X.buy_time_x.max()
        X['novelty'] = X['time_max'] - X['buy_time_x']

        few = pd.DataFrame(X['id'].value_counts())
        few = few.loc[few['id'] > 1]  # через функцию df.apply() дождаться результата нереально, пришлось колхозить

        # vas_id_day это период между первым и последним предложениями, признак необходим для бизнес-логики
        X['vas_id_day'] = 0
        for i in few.index:
            ix = X.loc[(X['id'] == i)].sort_values(by='buy_time_x', ascending=True)
            for k in range(1, ix.shape[0]):
                df1 = X.loc[ix.index[k - 1], 'date']  # предыдущее предложение
                df2 = X.loc[ix.index[k], 'date']  # первое предложение
                X.loc[ix.index[k], 'vas_id_day'] = (
                            pd.to_datetime(df2) - pd.to_datetime(df1)).days  # + 1 этого не было, только здесь добавила
                if (k == 2) & ((pd.to_datetime(df2) - pd.to_datetime(df1)).days == 0):
                    X.loc[ix.index[k], 'vas_id_day'] = X.loc[ix.index[k - 1], 'vas_id_day']

        """Обучаем Catboost"""
        X['is_action'] = 0
        X.loc[((X['day'] > start) & (X['day'] < stop) & (y == 1)), 'is_action'] = 1
        X_action = X.drop(columns=['is_action'])
        y_action = X['is_action']
        disbalance = y_action.value_counts()[0] / y_action.value_counts()[1]
        frozen_params = {
            'class_weights': [1, disbalance],
            'silent': True,
            'random_state': 21,
            'eval_metric': 'F1',
            'early_stopping_rounds': 60
        }
        self.action_model = catb.CatBoostClassifier(**frozen_params)
        self.action_model.fit(X_action, y_action)
        y_pred = self.action_model.predict(X)
        X['is_action'] = y_pred
        X.drop(columns=['date'], inplace=True)
        train_df = pd.merge(X, y, left_index=True, right_index=True)

        return train_df

    def transform(self, df_all):
        from datetime import datetime, date, time
        from datetime import timedelta
        import holidays
        from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
        X = df_all

        X = X.reset_index().merge(self.log_df, on=['id', 'vas_id'], how='left').set_index('index').fillna(0)
        list_id = list(X.id.unique())
        for i in list_id:
            ix = X.loc[X['id'] == i, 'vas_id'].value_counts()
            for k in ix.index:
                X.loc[X['id'] == i, 'vas_id_0' + str(int(k))] += ix[k]

        # Добываю максимум информации из 'buy_time_x', остальные признаки, включая выходные и праздники
        # к увеличению метрик не привели
        X['date'] = list(map(datetime.fromtimestamp, X['buy_time_x']))
        X['month'] = X['date'].apply(lambda x: x.timetuple()[1])
        X['day'] = X['date'].apply(lambda x: x.timetuple()[7])
        X['weekofyear'] = X['buy_time_x'].apply(lambda x: pd.to_datetime(date.fromtimestamp(x)).weekofyear)

        X['time_max'] = X.buy_time_x.max()
        X['novelty'] = X['time_max'] - X['buy_time_x']

        few = pd.DataFrame(X['id'].value_counts())
        few = few.loc[few['id'] > 1]  # через функцию df.apply() дождаться результата нереально, пришлось колхозить

        # vas_id_day это период между первым и последним предложениями, признак необходим для бизнес-логики
        X['vas_id_day'] = 0
        for i in few.index:
            ix = X.loc[(X['id'] == i)].sort_values(by='buy_time_x', ascending=True)
            for k in range(1, ix.shape[0]):
                df1 = X.loc[ix.index[k - 1], 'date']  # предыдущее предложение
                df2 = X.loc[ix.index[k], 'date']  # первое предложение
                X.loc[ix.index[k], 'vas_id_day'] = (
                            pd.to_datetime(df2) - pd.to_datetime(df1)).days  # + 1 этого не было, только здесь добавила
                if (k == 2) & ((pd.to_datetime(df2) - pd.to_datetime(df1)).days == 0):
                    X.loc[ix.index[k], 'vas_id_day'] = X.loc[ix.index[k - 1], 'vas_id_day']

        y_pred = self.action_model.predict(X)
        X['is_action'] = y_pred
        X.drop(columns=['date'], inplace=True)
        X = X.fillna(0)

        return X


class Estimator():
    def __init__(self):
        self.TARGET_NAME = 'target'
        self.columns1 = ['id', 'vas_id', 'buy_time_x', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                         '10', '11', '12', '13', '14', '18', '19', '20', '21', '25', '26', '28', '30',
                         '34', '36', '37', '38', '39', '40', '41', '43', '44', '45', '46', '47', '48',
                         '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61',
                         '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '73', '74', '76',
                         '77', '92', '96', '97', '98', '99', '100', '101', '102', '103', '104', '105',
                         '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116',
                         '117', '121', '123', '124', '125', '126', '127', '128', '129', '130', '131',
                         '132', '133', '134', '135', '136', '137', '138', '140', '141', '142', '143',
                         '144', '145', '146', '147', '148', '149', '150', '151', '152', '156', '157',
                         '158', '159', '160', '161', '162', '164', '165', '166', '167', '168', '169',
                         '170', '171', '172', '174', '175', '176', '181', '182', '183', '184', '185',
                         '186', '187', '188', '189', '190', '191', '192', '193', '195', '196', '198',
                         '200', '201', '202', '204', '205', '207', '208', '209', '210', '211', '212',
                         '213', '214', '215', '220', '222', '223', '224', '225', '226', '227', '228',
                         '229', '230', '231', '233', '234', '235', '236', '237', '238', '239', '240',
                         '241', '242', '243', '244', '245', '246', '247', '248', '249', '250', '251',
                         '252', 'time_delta', 'vas_id_01', 'weekofyear', 'vas_id_day', 'is_action']
        self.classifier1 = None
        self.columns2 = ['id', 'vas_id', 'buy_time_x', '0', '1', '2', '3', '4', '5', '7', '8',
                         '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21',
                         '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
                         '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47',
                         '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60',
                         '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73',
                         '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86',
                         '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99',
                         '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110',
                         '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121',
                         '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132',
                         '133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143',
                         '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154',
                         '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165',
                         '166', '167', '168', '169', '170', '171', '172', '173', '174', '175', '176',
                         '177', '178', '179', '180', '181', '182', '183', '184', '185', '186', '187',
                         '188', '189', '190', '191', '192', '193', '194', '195', '196', '197', '198',
                         '199', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209',
                         '210', '211', '212', '213', '214', '215', '216', '217', '218', '219', '220',
                         '221', '222', '223', '224', '225', '226', '227', '228', '229', '230', '231',
                         '232', '233', '234', '235', '236', '237', '238', '239', '240', '241', '242',
                         '243', '244', '245', '246', '247', '248', '249', '250', '251', '252', 'time_delta',
                         'vas_id_01', 'vas_id_02', 'vas_id_04', 'vas_id_05', 'vas_id_06', 'vas_id_07',
                         'vas_id_08', 'vas_id_09', 'month', 'day', 'weekofyear', 'time_max', 'novelty',
                         'vas_id_day', 'is_action']
        self.classifier2 = None
        self.columns4 = ['id', 'vas_id', 'buy_time_x', '0', '2', '4', '6', '8', '10', '11', '12', '14', '18',
                         '19', '20', '28', '30', '36', '38', '39', '40', '43', '44', '45', '50',
                         '51', '54', '59', '60', '61', '62', '63', '64', '66', '67', '69', '70',
                         '72', '73', '76', '82', '86', '110', '111', '112', '113', '114', '115', '116',
                         '126', '127', '128', '131', '132', '133', '135', '136', '137', '138',
                         '140', '143', '147', '148', '149', '150', '151', '152', '153', '156', '158',
                         '159', '160', '162', '164', '173', '176', '186', '188', '190', '192', '193',
                         '194', '195', '196', '198', '200', '205', '220', '233', '249', '251', '252',
                         'vas_id_04', 'weekofyear', 'novelty', 'vas_id_day', 'is_action']
        self.classifier4 = None
        self.columns5 = ['id', 'vas_id', 'buy_time_x', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                         '10', '11', '12', '13', '14', '18', '19', '20', '21', '22', '25', '26', '27',
                         '28', '29', '30', '34', '36', '37', '38', '39', '40', '41', '42', '43', '44',
                         '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56',
                         '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70',
                         '71', '72', '73', '74', '76', '77', '78', '79', '80', '82', '83', '86', '87',
                         '89', '90', '91', '92', '94', '96', '97', '98', '99', '100', '101', '102', '103',
                         '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114',
                         '115', '116', '117', '118', '119', '120', '121', '123', '124', '125', '126',
                         '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137',
                         '138', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150',
                         '151', '152', '153', '155', '156', '157', '158', '159', '160', '161', '162', '164',
                         '165', '166', '167', '168', '169', '170', '171', '172', '174', '175', '176',
                         '178', '179', '180', '181', '182', '183', '184', '185', '186', '187', '188',
                         '189', '190', '191', '192', '193', '194', '195', '196', '198', '199', '200',
                         '201', '202', '204', '205', '206', '207', '208', '209', '210', '211', '212',
                         '213', '214', '215', '217', '219', '220', '222', '223', '224', '225', '226',
                         '227', '228', '229', '230', '231', '232', '233', '234', '235', '236', '237',
                         '238', '239', '240', '241', '242', '243', '244', '245', '246', '247', '248',
                         '249', '250', '251', '252', 'time_delta', 'vas_id_05', 'month', 'day',
                         'weekofyear', 'novelty', 'vas_id_day', 'is_action']
        self.classifier5 = None
        self.columns6 = ['id', 'vas_id', 'buy_time_x', '0', '2', '4', '6', '8', '10', '11', '12', '14', '17', '18',
                         '19', '20', '22', '26', '27', '28', '29', '30', '31', '35', '38', '39', '40',
                         '43', '44', '45', '50', '51', '54', '60', '61', '62', '63', '64', '65', '66',
                         '69', '70', '71', '72', '73', '83', '86', '87', '88', '96', '101', '105', '110',
                         '111', '112', '113', '114', '115', '116', '123', '124', '125', '126', '127',
                         '128', '129', '132', '133', '135', '136', '137', '138', '141', '142', '143',
                         '148', '149', '150', '151', '152', '153', '156', '157', '159', '160', '161',
                         '162', '163', '175', '176', '177', '178', '186', '188', '189', '190', '192',
                         '194', '195', '196', '198', '200', '201', '202', '204', '205', '206', '209',
                         '214', '217', '219', '220', '252', 'vas_id_06', 'month', 'day', 'novelty',
                         'vas_id_day', 'is_action']
        self.classifier6 = None
        self.columns7 = ['id', 'vas_id', 'buy_time_x', '0', '2', '4', '6', '8', '10', '11', '12',
                         '14', '15', '16', '17', '18', '19', '20', '23', '24', '26', '27', '28', '29',
                         '31', '32', '33', '35', '36', '38', '39', '40', '43', '44', '45', '46', '50',
                         '51', '54', '57', '60', '61', '62', '63', '64', '65', '66', '67', '69', '70',
                         '71', '72', '73', '78', '79', '80', '83', '84', '85', '86', '87', '88', '91',
                         '92', '93', '110', '112', '113', '116', '118', '119', '121', '122', '123',
                         '124', '125', '126', '128', '131', '133', '135', '136', '137', '143', '147',
                         '148', '149', '150', '153', '154', '155', '159', '161', '162', '164', '167',
                         '170', '173', '174', '176', '177', '178', '179', '180', '182', '184', '185',
                         '186', '188', '192', '193', '194', '195', '196', '197', '198', '199', '200',
                         '201', '202', '204', '205', '206', '209', '212', '214', '217', '218', '219',
                         '220', '221', '252', 'vas_id_07', 'day', 'novelty', 'vas_id_day', 'is_action', ]
        self.classifier7 = None
        self.columns8 = ['id', 'vas_id', 'buy_time_x', '0', '1', '2', '3', '4', '5', '6', '7',
                         '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                         '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33',
                         '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46',
                         '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59',
                         '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72',
                         '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85',
                         '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98',
                         '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
                         '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120',
                         '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131',
                         '132', '133', '134', '135', '136', '137', '138', '139', '140', '141', '142',
                         '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153',
                         '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164',
                         '165', '166', '167', '168', '169', '170', '171', '172', '173', '174', '175',
                         '176', '177', '178', '179', '180', '181', '182', '183', '184', '185', '186',
                         '187', '188', '189', '190', '191', '192', '193', '194', '195', '196', '197',
                         '198', '199', '200', '201', '202', '203', '204', '205', '206', '207', '208',
                         '209', '210', '211', '212', '213', '214', '215', '216', '217', '218', '219',
                         '220', '221', '222', '223', '224', '225', '226', '227', '228', '229', '230',
                         '231', '232', '233', '234', '235', '236', '237', '238', '239', '240', '241',
                         '242', '243', '244', '245', '246', '247', '248', '249', '250', '251', '252',
                         'time_delta', 'vas_id_01', 'vas_id_02', 'vas_id_04', 'vas_id_05', 'vas_id_06',
                         'vas_id_07', 'vas_id_08', 'vas_id_09', 'month', 'day', 'weekofyear', 'time_max',
                         'novelty', 'vas_id_day', 'is_action']
        self.classifier8 = None
        self.columns9 = ['id', 'vas_id', 'buy_time_x', '0', '2', '4', '6', '8', '10', '11', '12', '14', '18', '19',
                         '20', '26', '27', '28', '29', '30', '36', '38', '39', '40', '43', '44', '45',
                         '50', '51', '54', '61', '62', '63', '64', '65', '66', '67', '71', '72', '73',
                         '78', '79', '80', '82', '90', '91', '92', '93', '101', '102', '110', '112', '113',
                         '114', '115', '116', '118', '123', '124', '125', '126', '127', '128', '131', '132',
                         '133', '135', '136', '137', '138', '140', '142', '143', '148', '149', '150',
                         '151', '152', '156', '157', '159', '162', '163', '164', '173', '174', '175',
                         '178', '186', '188', '193', '195', '196', '198', '199', '200', '201', '204',
                         '209', '214', '217', '219', '231', '252', 'month', 'day', 'novelty', 'vas_id_day',
                         'is_action']
        self.classifier9 = None

    def fit(self, train_df):
        import pickle
        X_train = train_df.drop(columns=[self.TARGET_NAME])
        y_train = train_df[self.TARGET_NAME]

        train_df_1 = train_df[train_df['vas_id'] == 1]
        X_train1 = train_df_1.drop(columns=['target'])
        y_train1 = train_df_1['target']
        df_for_balancing = pd.concat([X_train1, y_train1], axis=1)
        df_balanced = self.balance_df_by_target(df_for_balancing, self.TARGET_NAME)
        df_balanced[self.TARGET_NAME].value_counts()
        X_train1 = df_balanced.drop(columns=self.TARGET_NAME)
        y_train1 = df_balanced[self.TARGET_NAME]
        with open('models/model1.pkl', 'rb') as model:
            self.classifier1 = pickle.load(model)
        self.classifier1.fit(X_train1[self.columns1], y_train1)
        train_df_2 = train_df[train_df['vas_id'] == 2]
        X_train2 = train_df_2.drop(columns=['target'])
        y_train2 = train_df_2['target']
        df_for_balancing = pd.concat([X_train2, y_train2], axis=1)
        df_balanced = self.balance_df_by_target(df_for_balancing, self.TARGET_NAME)
        df_balanced[self.TARGET_NAME].value_counts()
        X_train2 = df_balanced.drop(columns=self.TARGET_NAME)
        y_train2 = df_balanced[self.TARGET_NAME]
        with open('models/model2.pkl', 'rb') as model:
            self.classifier2 = pickle.load(model)
        self.classifier2.fit(X_train2[self.columns2], y_train2)

        train_df_4 = train_df[train_df['vas_id'] == 4]
        X_train4 = train_df_4.drop(columns=['target'])
        y_train4 = train_df_4['target']
        df_for_balancing = pd.concat([X_train4, y_train4], axis=1)
        df_balanced = self.balance_df_by_target(df_for_balancing, self.TARGET_NAME)
        df_balanced[self.TARGET_NAME].value_counts()
        X_train4 = df_balanced.drop(columns=self.TARGET_NAME)
        y_train4 = df_balanced[self.TARGET_NAME]
        with open('models/model4.pkl', 'rb') as model:
            self.classifier4 = pickle.load(model)
        self.classifier4.fit(X_train4[self.columns4], y_train4)

        train_df_5 = train_df[train_df['vas_id'] == 5]
        X_train5 = train_df_5.drop(columns=['target'])
        y_train5 = train_df_5['target']
        with open('models/model5.pkl', 'rb') as model:
            self.classifier5 = pickle.load(model)
        self.classifier5.fit(X_train5[self.columns5], y_train5)

        train_df_6 = train_df[train_df['vas_id'] == 6]
        X_train6 = train_df_6.drop(columns=['target'])
        y_train6 = train_df_6['target']
        with open('models/model6.pkl', 'rb') as model:
            self.classifier6 = pickle.load(model)
        self.classifier6.fit(X_train6[self.columns6], y_train6)

        train_df_7 = train_df[train_df['vas_id'] == 7]
        X_train7 = train_df_7.drop(columns=['target'])
        y_train7 = train_df_7['target']
        df_for_balancing = pd.concat([X_train7, y_train7], axis=1)
        df_balanced = self.balance_df_by_target(df_for_balancing, self.TARGET_NAME)
        df_balanced[self.TARGET_NAME].value_counts()
        X_train7 = df_balanced.drop(columns=self.TARGET_NAME)
        y_train7 = df_balanced[self.TARGET_NAME]
        with open('models/model7.pkl', 'rb') as model:
            self.classifier7 = pickle.load(model)
        self.classifier7.fit(X_train7[self.columns7], y_train7)

        train_df_8 = train_df[train_df['vas_id'] == 8]
        X_train8 = train_df_8.drop(columns=['target'])
        y_train8 = train_df_8['target']
        with open('models/model8.pkl', 'rb') as model:
            self.classifier8 = pickle.load(model)
        self.classifier8.fit(X_train8[self.columns8], y_train8)

        train_df_9 = train_df[train_df['vas_id'] == 9]
        X_train9 = train_df_9.drop(columns=['target'])
        y_train9 = train_df_9['target']
        with open('models/model9.pkl', 'rb') as model:
            self.classifier9 = pickle.load(model)
        self.classifier9.fit(X_train9[self.columns9], y_train9)

        return self

    def predict(self, test_df):
        X_test = test_df

        test_df_1 = test_df[test_df['vas_id'] == 1]
        X_test1 = test_df_1
        y_pred1 = self.classifier1.predict(X_test1[self.columns1])
        X_test1['y_pred'] = y_pred1

        test_df_2 = test_df[test_df['vas_id'] == 2]
        X_test2 = test_df_2
        y_pred2 = self.classifier2.predict(X_test2[self.columns2])
        X_test2['y_pred'] = y_pred2

        test_df_4 = test_df[test_df['vas_id'] == 4]
        X_test4 = test_df_4
        y_pred4 = self.classifier4.predict(X_test4[self.columns4])
        X_test4['y_pred'] = y_pred4

        test_df_5 = test_df[test_df['vas_id'] == 5]
        X_test5 = test_df_5
        y_pred5 = self.classifier5.predict(X_test5[self.columns5])
        X_test5['y_pred'] = y_pred5

        test_df_6 = test_df[test_df['vas_id'] == 6]
        X_test6 = test_df_6
        y_pred6 = self.classifier6.predict(X_test6[self.columns6])
        X_test6['y_pred'] = y_pred6

        test_df_7 = test_df[test_df['vas_id'] == 7]
        X_test7 = test_df_7
        y_pred7 = self.classifier7.predict(X_test7[self.columns7])
        X_test7['y_pred'] = y_pred7

        test_df_8 = test_df[test_df['vas_id'] == 8]
        X_test8 = test_df_8
        y_pred8 = self.classifier8.predict(X_test8[self.columns8])
        X_test8['y_pred'] = y_pred8

        test_df_9 = test_df[test_df['vas_id'] == 9]
        X_test9 = test_df_9
        y_pred9 = self.classifier9.predict(X_test9[self.columns9])
        X_test9['y_pred'] = y_pred9

        y_pred = pd.DataFrame(pd.concat([X_test1['y_pred'], X_test2['y_pred'], X_test4['y_pred'],
                                         X_test5['y_pred'], X_test6['y_pred'], X_test7['y_pred'],
                                         X_test8['y_pred'], X_test9['y_pred']]))
        predicted = pd.merge(X_test, y_pred, left_index=True, right_index=True, how='left')[
            ['id', 'vas_id', 'buy_time_x', 'y_pred']]
        predicted = predicted.rename(columns={'y_pred': 'predict', 'buy_time_x': 'buy_time'})

        return predicted


    def balance_df_by_target(self, df, target_name):
        target_counts = df[target_name].value_counts()
        major_class_name = target_counts.argmax()
        minor_class_name = target_counts.argmin()
        disbalance_coeff = int(target_counts[major_class_name] / target_counts[minor_class_name]) - 1
        for i in range(disbalance_coeff):
            sample = df[df[target_name] == minor_class_name].sample(target_counts[minor_class_name])
            df = df.append(sample, ignore_index=True)
        return df.sample(frac=1)

if __name__ == "__main__":
    data_train = pd.read_csv('data/data_train.csv')
    features = dd.read_csv('data/features.csv', blocksize=25e6, sep='\t')
    features_train = features.loc[features['id'].isin(data_train['id'])].compute()

    data_test = pd.read_csv('data/data_test.csv')
    features_test = features.loc[features['id'].isin(data_test['id'])].compute()

    datapreprocessing = DataPreprocessing()
    df_all = datapreprocessing.transform(data_train, features_train)
    df_all = reduce_mem_usage(df_all)
    df_all_test = datapreprocessing.transform(data_test, features_train)
    df_all_test = reduce_mem_usage(df_all_test)
    df_all.to_csv('data/df_all.csv', index=False)
    df_all_test.to_csv('data/df_all_test.csv', index=False)
    data_train = None
    data_test = None
    features = None
    features_train = None

    featuregenerator = Featuregenerator()
    train_df = featuregenerator.fit(df_all, 316, 337)
    test_df = featuregenerator.transform(df_all_test)
    train_df.to_csv('data/train_df.csv', index=False)
    test_df.to_csv('data/test_df.csv', index=False)

    estimator = Estimator()
    estimator.fit(train_df)
    predicted = estimator.predict(test_df)
    print(predicted.head())

    predicted.to_csv('data/answers_test.csv', index=False)





