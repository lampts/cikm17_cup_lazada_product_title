import os, sys, gc

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import GridSearchCV
from feature_management import FeatureManagement
from data_preparation import DataPreparation
from sklearn.calibration import CalibratedClassifierCV

class XGBoostAlgo(object):

    '''
    get the current working directory
    '''
    def __init__(self, max_depth, learning_rate):
        self.HOME_DIR = os.path.dirname(os.path.abspath(__file__))
        self.MAX_DEPTH = max_depth
        self.LEARNING_RATE = learning_rate

    '''
    this method reads input data from files into Pandas data frames
    '''
    def get_input_data(self, features, target_column, label):
        selected_columns = list(features)
        selected_columns.append("label")
        if target_column == "both":
            selected_columns.append("clarity")
            selected_columns.append("conciseness")
        else:
            selected_columns.append(target_column)

        extra_feature_list = ["ct_ratio", "title_num_stopspecialwords", "desc_num_stopspecialwords"]
        for column in extra_feature_list:
            if column in selected_columns:
                selected_columns.remove(column)

        if target_column == "conciseness" or target_column == "both":
            filename = os.path.join(self.HOME_DIR, "input", "data_ready_conciseness.csv")
        else:
            filename = os.path.join(self.HOME_DIR, "input", "data_ready_clarity.csv")
        df_input = pd.read_csv(filename, usecols=selected_columns)
        print("df_input shape {}".format(df_input.shape))

        #calculate extra features if needed
        if "ct_ratio" in features:
            df_input["ct_ratio"] = df_input["ct_common"] / df_input["title_num_words"]
        if "title_num_stopspecialwords" in features:
            df_input["title_num_stopspecialwords"] = df_input["title_num_stopwords"] + df_input["title_num_specialwords"]
        if "desc_num_stopspecialwords" in features:
            df_input["desc_num_stopspecialwords"] = df_input["desc_num_stopwords"] + df_input["desc_num_specialwords"]

        if target_column == "both":
            df_input["both"] = df_input["clarity"] + df_input["conciseness"]

        df_train = df_input[(df_input["label"] == 0)]
        df_test = df_input[(df_input["label"] == label)]

        #include clarity in df_test if needed
        if "clarity" in features:
            clarity_filename = os.path.join(self.HOME_DIR, "output", "clarity_valid.predict")
            df_test["clarity"] = np.loadtxt(clarity_filename)

        print("Total number of features is {}".format(len(df_train.columns)))
        print("Total data in df_train, df_test are: {},{}".format(len(df_train), len(df_test)))
        return df_train, df_test

    '''
    use GridSearchCV to find optimal values for parameters in a grid
    '''

    def tune_parameters(self, df_train, features):
        grid_params = {
            'subsample':[0.55, 0.65, 0.75],
            'colsample_bytree':[0.25, 0.30, 0.35, 0.40],
            'colsample_bylevel':[0.4, 0.5, 0.6, 0.7]
        }

        grid_search = GridSearchCV(estimator=xgb.XGBClassifier(max_depth=self.MAX_DEPTH, \
            learning_rate=self.LEARNING_RATE, n_estimators=400, silent=True, \
            objective='binary:logistic', nthread=8, gamma=0.1, min_child_weight=1, \
            max_delta_step=0, subsample=0.75, colsample_bytree=0.25, colsample_bylevel=0.5, \
            reg_alpha=0.1, reg_lambda=1, scale_pos_weight=1, \
            base_score=0.685, seed=2017, missing=None),
            param_grid=grid_params, scoring='neg_mean_squared_error', iid=False, verbose=2, cv=5)

        grid_search.fit(df_train[features], df_train["conciseness"])
        print("Done fitting")
        print(grid_search.grid_scores_)
        print(grid_search.best_params_)
        print(grid_search.best_score_)

    '''
    select indices for training models
    '''
    def select_index(self, model, value):
        if model == value:
            return False
        else:
            return True

    '''
    train an xgboost model using sklearn API
    '''
    def train_clarity(self, df_train, features, num_of_models, num_of_rounds, is_calibration = False):
        if is_calibration:
            print("Calibration")
            clarity_model = xgb.XGBClassifier(max_depth=self.MAX_DEPTH, learning_rate=self.LEARNING_RATE, \
                                              n_estimators=num_of_rounds, silent=True, objective='binary:logistic', \
                                              gamma=0.1, min_child_weight=1, max_delta_step=0, \
                                              subsample=0.75, colsample_bytree=0.3, colsample_bylevel=0.5, \
                                              reg_alpha=0.1, reg_lambda=1, scale_pos_weight=1, \
                                              base_score=0.685, seed=2017, missing=None)

            clarity_model_calibrated = CalibratedClassifierCV(clarity_model, cv=10, method='isotonic')
            clarity_model_calibrated.fit(df_train[features], df_train["clarity"])
            return clarity_model_calibrated

        print("No calibration")
        clarity_models = []

        # create model indices and probablitity distribution
        model_indices = [0] * (num_of_models + 1)
        model_probabilities = [0.0] * (num_of_models + 1)
        leftout_percentage = 1.0 / num_of_models
        for i in range(num_of_models):
            model_indices[i] = i
            model_probabilities[i] = leftout_percentage
        model_indices[num_of_models] = num_of_models
        model_probabilities[num_of_models] = max(0, 1.0 - leftout_percentage * num_of_models)
        print(model_indices)
        print(model_probabilities)

        random_index = np.random.choice(model_indices, len(df_train), p=model_probabilities)
        for model_index in range(num_of_models):
            print("Training model %d" % model_index)

            train_index = [self.select_index(model_index, i) for i in random_index]
            watch_index = [not i for i in train_index]

            if num_of_models != 1:
                x_train = df_train[train_index]
                x_watch = df_train[watch_index]
            else:
                x_train = df_train
                x_watch = df_train

            clarity_model = xgb.XGBClassifier(max_depth=self.MAX_DEPTH, learning_rate=self.LEARNING_RATE, \
                                              n_estimators=num_of_rounds, silent=True, objective='binary:logistic', \
                                              gamma=0.1, min_child_weight=1, max_delta_step=0, \
                                              subsample=0.75, colsample_bytree=0.3, colsample_bylevel=0.5, \
                                              reg_alpha=0.1, reg_lambda=1, scale_pos_weight=1, \
                                              base_score=0.943, seed=2017, missing=None)

            clarity_model.fit(x_train[features], x_train["clarity"], \
                              eval_set=[(x_train[features], x_train["clarity"]),
                                        (x_watch[features], x_watch["clarity"])], \
                              eval_metric="rmse", early_stopping_rounds=50)

            clarity_models.append(clarity_model)

        return clarity_models

    '''
    update cat1, 2, 3 values with target encoding
    '''
    def get_target_encoding_values(self, df_input, col_name, target_name, alpha=5, output_column='y_mean_smooth'):
        target_global = df_input[target_name].mean()
        df_count = df_input.groupby([col_name])[target_name].count()
        df_count = df_count.reset_index()
        df_mean = df_input.groupby([col_name])[target_name].mean()
        df_mean = df_mean.reset_index()
        df_count.columns = [col_name, 'nb_row']
        df_mean.columns = [col_name, 'y_mean']
        df_ll = pd.merge(df_count, df_mean, on=col_name)
        df_ll[output_column] = df_ll.apply(lambda r: 1. * (r['nb_row'] * r['y_mean'] + alpha * target_global) / (r['nb_row'] + alpha), axis=1)
        encoding_values = dict(zip(df_ll[col_name], df_ll[output_column]))

        return encoding_values

    '''
    train an xgboost model using sklearn API
    '''
    def train_concise(self, df_train, features, num_of_models, num_of_rounds, is_target_encoding=False, is_calibration=False):
        if is_calibration:
            print("Calibration")
            concise_model = xgb.XGBClassifier(max_depth=self.MAX_DEPTH, learning_rate=self.LEARNING_RATE, \
                                              n_estimators=num_of_rounds, silent=True, objective='binary:logistic', \
                                              gamma=0.1, min_child_weight=1, max_delta_step=0, \
                                              subsample=0.75, colsample_bytree=0.3, colsample_bylevel=0.5, \
                                              reg_alpha=0.1, reg_lambda=1, scale_pos_weight=1, \
                                              base_score=0.685, seed=2017, missing=None)

            concise_model_calibrated = CalibratedClassifierCV(concise_model, cv=num_of_models, method='isotonic')
            concise_model_calibrated.fit(df_train[features], df_train["conciseness"])
            return concise_model_calibrated, []

        print("No calibration")
        concise_models = []
        target_encoding_values = []

        # create model indices and probablitity distribution
        model_indices = [0] * (num_of_models + 1)
        model_probabilities = [0.0] * (num_of_models + 1)
        leftout_percentage = 1.0 / num_of_models
        for i in range(num_of_models):
            model_indices[i] = i
            model_probabilities[i] = leftout_percentage
        model_indices[num_of_models] = num_of_models
        model_probabilities[num_of_models] = max(0, 1.0 -leftout_percentage * num_of_models)
        print(model_indices)
        print(model_probabilities)

        random_index = np.random.choice(model_indices, len(df_train), p=model_probabilities)
        for model_index in range(num_of_models):
            print("Training model %d" % model_index)

            train_index = [self.select_index(model_index, i) for i in random_index]
            watch_index = [not i for i in train_index]

            if num_of_models != 1:
                x_train = df_train[train_index]
                x_watch = df_train[watch_index]
            else:
                x_train = df_train
                x_watch = df_train

            if is_target_encoding:
                encoding_values = self.get_target_encoding_values(x_train, "cat1", "conciseness")
                target_encoding_values.append(encoding_values)

                for j in range(57):
                    x_train.loc[(x_train["cat2"] == j), "cat2"] = encoding_values.get(j)
                    x_watch.loc[(x_watch["cat2"] == j), "cat2"] = encoding_values.get(j)

            concise_model = xgb.XGBClassifier(max_depth=self.MAX_DEPTH, learning_rate=self.LEARNING_RATE, \
                n_estimators=num_of_rounds, silent=True, objective='binary:logistic', \
                gamma=0.1, min_child_weight=1, max_delta_step=0, \
                subsample=0.75, colsample_bytree=0.3, colsample_bylevel=0.5, \
                reg_alpha=0.1, reg_lambda=1, scale_pos_weight=1, \
                base_score=0.685, seed=2017, missing=None)

            concise_model.fit(x_train[features], x_train["conciseness"], \
                eval_set=[(x_train[features], x_train["conciseness"]), (x_watch[features], x_watch["conciseness"])], \
                eval_metric="rmse", early_stopping_rounds=50)

            concise_models.append(concise_model)

        return concise_models, target_encoding_values

    '''
    generate predictions for models created from sklearn API
    '''
    def predict_clarity(self, df_input, clarity_models, features, num_of_models, is_calibration=False):

        if is_calibration:
            clarity_predictions = np.array(concise_models.predict_proba(df_test[features])[:, 1])

        else:
            clarity_predictions = np.zeros(len(df_input))
            for i in range(num_of_models):
                model_predictions = np.array(clarity_models[i].predict_proba(df_input[features])[:,1])
                clarity_predictions = clarity_predictions + model_predictions

            clarity_predictions = clarity_predictions / num_of_models

        clarity_filename = os.path.join(self.HOME_DIR, "output", "clarity_test.predict")
        np.savetxt(clarity_filename, clarity_predictions, fmt='%1.10f', delimiter="\n")

    '''
    generate predictions for models created from sklearn API
    '''
    def predict_concise(self, df_test, concise_models, features, num_of_models, target_encoding_values=[], is_calibration=False):

        if is_calibration:
            concise_predictions = np.array(concise_models.predict_proba(df_test[features])[:, 1])

        else:
            concise_predictions = np.zeros(len(df_test))
            if len(target_encoding_values) == 0:
                for i in range(num_of_models):
                    model_predictions = np.array(concise_models[i].predict_proba(df_test[features])[:, 1])
                    concise_predictions = concise_predictions + model_predictions
            else:
                for i in range(num_of_models):
                    encoding_values = target_encoding_values[i]
                    for j in range(57):
                        df_test.loc[(df_test["cat2"] == j), "cat2"] = encoding_values.get(j)
                    model_predictions = np.array(concise_models[i].predict_proba(df_test[features])[:, 1])
                    concise_predictions = concise_predictions + model_predictions

            concise_predictions = concise_predictions / num_of_models

        concise_filename = os.path.join(self.HOME_DIR, "output", "conciseness_test.predict")
        np.savetxt(concise_filename, concise_predictions, fmt='%1.10f', delimiter="\n")

    '''
    train with sum of clarity and conciseness as target
    '''
    def train_both(self, df_train, features, num_of_models, num_of_rounds, is_calibration=False):
        if is_calibration:
            print("Calibration")
            both_model = xgb.XGBRegressor(max_depth=self.MAX_DEPTH, learning_rate=self.LEARNING_RATE, \
                            n_estimators=num_of_rounds, silent=True, objective='reg:linear', \
                            gamma=0.1, min_child_weight=1, max_delta_step=0, \
                            subsample=0.75, colsample_bytree=0.3, colsample_bylevel=0.5, \
                            reg_alpha=0.1, reg_lambda=1, scale_pos_weight=1, \
                            base_score=0.685, seed=2017, missing=None)

            both_model_calibrated = CalibratedClassifierCV(both_model, cv=num_of_models, method='isotonic')
            both_model_calibrated.fit(df_train[features], df_train["both"])
            return both_model_calibrated

        print("No calibration")
        both_models = []

        mean_value = df_train["both"].mean()

        # create model indices and probablitity distribution
        model_indices = [0] * (num_of_models + 1)
        model_probabilities = [0.0] * (num_of_models + 1)
        leftout_percentage = 1.0 / num_of_models
        for i in range(num_of_models):
            model_indices[i] = i
            model_probabilities[i] = leftout_percentage
        model_indices[num_of_models] = num_of_models
        model_probabilities[num_of_models] = max(0, 1.0 - leftout_percentage * num_of_models)
        print(model_indices)
        print(model_probabilities)

        random_index = np.random.choice(model_indices, len(df_train), p=model_probabilities)
        for model_index in range(num_of_models):
            print("Training model %d" % model_index)

            train_index = [self.select_index(model_index, i) for i in random_index]
            watch_index = [not i for i in train_index]

            if num_of_models != 1:
                x_train = df_train[train_index]
                x_watch = df_train[watch_index]
            else:
                x_train = df_train
                x_watch = df_train

            both_model = xgb.XGBRegressor(max_depth=self.MAX_DEPTH, learning_rate=self.LEARNING_RATE, \
                    n_estimators=num_of_rounds, silent=True, objective='reg:linear', \
                    gamma=0.1, min_child_weight=1, max_delta_step=0, \
                    subsample=0.75, colsample_bytree=0.3, colsample_bylevel=0.5, \
                    reg_alpha=0.1, reg_lambda=1, scale_pos_weight=1, \
                    base_score=mean_value, seed=2017, missing=None)

            both_model.fit(x_train[features], x_train["both"], \
                    eval_set=[(x_train[features], x_train["both"]), (x_watch[features], x_watch["both"])], \
                    eval_metric="rmse", early_stopping_rounds=50)

            both_models.append(both_model)

        return both_models

    def predict_both(self, df_test, both_models, features, num_of_models, is_calibration=False):

        if is_calibration:
            both_predictions = np.array(both_models.predict(df_test[features]))

        else:
            both_predictions = np.zeros(len(df_test))
            for i in range(num_of_models):
                model_predictions = np.array(both_models[i].predict(df_test[features]))
                both_predictions = both_predictions + model_predictions

            both_predictions = both_predictions / num_of_models

        both_filename = os.path.join(self.HOME_DIR, "output", "both_test.predict")
        np.savetxt(both_filename, both_predictions, fmt='%1.10f', delimiter="\n")

#=======================================================================================
if __name__ == '__main__':

    dp = DataPreparation()
    dp.build_combination(processing_mode=0, out_filename="data_all_clarity.csv")
    dp.clean_data(target_column="clarity")

    feature_man = FeatureManagement()
    phase = 2
    flags = [False, True]

    if flags[0]:

        features = feature_man.get_basic_features(is_clarity=True) + \
                   feature_man.get_text_features(mode=0, type=0) + \
                   feature_man.get_text_features(mode=0, type=1)
        print("Total number of training features {}".format(len(features)))

        no_models = 20
        no_rounds = 500
        max_depth = 8
        learning_rate = 0.1
        algo = XGBoostAlgo(max_depth=max_depth, learning_rate=learning_rate)

        df_train, df_test = algo.get_input_data(features, "conciseness", label=phase)

        concise_models, target_encoding_values = algo.train_concise(df_train, features, num_of_models=no_models, num_of_rounds=no_rounds, is_target_encoding=False, is_calibration=True)
        algo.predict_concise(df_test, concise_models, features, num_of_models=no_models, target_encoding_values=target_encoding_values, is_calibration=True)

        del df_train
        del df_test
        gc.collect()

    if flags[1]:

        features = feature_man.get_basic_features(is_clarity=True) + \
                   feature_man.get_text_features(mode=0, type=0) + \
                   feature_man.get_text_features(mode=0, type=1)
        print("Total number of clarity training features {}".format(len(features)))

        no_models = 10
        no_rounds = 400
        max_depth = 8
        learning_rate = 0.1
        algo = XGBoostAlgo(max_depth=max_depth, learning_rate=learning_rate)

        df_train, df_test = algo.get_input_data(features, "clarity", label=phase)
        clarity_models = algo.train_clarity(df_train, features, num_of_models=no_models, num_of_rounds=no_rounds, is_calibration=False)
        algo.predict_clarity(df_test, clarity_models, features, num_of_models=no_models, is_calibration=False)

        del df_train
        del df_test
        gc.collect()