import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, matthews_corrcoef

class Preprocessor:
    def __init__(self):
        self.inputter = SimpleImputer(missing_values = np.nan, strategy = 'mean', add_indicator = True)
        self.scaler = MinMaxScaler()
        self.X = None

    def fit_transform(self, df):
        filled_data = df.drop(['In-hospital_death', 'recordid'], axis=1)
        filled_data = self.inputter.fit_transform(filled_data)
        self.scaler.fit(filled_data)
        self.X = self.scaler.transform(filled_data)
        return self.X

    def transform(self, df):
        filled_data = df.drop(['In-hospital_death', 'recordid'], axis=1)
        filled_data = self.inputter.transform(filled_data)
        self.X = self.scaler.transform(filled_data)
        return self.X


class Model:
    def __init__(self):
        self.model = None

    def build_model(self, input_shape):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(units = 64, kernel_initializer = 'glorot_uniform', activation = 'relu', input_shape = input_shape))
        self.model.add(tf.keras.layers.Dense(units = 32, kernel_initializer = 'glorot_uniform', activation = 'relu'))
        self.model.add(tf.keras.layers.Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))

    def compile_model(self, loss):
        if loss == 'lasso':
            loss_fn = tf.keras.losses.MeanAbsoluteError()
        elif loss == 'ridge':
            loss_fn = tf.keras.losses.MeanSquaredError()
        else:
            raise ValueError("Invalid loss function.")
        
        self.model.compile(optimizer = 'adam', loss = loss_fn, metrics = ['accuracy'])

    def train(self, X_train, y_train, epochs, batch_size):
        self.model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size)

    def predict_proba(self, X):
        return self.model.predict(X)


class Pipeline:
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.model = Model()

    def run(self, X, test = False):
        if test == False:
            y = X["In-hospital_death"]
            x = self.preprocessor.fit_transform(X)
            x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0, test_size = 0.2, stratify = y)

            self.model.build_model(input_shape = (x_train.shape[1],))
            self.model.compile_model(loss = 'ridge')  

            self.model.train(x_train, y_train, epochs = 10, batch_size = 32)  

            y_pred_proba = self.model.predict_proba(x_test)
            threshold = 0.5 
            y_pred = np.where(y_pred_proba >= threshold, 1, 0)

            f1 = f1_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)

            return y_pred, threshold, f1, mcc

        else:
            X.drop("recordid", axis = 1, inplace = True)
            nan_transform_path = "nantransform.pickle"
            scale_transform_path = "scale_transform.pickle"
            prediction_boost_path = "prediction_boost.pickle"
            best_threshold_path = "best_threshold.pickle"
            
            with open(nan_transform_path, "rb") as f:
                nan_transform = pickle.load(f)

            with open(scale_transform_path, "rb") as f:
                scale_transform = pickle.load(f)

            with open(prediction_boost_path, "rb") as f:
                predict = pickle.load(f)

            with open(best_threshold_path, "rb") as f:
                threshold = pickle.load(f)

            new_data = nan_transform.transform(X)
            new_data = scale_transform.transform(new_data)
            predict_prob = predict.predict_proba(new_data)
            predictions = np.where(predict_prob >= threshold, 1, 0)

            return predictions