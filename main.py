from datetime import datetime

import classifier as classifier
import pandas as pd
import xgboost
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# from DataFrameSelector import DataFrameSelector

data = pd.read_csv("./datasets/diabetes.csv")
print(data.head(5))
# check if any null value is present
print(data.isnull().values.any())

# Train test split

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score

feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                   'DiabetesPedigreeFunction', 'Age']
predicted_class = ['Outcome']

x = data[feature_columns].values
y = data[predicted_class].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=10)

from sklearn.impute import SimpleImputer as Imputer

fill_values = Imputer(missing_values=0, strategy="mean")
X_train = fill_values.fit_transform(X_train)
X_test = fill_values.transform(X_test)

# Apply Algorithm
from sklearn.ensemble import RandomForestClassifier

random_forest_model = RandomForestClassifier(random_state=10)
random_forest_model.fit(X_train, y_train.ravel())

predict_train_data = random_forest_model.predict(X_test)
from sklearn import metrics

print("Accuracy = {0:.3f}".format(metrics.accuracy_score(y_test, predict_train_data)))

# HyperParameter Optimization

params = {
    "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
    "cols-ample_btree": [0.3, 0.4, 0.5, 0.7]
}


# HyperParameter optimization using RandomizedSearchCv
xgb_c1 = xgboost.XGBClassifier()

random_search = RandomizedSearchCV(xgb_c1, param_distributions=params, n_iter=5,
                                   scoring='roc_auc', n_jobs=-1, cv=5, verbose=3)


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3000)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes %i seconds.' % (thour, tmin, round(tsec, 2)))


from datetime import datetime

start_time = timer(None)
random_search.fit(X_train, y_train.ravel())
timer(start_time)

# print(random_search.best_estimator_)

xgb_c1 = xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel1=1,
                               colsample_bytree=0.7, gamma=0.1, learning_rate=0.05,
                               max_delta_step=0, max_depth=5, min_child_weight=7, missing=1,
                               n_estimator=100, n_jobs=1, nthread=None,
                               objective='binary:logistic', random_state=0, reg_alpha=0,
                               reg_lamda=1, scale_pos_weight=1, seed=None, silent=True,
                               subsample=1,
                               eval_metric='mlogloss')

xgb_c1.fit(X_train, y_train.ravel())

y_pred = xgb_c1.predict(X_test)

score = cross_val_score(xgb_c1, X_train, y_train.ravel(), cv=10)
print(score.mean())



