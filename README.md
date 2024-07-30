import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
!gdown https://drive.google.com/uc?id=1deu_mXHCOUJbZc_o5kaYJ8PDb4VXmTOe

!unrar x 'HSE IB - Apartment Prices.rar'
pd.get_dummies(df['Okrug']).columns

df[['ВАО', 'ЗАО', 'САО', 'СВАО', 'СЗАО', 'ЦАО', 'ЮАО', 'ЮВАО', 'ЮЗАО']] = pd.get_dummies(df['Okrug'], dtype=int)

df.head()

df.drop(columns=['Okrug'], inplace=True)

df['District'].value_counts()

pd.get_dummies(df['District']).columns

color_list = pd.get_dummies(df['District'], dtype=int).columns

df[color_list] = pd.get_dummies(df['District'], dtype=int)

df.drop(columns=['District'], inplace=True)
df.drop(columns=['Id', 'Price']).columns

df.columns

df.info()

df.isnull().sum()

columns_list = df.drop(columns=['Id', 'Price']).columns

X = df[columns_list]

Y = df['Price']



X.shape, Y.shape
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y.values, test_size=0.10, random_state=10)

X_train.shape, X_test.shape

X_train

from sklearn.preprocessing import LabelEncoder


labelencoder = LabelEncoder()

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score, mean_squared_error


def all_reg_scores(model, name_model, X_test, Test_y):

    MAE = round(mean_absolute_error(Y_test, model.predict(X_test)), 2)
    MSE = round(mean_squared_error(Y_test, model.predict(X_test)), 2)
    RMSE = round(np.sqrt(mean_squared_error(Y_test, model.predict(X_test))), 2)
    MAPE = round(mean_absolute_percentage_error(Y_test, model.predict(X_test))*100, 4)
    R2 = round(r2_score(Y_test, model.predict(X_test)), 4)

    print(f'{name_model} model: \n', '      r2_score: {0}     MAPE (%): {1}     MAE: {2}     RMSE: {3}     MSE: {4}'.format(R2, MAPE, MAE, RMSE, MSE))

from sklearn.linear_model import LinearRegression

LR = LinearRegression()

LR.fit(X_train, Y_train)

LR.score(X_test, Y_test)
y_pred = LR.predict(X_test)

all_reg_scores(LR, 'LR', X_test, Y_test)
plt.figure(figsize=(17, 3))
plt.plot(y_pred, color='r', label='прогнозные')
plt.plot(Y_test, color='g', label='реальные')
plt.grid()
plt.legend()

plt.figure(figsize=(17, 3))
plt.plot(pd.DataFrame(y_pred).sort_values(0).reset_index().drop('index', axis=1), color='r', label='прогнозные')
plt.plot(pd.DataFrame(Y_test).sort_values(0).reset_index().drop('index', axis=1), color='g', label='реальные')
plt.grid()
plt.legend()

from sklearn.neighbors import KNeighborsRegressor

KNNR = KNeighborsRegressor(n_neighbors=6)

KNNR.fit(X_train, Y_train)

KNNR.score(X_test, Y_test)

all_reg_scores(KNNR, 'KNNR', X_test, Y_test)
from sklearn.tree import DecisionTreeRegressor

DTR = DecisionTreeRegressor(max_depth=12, random_state=10)

DTR.fit(X_train, Y_train)

DTR.score(X_test, Y_test)

all_reg_scores(DTR, 'DTR', X_test, Y_test)
from sklearn.ensemble import BaggingRegressor

BGR = BaggingRegressor(base_estimator=LinearRegression(),
                        n_estimators=10,
                        max_samples=1.0,
                        max_features=1.0,
                        bootstrap=True,
                        bootstrap_features=False,
                        oob_score=False,
                        warm_start=False,
                        n_jobs=None,
                        random_state=10,
                        verbose=0,)

BGR.fit(X_train, Y_train.ravel())
BGR.score(X_test, Y_test.ravel())

all_reg_scores(BGR, 'BGR', X_test, Y_test)
Pred_BGR = BGR.predict(X_test)
from sklearn.ensemble import RandomForestRegressor
RFR = RandomForestRegressor(n_estimators=100,
                            max_depth=None,
                            min_samples_split=2,
                            min_samples_leaf=4,
                            min_weight_fraction_leaf=0.0,
                            max_leaf_nodes=None,
                            min_impurity_decrease=0.0,
                            bootstrap=True,
                            oob_score=False,
                            n_jobs=None,
                            random_state=0,
                            verbose=0,
                            warm_start=False,
                            ccp_alpha=0.0,
                            max_samples=None,)

RFR.fit(X_train, Y_train.ravel())
RFR.score(X_test, Y_test.ravel())

all_reg_scores(RFR, 'RFR', X_test, Y_test)

columns = X.columns

sorted_idx = RFR.feature_importances_.argsort()
plt.barh(columns[sorted_idx], RFR.feature_importances_[sorted_idx])
plt.xlabel("RFR Feature Importance")
from sklearn.ensemble import ExtraTreesRegressor

ExTR = ExtraTreesRegressor(n_estimators=100,
                            max_depth=None,
                            min_samples_split=2,
                            min_samples_leaf=1,
                            min_weight_fraction_leaf=0.0,
                            max_leaf_nodes=None,
                            min_impurity_decrease=0.0,
                            bootstrap=False,
                            oob_score=False,
                            n_jobs=None,
                            random_state=66,
                            verbose=0,
                            warm_start=False,
                            ccp_alpha=0.0,
                            max_samples=None,)

ExTR.fit(X_train, Y_train.ravel())

ExTR.score(X_test, Y_test.ravel())

all_reg_scores(ExTR, 'ExTR', X_test, Y_test)

y_pred_ExTR = ExTR.predict(X_test)

sorted_idx = ExTR.feature_importances_.argsort()
plt.barh(columns[sorted_idx], ExTR.feature_importances_[sorted_idx])
plt.xlabel("ExTR Feature Importance")
from sklearn.ensemble import AdaBoostRegressor

AdBR = AdaBoostRegressor(random_state=0, n_estimators=100)

AdBR.fit(X_train, Y_train.ravel())

AdBR.score(X_test, Y_test.ravel())

all_reg_scores(AdBR, 'AdBR', X_test, Y_test)
from sklearn.ensemble import GradientBoostingRegressor

GBR = GradientBoostingRegressor(
                                learning_rate=0.1,
                                n_estimators=100,
                                subsample=1.0,
                                criterion='friedman_mse',
                                min_samples_split=2,
                                min_samples_leaf=1,
                                min_weight_fraction_leaf=0.0,
                                max_depth=3,
                                min_impurity_decrease=0.0,
                                init=None,
                                random_state=21,
                                max_features=None,
                                alpha=0.9,
                                verbose=0,
                                max_leaf_nodes=None,
                                warm_start=False,
                                validation_fraction=0.1,
                                n_iter_no_change=None,
                                tol=0.0001,
                                ccp_alpha=0.0,)

GBR.fit(X_train, Y_train.ravel())
GBR.score(X_test, Y_test.ravel())

all_reg_scores(GBR, 'GBR', X_test, Y_test)
from xgboost import XGBRegressor

XGBR = XGBRegressor(max_depth=10,
                   learning_rate=0.1,
                   n_estimators=1000,
                   reg_alpha=0.001,
                   reg_lambda=0.000001,
                   n_jobs=-1,
                   min_child_weight=3)

XGBR.fit(X_train, Y_train)

XGBR.score(X_test, Y_test)

all_reg_scores(XGBR, 'XGBR', X_test, Y_test)
import lightgbm as ltb

lgbm = ltb.LGBMRegressor()

#Defining a dictionary containing all the releveant parameters
param_grid = {
    "boosting_type": ['gbdt'],
    "num_leaves": [9, 19],  #[ 19, 31, 37, 47],
    "max_depth": [29], #[7, 15, 29, 37, 47, 53],
    "learning_rate": [0.1, 0.15],
    "n_estimators": [1000], #[500, 1000, 2000],
    "subsample_for_bin": [200000], #[20000, 200000, 2000000],
    "objective": ["regression"],
    "min_child_weight": [0.01], #[0.001, 0.01],
    "min_child_samples":[100, 200], #[20, 50, 100],
    "subsample":[1.0],
    "subsample_freq":[0],
    "colsample_bytree":[1.0],
    "reg_alpha":[0.0],
    "reg_lambda":[0.0]
}

from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn import model_selection

model_lgbm = model_selection.RandomizedSearchCV(
            estimator=lgbm,
            param_distributions=param_grid,
            n_iter=100,
            scoring="neg_root_mean_squared_error",
            verbose=10,
            n_jobs=-1,
            cv=5
        )

model_lgbm.fit(X_train, Y_train)

print(f"Best score: {model_lgbm.best_score_}")
print("Best parameters from the RandomSearchCV:")
best_parameters = model_lgbm.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print(f"\t{param_name}: {best_parameters[param_name]}")

# Get best model
LGBMR = model_lgbm.best_estimator_
Y_pred_lgb = LGBMR.predict(X_test)

r2_score(Y_pred_lgb, Y_test)

all_reg_scores(LGBMR, 'LGBMR', X_test, Y_test)
### **TOTAL Results**

# Original
all_reg_scores(LR, 'LR', X_test, Y_test)
all_reg_scores(KNNR, 'KNNR', X_test, Y_test)
all_reg_scores(DTR, 'DTR', X_test, Y_test)
all_reg_scores(BGR, 'BGR', X_test, Y_test)
all_reg_scores(RFR, 'RFR', X_test, Y_test)
all_reg_scores(ExTR, 'ExTR', X_test, Y_test)
all_reg_scores(AdBR, 'AdBR', X_test, Y_test)
all_reg_scores(GBR, 'GBR', X_test, Y_test)
all_reg_scores(XGBR, 'XGBR', X_test, Y_test)
all_reg_scores(LGBMR, 'LGBMR', X_test, Y_test)
