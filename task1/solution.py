from sklearn.preprocessing import StandardScaler,PolynomialFeatures,RobustScaler
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.feature_selection import SequentialFeatureSelector, RFE,SelectFromModel
from sklearn import model_selection
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline,TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.utils.estimator_checks import check_estimator
from sklearn.linear_model import Lasso,LinearRegression,LassoCV,RidgeCV,ElasticNetCV
from sklearn.neighbors import LocalOutlierFactor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel,DotProduct
import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(0)

def load_data():
    x_train_df = pd.read_csv('X_train.csv').drop(labels='id',axis=1)
    x_test_df = pd.read_csv('X_test.csv').drop(labels='id',axis=1)
    y_train_df = pd.read_csv('y_train.csv').drop(labels='id',axis=1)
    
    return x_train_df,y_train_df,x_test_df

def remove_outliers(X,y=None, threshold=3.25):
    """
    Remove outliers from a DataFrame using the Z-score method.

    Parameters:
    df (DataFrame): Input DataFrame.
    threshold (int or float): Z-score threshold for outlier removal. Default is 3.

    Returns:
    DataFrame: DataFrame with outliers removed.
    """
    z_scores = np.abs(stats.zscore(X))
    # Create a boolean mask for outliers
    outliers_mask = (z_scores > threshold).any(axis=1)
    # Return the DataFrame without outliers
    return X[~outliers_mask], y[~outliers_mask]

#print(check_estimator(OutlierHandler(),generate_only=False))

def preprocess(X,y,Xtest):
    k = 30
    imputer = KNNImputer(n_neighbors=k)
    X = imputer.fit_transform(X)
    Xtest = imputer.transform(Xtest)
    X = pd.DataFrame(X)
    Xtest = pd.DataFrame(Xtest)
    """
    contamination = 0.05
    clf = IsolationForest(contamination=contamination, random_state=30)
    clf.fit(X)
    outlier = clf.fit_predict(X)
    inlier = outlier == 1
    outlier_index = X[~inlier].index
    X = X[inlier]
    y = y.drop(outlier_index)
    """
    X,y=remove_outliers(X,y)
    return X,y,Xtest

def get_data_ready(Xtrain, Ytrain, Xtest):
    Xtrain,Ytrain,Xtest = preprocess(Xtrain,Ytrain,Xtest)
    Xtrain,Ytrain,Xtest = Xtrain.to_numpy(),Ytrain.to_numpy(),Xtest.to_numpy()
    scaler = StandardScaler()
    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.fit_transform(Xtest)
    yscaler = StandardScaler()
    #Ytrain = yscaler.fit_transform(Ytrain)
    Poly = PolynomialFeatures(degree=1,include_bias=True,interaction_only=True)
    Xtrain = Poly.fit_transform(Xtrain)
    Xtest = Poly.transform(Xtest)
    xtrain, xeval, ytrain, yeval = model_selection.train_test_split(Xtrain,Ytrain)
    return xtrain, ytrain.ravel(), xeval, yeval.ravel(), Xtest, yscaler, Xtrain, Ytrain.ravel()

def fitModelMakePredictions(Xtrain,ytrain, Xeval, yeval,Xwhole,ywhole, Xtest,scaler):
    
    lasso = LassoCV(alphas = np.linspace(1e-2,1,1000),fit_intercept=False)
    
    lasso.fit(Xtrain, ytrain)
    print(lasso.alpha_)
    lasso_coef = lasso.coef_
    lasso.fit(Xtrain,ytrain)
    lasso_coef_threshold = 1e-7
    Xtrain_features = Xtrain[:, lasso_coef >= lasso_coef_threshold]
    Xeval_features = Xeval[:, lasso_coef >= lasso_coef_threshold]
    Xtest_features = Xtest[:, lasso_coef >= lasso_coef_threshold]
    Xwhole_features = Xwhole[:, lasso_coef >= lasso_coef_threshold]
    ols = Lasso(alpha = lasso.alpha_,fit_intercept=False)
    ols.fit(Xtrain_features,ytrain)
    #print(ols.alpha_)
    print(f"OLS fitted with evaluation {ols.score(Xeval_features,yeval)}")
    ols.fit(Xwhole,ywhole)
    result = ols.predict(Xtest)
    #result = scaler.inverse_transform(result.reshape(1,-1))
    return result.reshape(1,-1)
    
    """
    kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(0.0, 10.0)) * RBF(length_scale_bounds=(0.0, 10.0))+DotProduct(sigma_0_bounds = (1e-5,100000))
    gpr = GaussianProcessRegressor(kernel = kernel,normalize_y=True)
    gpr.fit(Xtrain,ytrain)
    print(f"GPR fitted with evaluation {gpr.score(Xeval,yeval)}")
    gpr.fit(Xwhole,ywhole)
    result = gpr.predict(Xtest)
    #result = scaler.inverse_transform(result.reshape(1,-1))
    
    return result.reshape(1,-1)
    """
def writeToFile(results):
    print("Writing results to file")
    res_df = pd.read_csv('sample.csv')
    res_df['y']=results
    res_df.to_csv('results.csv',index=False,header=True)
    
x_train_df,y_train_df,x_test_df = load_data()
xtrain, ytrain, xeval, yeval, Xtest, yscaler, Xtrain, Ytrain = get_data_ready(x_train_df, y_train_df, x_test_df)
result = fitModelMakePredictions(xtrain, ytrain, xeval, yeval, Xtrain,Ytrain, Xtest,yscaler)
writeToFile(result.reshape(-1))
