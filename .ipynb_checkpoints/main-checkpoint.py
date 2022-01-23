#!/usr/bin/env python
# coding: utf-8

# #                                           Technical test Vegetal Signal
# 
# ## Summary
# 
# ### [**I - Loading dataset and main dataset structure**](#start_load_dataset)
# ### [**II - Exploratory analysis**](#start_exploratory_analysis)
# ### [*II - 1 - Univariate exploratory analysis*](#exploratory_univariate)
# #### [Categorical feature: `DEVICE`](#exploratory_univariate_categorical)
# #### [Datetime feature: `DATETIME`](#exploratory_univariate_datetime)
# #### [Quantitatives features: `feat0` to `feat99`](#exploratory_univariate_quantitative)
# ### [*II - 2 - Multivariate exploratory analysis*](#exploratory_multivariate)
# ### [**III - Modelling strategy**](#start_modeling)
# ### [*III - 1 Static models*](#static_models)
# #### [Static models with quantitatives features only](#static_models_quantiatives_features)
# #### [Static models with datetime as predictor](#static_models_datetime)
# ### [*III - 2 Dynamic models*](#dynamic_models)
# ### [**IV - Conclusion**](#start_conclusion)
# 
# ## <a id='start_load_dataset'>I - Loading dataset and main dataset structure</a>
# 
# 
# In this part we load the different dataset (train and test set) and summarize their main structure.
# The main goal here is to get a good understanding of the generals caracteristics (dimensionality, type of variables, missing values, scale of variables) of the three dataset.
# This understanding will give us the opportunity to guide our exploratory/modeling approach in the future parts.
# 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display_html
from itertools import chain,cycle

sns.set(rc={'figure.figsize': (20, 15)})

sns.set_theme(style="ticks", palette="pastel")

def display_side_by_side(*args, titles=cycle([''])):
    html_str=''
    for df, title in zip(args, chain(titles, cycle(['</br>'])) ):
        html_str+='<th style="text-align:center"><td style="vertical-align:top">'
        html_str+=f'<h2>{title}</h2>'
        html_str+=df.to_html().replace('table', 'table style="display:inline"')
        html_str+='</td></th>'
    display_html(html_str, raw=True)


X_TRAIN_PATH = "dataset/X_train.feather"
Y_TRAIN_PATH = "dataset/Y_train.feather"
X_TEST_PATH = "dataset/X_test.feather"

x_train = pd.read_feather(X_TRAIN_PATH)
y_train = pd.read_feather(Y_TRAIN_PATH)
x_test = pd.read_feather(X_TEST_PATH)

training_set = pd.concat([y_train, x_train], axis=1)

n_xtrain_instances, n_xtrain_features = x_train.shape
n_ytrain_instances, n_ytrain_features = y_train.shape
n_xtest_instances, n_xtest_features = x_test.shape

prevalence = y_train.mean()[0]

n_xtrain_missing = x_train.isna().sum().sum()
n_ytrain_missing = y_train.isna().sum().sum()
n_xtest_missing = x_test.isna().sum().sum()

n_duplicated = len(training_set) - len(training_set.drop_duplicates())

print("Training X set contain {0} instances and {1} features (ratio = {2:.0f} obs/feature) with {3} missing data"      .format(n_xtrain_instances,
              n_xtrain_features,
              n_xtrain_instances/n_xtrain_features,
              n_xtrain_missing,)
     )

print("Training Y set contain {0} binary instances and {1} features with a prevalence of {2:.0%} with {3} missing data"      .format(n_ytrain_instances,
              n_ytrain_features,
              prevalence,
              n_ytrain_missing)
     )

print("Testing X features contain {0} instances and {1} features with {2} missing data"      .format(n_xtest_instances,
              n_xtest_features,
              n_xtest_missing,)
     )

print(f"There is {n_duplicated} duplicated rows in the training dataset.")

training_set = training_set.drop_duplicates()

y_train = training_set.loc[:,["target"]]

x_train = training_set.drop(columns="target")


# 
# **Main observations:**
# - We have three datasets here:
# - a train set with 24632 instances and 103 features, composed by `x_train` and `y_train` to make the predictive model. 
#     - the `x_train` set is composed of 102 features with:
#         - 1 categorical features (`DEVICE`)
#         - 1 datetime features (`DATETIME`)
#         - 100 quantitatives features with uninformatives names (`feat0` to `feat99`)
#     - the y_train set is composed by a single dependent binary variable
# - a test set `x_test` with only the features which we will use to get the final predictions (same as `x_train` features)
# - obersavtions were collected over time so we have multiples time series
# - the dependent variable is binary and balanced (~45% of 1)
# - there is no missing data at all (no need to do observations/features deletion or imputation...)
# - there is 2248 duplicated observations in the training dataset
# 
# **Main conlusions:**
# - We have a consistent ratio of number observations/features with 241 observations per feature in training set.
# - We will use a binary classifier here
# - We will drop duplicated rows in the training set.
# - As the data were collected over time we will taking into account this dependency in our reflexion.
# - Because we don't have the dependent variable in the test set we can't used this variable in an autoregressive like model. In other word, only the features in x_train will be used to modelling the output feature y_train.
# 
# ## <a id='start_exploratory_analysis'>II - Exploratory analysis</a>
# 
# In this section we will make some descriptive analysis of the dataset to now what kind of preprocessing we should make on our data (transformation/deletion/selection of features).
# 
# First we start by describe/visualise each type of variables in an univariate way:
# - datetime variables: granularity and range and check if it match in xtrain/xtest dataset
# - categorical variable:  check if modality match in xtrain/xtest dataset (check data repartition in each category if they match)
# - quantitative variables: boxplot to check scale and distribution of each variables
# 
# Study variables relationship in the dataset:
# - quantitative variables: linear/non linear correlation matrix + PCA to see relation shape and possible redundancies between variable
# - response variables/quantitative relationship: boxplot + PCA
# 
# ### <a id='exploratory_univariate'> II - 1 Univariate exploratory analysis</a>
# 
# #### <a id="exploratory_univariate_categorical"> Categorical feature: DEVICE </a>
# 
# We don't know what is `DEVICE` feature. The first thing we can do is to check if modalities match in both training and testing set. If modalities match we can use `DEVICE` as candidate feature in our models. If not we will cannot use `DEVICE` as predictor.
# 
# For that we first look if modality of `DEVICE` in test set are present in the training set:
# 

set_train_device = set(x_train.DEVICE)
set_test_device = set(x_test.DEVICE)
train_test_device_intersection = set_test_device.intersection(set_train_device)
print()
print(f"Modalities present in both test and training set for the DEVICE feature: {train_test_device_intersection}")
print()


# We can see that none of the `DEVICE` modalities in training set are present in the testing set. So we cannot use `DEVICE` as predictor in our model.
# Another possibilities is that DEVICE represent different sampling device (at different position?). In this case we can imagine that we have one time serie per device modality. 
# 
# #### <a id='exploratory_univariate_datetime'> DATETIME features </a>
# 
# According to previous reflexion on the DEVICE feature we will check if each device represent a distinct time serie in both training and testing datasets. 
# 
# After that we will be interested in the sampling frequency over time. The main objective is to now what is the main granularity of sampling and if there is any cut-off in the differents time series to know if we have to handle this cut-off when we will use dynamic models.
# 

time_train_dataset = training_set.loc[:, ['DEVICE', 'DATETIME']].sort_values(['DEVICE', 'DATETIME'])

time_train_dataset["dataset"] = "train"

time_test_dataset = x_test.loc[:, ['DEVICE', 'DATETIME']].sort_values(['DEVICE', 'DATETIME'])

time_test_dataset["dataset"] = "test"

time_dataset = pd.concat([time_train_dataset, time_test_dataset], axis=0)

sns.scatterplot(data=time_dataset,
                x="DATETIME",
                y="DEVICE",
                hue="dataset",
                units="DEVICE",
                estimator=None,
                palette=['blue', "red"])


# Both training and testing dataset were sampled from the end of may to september 2021. As we suspect there is one time series per device in both training and testing set. Also, we can see cut-off in some time series and differents starting and ending time event.
# 
# 
# Another interesting thing is to plot our target feature for each sub time series to see if their are any obvious temporal pattern.

time_train_dataset_target = training_set.loc[:, ['DEVICE', 'DATETIME', 'target']].sort_values(['DEVICE', 'DATETIME'])

sns.scatterplot(data=time_train_dataset_target,
                x="DATETIME",
                y="DEVICE",
                hue="target",
                units="DEVICE",
                estimator=None,
                palette=['blue', "red"])


# We can see that a vast majority of the 1 values for the target feature are on the right part of the plot (durign the period from july to september) with nearly only 1 vlaues from august to september.
# Next we decribe more precisely the main granularity of the dataset and check for number and type of cut-off:

dfs_train = [x.sort_values('DATETIME').assign(dataset='train') for _, x in training_set.groupby('DEVICE')]
dfs_test = [x.sort_values('DATETIME').assign(dataset='test') for _, x in x_test.groupby('DEVICE')]
dfs_train_test = dfs_train + dfs_test
diff_series = [] 
dataset_type = []

for df in dfs_train_test:
    diff_series.append((df.DATETIME - df.DATETIME.shift(1)).dt.total_seconds()/3600)
    dataset_type.append(df.dataset)

diff_dataframe = pd.DataFrame({"diff (hours)" : pd.concat(diff_series),
              "dataset": pd.concat(dataset_type)}).reset_index(drop=True).dropna()

display(diff_dataframe.groupby(["dataset", "diff (hours)"])        .size()        .to_frame("n_obs")        .sort_values(["dataset", "n_obs"], ascending=False))


# **Main conlusion**
# 
# In the training dataframe we have different time series (one per device). To make temporal analysis we must taking into account breaks in temporality between to sample. In the general case we have a dataset sampling at the hour scale.  Even if we take into account duplicated rows and DEVICE to make our temporal dataframe we have break in each device temporal series with xt+1 - xt > 1hours for some pairs of points. Lot of target values equal to one are present on the period from july to september
# 

# ### <a id="exploratory_univariate_quantitative"> Quantitatives features: feat0 to feat99 </a>
# 
# Here we check for main caracteristics of the quantitatives features which will be our predictives variables.
# For that we are interesting in the scale of the features, the shape of the distribution and the presence of outliers.

print("Global summary statistic of quantitatives variables in x_train: \n")

display(x_train.describe())

quantitative_features = x_train.describe().columns

x_train_quanti = x_train[quantitative_features]
x_test_quanti = x_test[quantitative_features]

x_train_quanti_sample = x_train_quanti.sample(frac=0.01).melt(var_name="Variables", value_name="Values")
x_test_quanti_sample = x_train_quanti.sample(frac=0.01).melt(var_name="Variables", value_name="Values")

def plot_quantitative_var(df_quanti):
    """make boxplot of the quantitatives features"""
    fig = sns.boxplot(x="Variables", y="Values", data=df_quanti, showfliers=False).set_title('Boxplot of quantitative features')
    fig = sns.stripplot(x="Variables", y="Values", data=df_quanti, alpha=1.0, size=1.5)
    plt.xticks(rotation=45)

plot_quantitative_var(x_train_quanti_sample)
plt.show()
plot_quantitative_var(x_test_quanti_sample)
plt.show()


# **Scale of features** 
# 
# According to the summary and boxplot of the quantitatives features we can say that the features are globally on a relatively same scale.
# Even if features have relatively on the same scale (summary table + boxplot), it may be wise to rescale features (center/reduce) to be sure to not give an extra importance to some features for some of our models.
# 
# **Shape of the distributions**
# 
# Concerning shape of the features distribution, even if the vast majority of the data/features seem to form a symmetric distribution (see boxplot), we can see that a good part of the features are right/left skewed. For now we don't choose to make  features reshaping (like log/sqrt/boxcox transform).
# 
# **Outliers**
# 
# Even if we have points outside of the whisker range (depending on features) it's difficult to say if they are really "bad data" here. They seem to be in a general logical pattern when they are present. Without any extra informations about the dataset we choose to keep this data.
# 

# 
# ## <a id="exploratory_multivariate">II - 2 Multivariate analysis </a>
# 
# The main goal here is to describe the correlations between our quantitatives features (our only predictors here) to know if there are redundancy in our dataset. Because highly correlated features can be tricky for linear model  without regularisation (lasso, ridge...), we can perhaps choose to delete some of them or replace an all set of correlated variable by latent variable like PCA synthetic axis. 
# First of all we will check correlation matrix (pearson/spearman) and pairsplots to have an intuition of the shape and intensity of the relationship between the quantitatives features. Then, we will make a PCA to summarize the main linear structure of the dataset.
# 

cmap = sns.choose_diverging_palette()


# cmap = sns.diverging_palette(240, 10, n=20, as_cmap=True)

x_train_pearson_corr = x_train_quanti.corr()
x_train_spearman_corr = x_train_quanti.corr(method="spearman")

def heatmap(corr):
    """plot correlation heatmap"""
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(20, 15))
        ax = sns.heatmap(corr, mask=mask, square=True, center=0, cmap=cmap)
        
heatmap(x_train_spearman_corr)

heatmap(x_train_pearson_corr)


# According to the linear/non linear correlations heatmaps we can see groups of highly correlated (mostly positives) features with some correlations above 0.99. For example we can see a group of var (feat20 to feat31) which are mostly the same feature.
# To verify we can check the correlation dataset:

def mat_to_long(mat_corr):
    """transform correlation matrix to 3D paired correlations dataframe (var1, var2, corr)"""
    return mat_corr            .unstack()            .reset_index()            .query('level_0 != level_1')            .rename(columns={"level_0": "var1", "level_1": "var2", 0:"correlation"})            .sort_values('correlation', ascending=False)            .reset_index(drop=True)

top_10positives_corr = mat_to_long(x_train_quanti.corr()).head(10)
top_10negatives_corr = mat_to_long(x_train_quanti.corr()).sort_values('correlation').head(10)

display_side_by_side(top_10positives_corr,
                     top_10negatives_corr,
                     titles=['Top10: positives correlations',' negatives correlations'])


# As we expect there are lot of perfectly positives correlated (linearly) features and less higly negatives correlated features.
# Then, we want to drop highly correlated features to limit collinearity (if we want to use linear models) and to accelerate training in general. Another drawback of higly correlated features is it make it tricky to identify variables importances in general because mostly multiple features have the same effect/interactions with target/other features.
# 
# there is lot of methods to treat collinearity:
# 
# - select a set of features with correlations behind a correlation threshold (coupling with knowledge on the features to  decide to choose between two features)
# - make synthetics/latents features with group of features (e.g. dimensionality reduction: PCA (linear)/autoencoders (non linear))
# - for linear models use regularisation (lasso to select features or ridge or both)
# 
# Here we keep it simple and select features with correlation behind 0.80:
# 

from feature_engine.selection import DropCorrelatedFeatures


CORR_THRESHOLD = 0.8

tr_pearson = DropCorrelatedFeatures(variables=None, method='pearson', threshold=CORR_THRESHOLD)

x_train_quanti_uncorrelated_pearson = tr_pearson.fit_transform(x_train_quanti)

quantitative_features_uncorrelated_pearson = x_train_quanti_uncorrelated_pearson.columns

group_of_correlated_features_pearson = tr_pearson.correlated_feature_sets_



tr_spearman = DropCorrelatedFeatures(variables=None, method='spearman', threshold=CORR_THRESHOLD)

x_train_quanti_uncorrelated = tr_spearman.fit_transform(x_train_quanti_uncorrelated_pearson)

quantitative_features_uncorrelated = x_train_quanti_uncorrelated.columns

group_of_correlated_features_spearman = tr_spearman.correlated_feature_sets_



heatmap(x_train_quanti_uncorrelated.corr(method='pearson'))

group_of_correlated_features = group_of_correlated_features_pearson + group_of_correlated_features_pearson 


print("Group of correlated features: \n", group_of_correlated_features)
print()
print("Number of selected features: ", len(quantitative_features_uncorrelated))


# With our approach we have select 26 "uncorrelated" features (29 if we dropped only according to pearson correlation as criterion). A total of 76 features were dropped.
# 
# In the next sections we will continue with this 26 features as predictors.
# 
# We can check dimensionality of our dataset with a simple PCA. We colorized points with target value to see if there are any pattern.

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

LABELS = x_train_quanti_uncorrelated.columns

TARGET = y_train.target

def biplot(df, axes=(0, 1), colors=None, labels=None):
    n_var = len(df.columns)
    pca = PCA(n_components=n_var)
    score = pca.fit_transform(df)[:, axes]
    explained_var1, explained_var2 = pca.explained_variance_ratio_[[*axes]]
    xs = score[:, 0]
    ys = score[:, 1]
    coeff = np.transpose(pca.components_[axes, :])
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex, ys * scaley, c=colors, cmap='viridis', alpha=0.2)
    plt.axvline(0, 0, color="black", alpha=0.5, linestyle="--")
    plt.axhline(0, 0, color="black", alpha=0.5, linestyle="--")
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.8)
        if labels is None:
            plt.text(coeff[i, 0] * 1.1, coeff[i, 1] * 1.1, "Var" + str(i + 1), color='black', fontsize="x-large", ha='center', va='center')
        else:
            plt.text(coeff[i, 0] * 1.1, coeff[i, 1] * 1.1, labels[i], color='black', fontsize="x-large", ha='center', va='center')
    plt.xlabel("PC{0} ({1:.0%})".format(axes[0]+1, explained_var1))
    plt.ylabel("PC{0} ({1:.0%})".format(axes[1]+1, explained_var2))
    bar_pos = [.15, .71, .15, .15]
    bar_colors = np.array(["white"]*n_var)
    bar_colors[[*axes]] = "black"
    a = plt.axes(bar_pos, facecolor="white")
    plt.bar(range(n_var),
            pca.explained_variance_ratio_,
            color=tuple(bar_colors),
            edgecolor='black')
    plt.title('Variance explained (%)')
    plt.xticks([])
    plt.grid()
    
def many_biplot(df, list_of_axes_pairs, colors, labels):
    """plot many biplots (one for each pairs of axes)"""
    for axes in list_of_axes_pairs:
        biplot(df, colors=colors, labels=labels, axes=axes)
        plt.show()
        
scaled_x_train_quanti = pd.DataFrame(StandardScaler().fit_transform(x_train_quanti_uncorrelated), 
                           index = x_train_quanti_uncorrelated.index,
                           columns = x_train_quanti_uncorrelated.columns)

list_of_axes_pairs = [(0,1), (2,3), (4,5)]

many_biplot(scaled_x_train_quanti, list_of_axes_pairs, TARGET, LABELS)
        


# The first two axes of the PCA seem to be the largely above the other in term of explained variance (26% and 18% respectively) with a major drop after the 2nd axes. We cannot spot any pattern for the target value which seems to be not obviously related to some features. We can see that some points pull some PCA axis and give more importance to this axis.
# 
# We can use PCA a second time to see if there are major differences between features in our training/testing set. For that we run our PCA on all our observations and colorize points according to their dataset origin (training vs testing):

list_of_axes_pairs = [(0,1), (2,3), (4,5)]

df_train_to_concat = x_train.copy()
df_train_to_concat['name'] = 0
df_test_to_concat = x_test.copy()
df_test_to_concat['name'] = 1

test_train_dataset = pd.concat([df_train_to_concat, df_test_to_concat])


many_biplot(test_train_dataset.loc[:, quantitative_features_uncorrelated],
           list_of_axes_pairs,
           test_train_dataset.name,
           quantitative_features_uncorrelated)


# We dont see any obvious difference between our testing/training features with the PCA. We can look at the boxplot a last time to see our new selected features and comparing testing and training observations for each feature:

x_train_quanti_uncorrelated_plot = training_set[quantitative_features_uncorrelated]
x_train_quanti_uncorrelated_plot["dataset"] = "train"
x_test_quanti_uncorrelated_plot = x_test[quantitative_features_uncorrelated]
x_test_quanti_uncorrelated_plot["dataset"] = "test"

plot_dataset = pd.concat([x_train_quanti_uncorrelated_plot, x_test_quanti_uncorrelated_plot]).melt(id_vars=['dataset'])

sns.boxplot(data=plot_dataset,
            x="variable",
            y="value",
            hue="dataset")

plt.axhline(0,  linestyle="--", color="black", alpha=0.5)

sns.despine(offset=10, trim=True)


# As we saw previously training and testing features seem to be relatively equivalent with no major size effect. If we want to be more specific we could have used hypothesis testing but we keep it simple and even if there are significant differences the size effect would be relatively small according to the boxplot for a vast majority of the features.

# ## <a id='start_modeling'>III - Modeling strategy</a>
# 
# **What we know**:
# 
# - we want to build a binary classifier
# - we have multiple quantitative time series as predictors
# - test set and training set were collect relatively on the same period
# - the dataset leave in a lower dimensional space than the original dataset (see PCA/correlation matrix)
#     - feature selection (choosing one feature when multiple features are higly correlated)
#     - feature creation (used PCA axis as synthetic features)
# 
# **What we dont know**:
# 
# - if we want to build a time series model to predict future target value with past/current X values OR static model that use only current values
#     - pros: we have temporal data so past values must be good predictor of current values
#     - cons: we must build multiple models because we must predict at least the first values without previous values (number of steps choose to learn the temporal model)
# - what's more important to optimize to : (*e.g.* precision vs recall) ?
# 
# **Proposed modeling strategy**
# - evaluation strategy:
#     - dataset spliting strategy: because we have relatively long time series we choose to do a simple hold out train/val/test set.
#     - metrics choose:
#         - target values are balanced (prevalence=45%) + because we dont know what is important to optimize for (precision vs recall) we evaluate the model with f1 score/accuracy.
#         => the baseline accuracy is about 55% according to the prevalence
#     
# - In the first part we test simple static models like logistic regression/random forest/xgboost
# - In a second part we will used temporal deep learning model with LSTM model + best static model to predict first values
# 
# NOTE: If the main objective is to create the best predictor model for the test set maybe the best strategy is to use a bidirectionnal LSTM model (or Attention based model) and use future and past values to predict the target yt, and create different(s) model(s) to predict the first target values (when we don't have the previous values to predict yt).
# 
# ### <a id="static_models">III - 1 Static models<a>
#     
# In this part we just use some classic machine learning classifier to predict our target variable. We don't use the dynamic structure of our dataset here, we use it as if  each observations were independently sample.
#     
# #### <a id="static_models_quantiatives_features">Static models with quantitatives features only<a>
# 

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb

X_train, X_val_test, Y_train, Y_val_test = train_test_split(x_train[quantitative_features_uncorrelated], y_train.target, test_size=0.3)

X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, test_size=0.5)

print(
    X_train.shape,
    X_val.shape,
    X_test.shape,
)


# We start by a simple logistic regression as classifier. 


pipe_lr = Pipeline([('scaler', StandardScaler()),('logistic', LogisticRegression())])

param_grid_lr = {
    "logistic__C": np.logspace(-4, 4, 4),
}

rds_lr_model = GridSearchCV(pipe_lr,
                            param_grid_lr,
                            n_jobs=-1)

rds_lr_model.fit(X_train, Y_train)

print('Training score:', rds_lr_model.score(X_train, Y_train))
print('Validation score:', rds_lr_model.score(X_val, Y_val))


# The logistic regression model seem to not beat the baseline accuracy (around 55%)

pipe_svm = Pipeline([('scaler', StandardScaler()),('SVM', SVC())])

param_grid_svm = {
    "SVM__kernel": ["linear", "poly", "rbf"],
    "SVM__degree": [2, 3, 4, 5, 6],
    "SVM__C": np.logspace(-4, 4, 4),
}

rds_svc_model = RandomizedSearchCV(pipe_svm,
                                   param_grid_svm,
                                   n_iter=30,
                                   n_jobs=-1)

rds_svc_model.fit(X_train, Y_train)


print('Training score:', rds_svc_model.score(X_train, Y_train))
print('Validation score:', rds_svc_model.score(X_val, Y_val))


# Much better for the svm classifier with a validation accuracy of 70%

model_rf = RandomForestClassifier()

rf_params = {
    "criterion": ["gini", "entropy"],
    "max_features": ["auto", "log2"],
    "n_estimators": [50, 100, 200, 400, 800, 1000, 1500],
    "min_samples_split": np.linspace(0.01, 0.1, 10),
}


rf_rand_grid = RandomizedSearchCV(
        estimator=model_rf,
        n_iter=30,
        param_distributions=rf_params,
        cv=5,
        n_jobs=-1,
)

rf_rand_grid.fit(X_train, Y_train)

print('Training score:', rf_rand_grid.score(X_train, Y_train))
print('Validation score:', rf_rand_grid.score(X_val, Y_val))



# xgb_params = {
#     "eta": [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
#     "max_depth": [4, 5, 6, 7, 8, ,10, 12],
#     "subsample": [0.5, 0.6, 0.7, 0.8, 0.9 ,0.95, 1.0],
#     "min_child_weight": [1, 2, 3, 4, 5, 6], 
#     "gamma": [0, 1, 2, 3, 4, 5, 10, 15, 20],
#     "reg_lambda": [0, 0.1, 0.4, 0.8, 1, 3, 6, 15, 20],
#     "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9 ,0.95, 1.0], 
#     "n_estimators": [400,  600, 800,  1000, 1200,  1400, 1500],
# }

# xgb_model = xgb.XGBClassifier()

# rd_cv_xgb = RandomizedSearchCV(
#         estimator=xgb_model,
#         n_iter=300,
#         param_distributions=xgb_params,
#         cv=2,
#         n_jobs=-1,
# )

# rd_cv_xgb.fit(X_train, Y_train)

# rd_cv_xgb.score(X_val, Y_val)

xgb_model = xgb.XGBClassifier(booster="gbtree",
                              n_estimators=1200,
                              max_depth=10,
                              subsample=0.75,
                              gamma=1,
                              reg_lambda=1,
                              colsample_bytree=0.75, 
                              eta=0.05,
                              n_jobs=-1,
                              tree_method="hist",
                              use_label_encoder=False)


xgb_model.fit(X_train, Y_train)


print('Training score:', xgb_model.score(X_train, Y_train))
print('Validation score:', xgb_model.score(X_val, Y_val))


# Xgboost reach 76% of accuracy in our validation set but overfit largely our training dataset. Next we will see feature importance with shapley values.

import shap


explainer = shap.Explainer(xgb_model)

shap_values = explainer(X_train)

shap.plots.bar(shap_values)


shap.plots.beeswarm(shap_values)


xgb.plot_importance(xgb_model)


# We can see that there is no very important feature given the other.
# 
# Because we dramasticly decreased the number of features with an arbitrary correlation threshold we can refit an xgboost with a les restrictives threshold to test if some important features were dropped.

from feature_engine.selection import DropCorrelatedFeatures


CORR_THRESHOLD = 0.95

tr_pearson_95 = DropCorrelatedFeatures(variables=None, method='pearson', threshold=CORR_THRESHOLD)

x_train_quanti_uncorrelated_pearson_95 = tr_pearson_95.fit_transform(x_train_quanti)

quantitative_features_uncorrelated_pearson_95 = x_train_quanti_uncorrelated_pearson_95.columns

group_of_correlated_features_pearson_95 = tr_pearson_95.correlated_feature_sets_



tr_spearman_95 = DropCorrelatedFeatures(variables=None, method='spearman', threshold=CORR_THRESHOLD)

x_train_quanti_uncorrelated_95 = tr_spearman_95.fit_transform(x_train_quanti_uncorrelated_pearson_95)

quantitative_features_uncorrelated_95 = x_train_quanti_uncorrelated_95.columns

group_of_correlated_features_spearman_95 = tr_spearman_95.correlated_feature_sets_

X_train, X_val_test, Y_train, Y_val_test = train_test_split(x_train[quantitative_features_uncorrelated_95], y_train.target, test_size=0.3)

X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, test_size=0.5)

print(
    X_train.shape,
    X_val.shape,
    X_test.shape,
)


# With a threshold of 0.95 we have almostly double our number of features

xgb_model = xgb.XGBClassifier(booster="gbtree",
                              n_estimators=1200,
                              max_depth=10,
                              subsample=1,
                              gamma=0,
                              reg_lambda=1,
                              colsample_bytree=1, 
                              eta=0.2,
                              n_jobs=-1,
                              tree_method="hist",
                              use_label_encoder=False)


xgb_model.fit(X_train, Y_train)


print('Training score:', xgb_model.score(X_train, Y_train))
print('Validation score:', xgb_model.score(X_val, Y_val))


# **To sum up**
# 
# We have tested here some models (logistic regression, svm, random forest, xgboost) with a best accuracy of around 81% reach by xgboost. We use only our quantitatives features as predictors, without any other informations on datetime or previous values (lag features for example).
# 
# #### <a id="static_models_datetime">Static models with datetime as predictor<a>
# 
# Previously we have see that our target feature seemed to be very time related in our training set.
# So, before to start with pure dynamic models we can just test to add our datetime feature in our model. We can use this feature easily here because our test set were sampled over the same period. 
# We dont do very complicated feature engineering with our datetime  feature (like break it into multiple day/month/hour or create periodic trigonometric features) we just transform into epochs from 1970 to simplify for our algorithm.
# We continue with an xgboost model because it was the best previously.

import datetime as dt

x_train_quanti_uncorrelated_date = x_train[quantitative_features_uncorrelated].copy()
x_train_quanti_uncorrelated_date["dt"] = (x_train.DATETIME - dt.datetime(1970,1,1)).dt.total_seconds()

X_train, X_val_test, Y_train, Y_val_test = train_test_split(x_train_quanti_uncorrelated_date, y_train.target, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, test_size=0.5)


xgb_model_dt = xgb.XGBClassifier(n_estimators=1500,
                              max_depth=7,
                              subsample=1.0,
                              colsample_bytree=1, 
                              eta=0.2,
                              n_jobs=-1)

xgb_model_dt.fit(X_train, Y_train)

xgb_model_dt.score(X_val, Y_val)


# We can see that add our datetime feature drasticly rise our accuracy (81 to 96 ; ~ +17 points). It's not surprising given the distribution of our target value along the time axis where it is very clustered and localized.
# As before we can use shap values to see features importances:

explainer = shap.Explainer(xgb_model_dt)

shap_values = explainer(X_train)

shap.plots.bar(shap_values)


xgb.plot_importance(xgb_model_dt)


# As we could expect the datetime feature are a very important feature here.

# ### <a id="dynamic_models">III - 2 Dynamic models<a>
# 
# In this part we will use the temporal structure of the dataset to create dynamic models.
# Our main goal is to predict correctly the target in the test set. Without any other information about the goal of the model (sequence model? to forecast in the future? how many time in the future? one prediction? a sequence of prediction?) we will only stay on a simple RNN like model a LSTM.
# 

import datetime
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_v2_behavior()



def add_marker_of_cut_off(df):
    """Add marker (in a new column call 'cut_off') to identify sub time series with homogenous time sampling"""
    name = df.DEVICE.iloc[0]
    cut_off = []
    lag = datetime.timedelta(3600)
    start =  1
    for i in range(len(df)-1):
        diff = df.iloc[i+1].DATETIME - df.iloc[i].DATETIME
        cut_off.append(name + "_" + str(start))
        if diff > lag:
            start += 1
        lag = diff
    cut_off.append(name + "_" +  str(start))
    df['cut_off'] = cut_off
    return  df

def create_time_dataset(dataset, n_step, offset):
    """create a dataset with n_step values before offset target for RNN like model."""
    X = []
    Y = []
    n = len(dataset)
    for i in range(n - n_step - offset):
        X.append(dataset.iloc[i:i + n_step, 1:])
        Y.append(dataset.iloc[n_step - 1 + i + offset].target)
    return np.asarray(Y), np.asarray(X)

def create_time_datasets(datasets, n_step, offset):
    """create time dataset for multiple input series"""
    Y, X = create_time_dataset(datasets[0], n_step, offset)
    for df in datasets[1:]:
        y, x = create_time_dataset(df, n_step, offset)
        X = np.concatenate([X, x])
        Y = np.concatenate([Y, y])
    return Y.astype("uint8"), X.astype("float32")

def create_dataset(dataset, WINDOW_LENGTH=30, offset=0, ascending=True):
    """create final time dataset"""
    VARS = ["target"] + list(quantitative_features_uncorrelated)
    dfs = [x for _, x in dataset.groupby('DEVICE')]
    df_cutoff = [add_marker_of_cut_off(df) for df in dfs]
    time_training_set = pd.concat(df_cutoff).reset_index(drop=True).sort_values(["cut_off", "DATETIME"], ascending=ascending)
    time_training_sets = [x[VARS] for _, x in time_training_set.groupby('cut_off') if len(x) > WINDOW_LENGTH + offset]
    Y, X = create_time_datasets(time_training_sets, WINDOW_LENGTH, offset)
    return Y, X

def plot_model_metrics(model_history):
    """plot training and validation loss and accuracy"""
    plt.figure(figsize=(20, 8))
    history = model_history.history
    train_loss = history['loss']
    val_loss = history['val_loss']
    train_acc = history['acc']
    val_acc = history['val_acc']
    epochs = range(len(train_loss))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="training loss")
    plt.plot(epochs, val_loss, c="red", label="validation loss")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss (binary crossentropy)")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="training accuracy")
    plt.plot(epochs, val_acc, c="red", label="validation accuracy")
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    
def get_prediction(dataset, impute_model=xgb_model_dt):
    """take a dataset a return all prediction"""
    dfs_test = [x.sort_values("DATETIME") for _, x in dataset.groupby("DEVICE")]
    df_test_cutoff = [add_marker_of_cut_off(df) for df in dfs_test]
    list_cut_off = [series for _, series in pd.concat(df_test_cutoff).groupby("cut_off")]
    list_final_pred = np.asarray([])
    for df in list_cut_off:
        if len(df) >= 2*WINDOW_LENGTH:
            u, X = create_dataset(df, WINDOW_LENGTH, OFFSET, ascending=True)
            _, X_reverse = create_dataset(df, WINDOW_LENGTH, OFFSET, ascending=False)
            y_pred_bool = lstm_model.predict(X) > 0.5
            y_pred_bool_reverse = lstm_model_reverse.predict(X_reverse) > 0.5
            y_pred = np.squeeze(y_pred_bool.astype('uint8'))
            y_pred_reverse = np.squeeze(y_pred_bool_reverse.astype('uint8'))[::-1]
            final_pred = np.concatenate([y_pred_reverse[:WINDOW_LENGTH], y_pred])
            list_final_pred = np.concatenate([list_final_pred, final_pred])
        else:
            if impute_model is not None:
                df_xgb = df.loc[:, ["DATETIME"]+list(quantitative_features_uncorrelated)]
                df_xgb["dt"] = (df_xgb.DATETIME - dt.datetime(1970,1,1)).dt.total_seconds()
                df_xgb = df_xgb.drop(columns=['DATETIME'])
                result_impute_pred = impute_model.predict(df_xgb)
                list_final_pred = np.concatenate([list_final_pred, result_impute_pred])
            else:
                list_final_pred = np.concatenate([list_final_pred, np.array([np.nan]*len(df))])
    final_dataset_pred = dataset.sort_values(["DEVICE",'DATETIME']).loc[:, ["target", "DEVICE", "DATETIME"]]
    final_dataset_pred["target_pred"] = list_final_pred
    final_dataset_pred = final_dataset_pred.fillna(method='ffill')
    return final_dataset_pred


# We start by using 30 previous predictors values (t to t-30) to predict the 30th target value.

WINDOW_LENGTH = 30

OFFSET = 0

Y, X = create_dataset(training_set, WINDOW_LENGTH, OFFSET)

X_train, X_test_val, Y_train, Y_test_val = train_test_split(X, Y, test_size=0.3, shuffle=True)

X_val, X_test, Y_val, Y_test = train_test_split(X_test_val, Y_test_val, test_size=0.5, shuffle=True)


# To start we will test a very simple network with one LSTM layers follow by a single dense layer with one neurons. We add reccurent dropout as regularization strategy.


N_NEURONS = 40
RECURRENT_DROPOUT = 0.35
LEARNING_RATE = 1e-3
BATCH_SIZE = 30
EPOCHS = 80

lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(N_NEURONS, recurrent_dropout=RECURRENT_DROPOUT),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

lstm_model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    metrics=["acc"]
)

lstm_history = lstm_model.fit(
    X_train,
    Y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, Y_val),
    shuffle=True,
)


# 
# 

plot_model_metrics(lstm_history)


# According to the learning curves we dont overfit our training set and seem to reach the plateau rapidly (~20 epochs).  We can see a little bit of loss/accuracy fluctuation, maybe a much bigger batch size would be necessary to reduce this phenomenon.
# 
# Now we can test our model on the test set:

y_pred_bool = lstm_model.predict(X_test) > 0.5
y_pred = np.squeeze(y_pred_bool.astype('uint8'))
print(classification_report(y_pred, Y_test))


# We can see that with a very simple LSTM model we can predict almost perfectly our validation and testing set. Because we have a very high accuracy score we stop our search of differents dynamics models.
# If we want to predict all the test set we must create a second LSTM model to predict the first 30 values of each time series. And because certain pieces of time series are very short (below 30 values) we will use our xgboost model as imputer.

WINDOW_LENGTH = 30

OFFSET = 0

Y, X = create_dataset(training_set, WINDOW_LENGTH, OFFSET, ascending=False)

X_train, X_test_val, Y_train, Y_test_val = train_test_split(X, Y, test_size=0.3, shuffle=True)

X_val, X_test, Y_val, Y_test = train_test_split(X_test_val, Y_test_val, test_size=0.5, shuffle=True)


N_NEURONS = 40
RECURRENT_DROPOUT = 0.35
LEARNING_RATE = 1e-3
BATCH_SIZE = 30
EPOCHS = 70

lstm_model_reverse = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(N_NEURONS, recurrent_dropout=RECURRENT_DROPOUT),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

lstm_model_reverse.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    metrics=["acc"]
)

lstm_history_reverse = lstm_model_reverse.fit(
    X_train,
    Y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, Y_val),
    shuffle=True,
)


y_pred_bool = lstm_model_reverse.predict(X_test) > 0.5
y_pred = np.squeeze(y_pred_bool.astype('uint8'))
print(classification_report(y_pred, Y_test))


# Get final prediction with the two LSTM models for the test set with xgboost model as imputer for sub temporal series with length below 2 * WINDOW_length.

testing_set = x_test.copy()
testing_set['target'] = 0
x_test_pred = get_prediction(testing_set, impute_model=xgb_model_dt)


# We can look at shapley values for features importance

explainer = shap.DeepExplainer(lstm_model, X_train[:1000])
shap_values = explainer.shap_values(X_test[:500])
shap_df = pd.DataFrame({'shap': np.mean(shap_values[0], axis=(0,1)), "features": quantitative_features_uncorrelated})
shap_df.sort_values("shap", key=abs, ascending=False).plot("features", "shap", kind='bar')


# Now we can look at the x_test prediction of our top 2 best models lstm and xgboost. We will plot our prediction for each sub time series.
# We start with the LSTM models:

time_test_dataset_target_lstm = x_test_pred.loc[:, ['DEVICE', 'DATETIME', 'target_pred']].sort_values(['DEVICE', 'DATETIME'])

sns.scatterplot(data=time_test_dataset_target_lstm,
                x="DATETIME",
                y="DEVICE",
                hue="target_pred",
                units="DEVICE",
                estimator=None,
                palette=['blue', "red"])


# Predictions seems to be more dispatch (less clustered) along the datetime axis comparing to the training dataset.
# 
# Check of our xgboost model with datetime feature:

x_test_quanti_uncorrelated_date = x_test[quantitative_features_uncorrelated].copy()
x_test_quanti_uncorrelated_date["dt"] = (x_test.DATETIME - dt.datetime(1970,1,1)).dt.total_seconds()
target_test_xgb = xgb_model_dt.predict(x_test_quanti_uncorrelated_date)
time_test_dataset_target_xgbdt = x_test.copy()
time_test_dataset_target_xgbdt['target_pred'] = target_test_xgb

sns.scatterplot(data=time_test_dataset_target_xgbdt,
                x="DATETIME",
                y="DEVICE",
                hue="target_pred",
                units="DEVICE",
                estimator=None,
                palette=['blue', "red"])


# On the contrary of our LSTM model, here predictions are really close of our training set in terms of datetime repartition.

# ### <a id="start_conclusion">**IV - Conclusion**<a>
#         
# The most difficult part here is to choosing between a model that incorporate the datetime variable (xgboost here) and the LSTM model which uses multiple previous values of the features without using explictly DATETIME feature.
# The LSTM model achieves a perfect classification (> 99%) on the training dataset and does not appear to be overfitted. The less good but very efficient xgboost model (97% accuracy) gives test predictions that are closer of the training data in terms of temporal distribution pattern (which seems logical given that it uses the datetime feature as predictor). If we compare this two models, they gives relatively different results on the test set but with a very high accuracy on the training set. So, it is difficult to decide between these two models here without any additional information. 
# Here we choose to keep LSTM (because we choose to juge on accuracy and maybe more parcimonious because it do not use the datetime feature) but with lot of reserve.
# 
# 

final_dataset = pd.merge(x_test, x_test_pred, on=["DEVICE", "DATETIME"], how="inner").loc[:, ["target_pred"]].astype("uint8")
final_dataset.to_feather("y_pred.feather")

