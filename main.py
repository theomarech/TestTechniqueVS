#!/usr/bin/env python
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
set_train_device = set(x_train.DEVICE)
set_test_device = set(x_test.DEVICE)
train_test_device_intersection = set_test_device.intersection(set_train_device)
print()
print(f"Modalities present in both test and training set for the DEVICE feature: {train_test_device_intersection}")
print()
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
time_train_dataset_target = training_set.loc[:, ['DEVICE', 'DATETIME', 'target']].sort_values(['DEVICE', 'DATETIME'])
sns.scatterplot(data=time_train_dataset_target,
                x="DATETIME",
                y="DEVICE",
                hue="target",
                units="DEVICE",
                estimator=None,
                palette=['blue', "red"])
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
cmap = sns.choose_diverging_palette()
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
def mat_to_long(mat_corr):
  """transform correlation matrix to 3D paired correlations dataframe (var1, var2, corr)"""
return mat_corr            .unstack()            .reset_index()            .query('level_0 != level_1')            .rename(columns={"level_0": "var1", "level_1": "var2", 0:"correlation"})            .sort_values('correlation', ascending=False)            .reset_index(drop=True)
top_10positives_corr = mat_to_long(x_train_quanti.corr()).head(10)
top_10negatives_corr = mat_to_long(x_train_quanti.corr()).sort_values('correlation').head(10)
display_side_by_side(top_10positives_corr,
                     top_10negatives_corr,
                     titles=['Top10: positives correlations',' negatives correlations'])
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
import shap
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_train)
shap.plots.bar(shap_values)
shap.plots.beeswarm(shap_values)
xgb.plot_importance(xgb_model)
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
explainer = shap.Explainer(xgb_model_dt)
shap_values = explainer(X_train)
shap.plots.bar(shap_values)
xgb.plot_importance(xgb_model_dt)
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
WINDOW_LENGTH = 30
OFFSET = 0
Y, X = create_dataset(training_set, WINDOW_LENGTH, OFFSET)
X_train, X_test_val, Y_train, Y_test_val = train_test_split(X, Y, test_size=0.3, shuffle=True)
X_val, X_test, Y_val, Y_test = train_test_split(X_test_val, Y_test_val, test_size=0.5, shuffle=True)
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
plot_model_metrics(lstm_history)
y_pred_bool = lstm_model.predict(X_test) > 0.5
y_pred = np.squeeze(y_pred_bool.astype('uint8'))
print(classification_report(y_pred, Y_test))
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
testing_set = x_test.copy()
testing_set['target'] = 0
x_test_pred = get_prediction(testing_set, impute_model=xgb_model_dt)
explainer = shap.DeepExplainer(lstm_model, X_train[:1000])
shap_values = explainer.shap_values(X_test[:500])
shap_df = pd.DataFrame({'shap': np.mean(shap_values[0], axis=(0,1)), "features": quantitative_features_uncorrelated})
shap_df.sort_values("shap", key=abs, ascending=False).plot("features", "shap", kind='bar')
time_test_dataset_target_lstm = x_test_pred.loc[:, ['DEVICE', 'DATETIME', 'target_pred']].sort_values(['DEVICE', 'DATETIME'])
sns.scatterplot(data=time_test_dataset_target_lstm,
                x="DATETIME",
                y="DEVICE",
                hue="target_pred",
                units="DEVICE",
                estimator=None,
                palette=['blue', "red"])
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
final_dataset = pd.merge(x_test, x_test_pred, on=["DEVICE", "DATETIME"], how="inner").loc[:, ["target_pred"]].astype("uint8")
final_dataset.to_feather("y_pred.feather")