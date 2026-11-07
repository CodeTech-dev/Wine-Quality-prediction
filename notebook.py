#!/usr/bin/env python
# coding: utf-8

# In[61]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

import pickle

get_ipython().run_line_magic('matplotlib', 'inline')


# In[62]:


red = pd.read_csv('/content/winequality-red.csv', sep=';')
white = pd.read_csv('/content/winequality-white.csv', sep=';')

red['wine_type'] = 'red'
white['wine_type'] = 'white'

# merging the two datasets
df = pd.concat([red, white], ignore_index=True)
df.shape


# In[63]:


df.head()


# In[64]:


df.columns = df.columns.str.lower().str.replace(' ', '_')
df.columns


# In[65]:


df.isnull().sum()


# In[66]:


df.describe()


# In[67]:


print('Rows before:', len(df))
print('Duplicates:', df.duplicated().sum())

# drop the duplicate values
df = df.drop_duplicates().reset_index(drop=True)
print('Rows after:', len(df))


# In[68]:


df['good'] = (df.quality >= 7).astype(int)
df.good.value_counts(normalize=True)


# In[69]:


categorical_features = ['wine_type']
numerical_features = [col for col in df.columns
                      if col not in categorical_features + ['quality', 'good']]

print('Numerical:', numerical_features)
print('Categorical:', categorical_features)


# # EDA

# In[70]:


df.good.value_counts().plot(kind='bar')
plt.title('Good vs not good')
plt.show()

pd.crosstab(df.wine_type, df.good, normalize='index')


# In[71]:


for col in numerical_features:
    sns.histplot(df[col], kde=True)
    plt.title(col)
    plt.show()


# In[72]:


df.groupby('good')[numerical_features].mean().T


# In[73]:


plt.figure(figsize=(8, 6))
sns.heatmap(df[numerical_features + ['good']].corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.show()


# In[74]:


df.groupby(['wine_type', 'good'])[numerical_features].mean().T


# # Setting up validation framework

# In[75]:


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df.good)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42, stratify=df_full_train.good)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

len(df_train), len(df_val), len(df_test)


# In[76]:


print(df_train.good.mean(), df_val.good.mean(), df_test.good.mean())


# In[77]:


y_train = df_train.good.values
y_val = df_val.good.values
y_test = df_test.good.values

for d in [df_train, df_val, df_test]:
    del d['good']
    del d['quality']


# In[78]:


train_dicts = df_train.to_dict(orient='records')
val_dicts = df_val.to_dict(orient='records')
test_dicts = df_test.to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)
X_test = dv.transform(test_dicts)

dv.get_feature_names_out()


# In[79]:


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score, recall_score, f1_score


# In[80]:


def evaluate(model, X, y, threshold=0.5):
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    return {
        'roc_auc': roc_auc_score(y, y_prob),
        'pr_auc': average_precision_score(y, y_prob),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
    }


# # A. Logistic regression model

# In[81]:


lr = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)
lr.fit(X_train, y_train)

results = []
results.append({'model': 'LogReg', **evaluate(lr, X_val, y_val)})
pd.DataFrame(results).round(3)


# In[82]:


lr_bal = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
)
lr_bal.fit(X_train, y_train)

results.append({'model': 'LogReg (balanced)', **evaluate(lr_bal, X_val, y_val)})
pd.DataFrame(results).round(3)


# In[83]:


scores = []

for C in [0.001, 0.01, 0.1, 1, 10, 100]:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=1000, random_state=42)
    )
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    scores.append((C, auc))
    print(f'C={C}: Validation AUC = {auc:.3f}')


# # B. Random forest Model classifier

# In[84]:


scores = []

for n in range(10, 201, 10):
    rf = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    y_pred = rf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)

    scores.append((n, auc))

df_scores = pd.DataFrame(scores, columns=['n_estimators', 'auc'])
df_scores


# In[85]:


plt.plot(df_scores.n_estimators, df_scores.auc)
plt.xlabel('n_estimators')
plt.ylabel('Validation AUC')
plt.show()


# In[86]:


scores = []

for d in [5, 10, 15, None]:
    for n in range(10, 201, 10):
        rf = RandomForestClassifier(n_estimators=n, max_depth=d,
                                    random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)

        y_pred = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        scores.append((d, n, auc))

df_scores = pd.DataFrame(scores, columns=['max_depth', 'n_estimators', 'auc'])


# In[87]:


df_scores


# In[88]:


for d in [5, 10, 15, None]:
    df_subset = df_scores[df_scores.max_depth.isna()] if d is None \
                else df_scores[df_scores.max_depth == d]
    plt.plot(df_subset.n_estimators, df_subset.auc, label=f'max_depth={d}')

plt.legend()
plt.xlabel('n_estimators')
plt.ylabel('Validation AUC')
plt.show()


# In[89]:


max_depth = 15

scores = []

for s in [1, 3, 5, 10, 20]:
    for n in range(10, 201, 10):
        rf = RandomForestClassifier(n_estimators=n, max_depth=max_depth,
                                    min_samples_leaf=s,
                                    random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)

        y_pred = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        scores.append((s, n, auc))

df_scores = pd.DataFrame(scores, columns=['min_samples_leaf', 'n_estimators', 'auc'])


# In[90]:


df_scores


# In[91]:


for s in [1, 3, 5, 10, 20]:
    df_subset = df_scores[df_scores.min_samples_leaf == s]
    plt.plot(df_subset.n_estimators, df_subset.auc, label=f'min_samples_leaf={s}')

plt.legend()
plt.xlabel('n_estimators')
plt.ylabel('Validation AUC')
plt.show()


# Random forest parameter tunning

# In[92]:


from sklearn.model_selection import GridSearchCV

params = {
    'n_estimators': [100, 150, 200],
    'max_depth': [10, 15, None],
    'min_samples_leaf': [1, 2, 3, 5],
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=params,
    scoring='roc_auc',
    cv=5,
    verbose=1,
)
grid_search.fit(X_train, y_train)

print('Best params:', grid_search.best_params_)
print('Best CV AUC:', round(grid_search.best_score_, 4))


# In[93]:


best_rf = grid_search.best_estimator_

results.append({'model': 'RandomForest', **evaluate(best_rf, X_val, y_val)})
pd.DataFrame(results).round(3)


# # C. XGBoost Model

# In[94]:


xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    eval_metric='auc',
    random_state=42,
    n_jobs=-1,
)

xgb.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=False,
)

evals = xgb.evals_result()
plt.plot(evals['validation_0']['auc'], label='train')
plt.plot(evals['validation_1']['auc'], label='validation')
plt.xlabel('number of trees')
plt.ylabel('AUC')
plt.legend()
plt.show()


# In[95]:


scores = {}

for eta in [0.01, 0.05, 0.1, 0.3]:
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=eta,
        max_depth=6,
        eval_metric='auc',
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)], verbose=False)
    scores[eta] = model.evals_result()['validation_0']['auc']

for eta, aucs in scores.items():
    plt.plot(aucs, label=f'eta={eta}')

plt.xlabel('number of trees')
plt.ylabel('Validation AUC')
plt.legend()
plt.show()

for eta, aucs in scores.items():
    print(f'eta={eta}: best AUC = {max(aucs):.4f} at tree {int(np.argmax(aucs)) + 1}')


# In[96]:


scores = {}

for d in [2, 3, 4, 6]:
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=d,
        eval_metric='auc',
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)], verbose=False)
    scores[d] = model.evals_result()['validation_0']['auc']

for d, aucs in scores.items():
    plt.plot(aucs, label=f'max_depth={d}')

plt.xlabel('number of trees')
plt.ylabel('Validation AUC')
plt.legend()
plt.show()

for d, aucs in scores.items():
    print(f'max_depth={d}: best AUC = {max(aucs):.4f} at tree {int(np.argmax(aucs)) + 1}, '
          f'AUC at tree 200 = {aucs[199]:.4f}')


# In[97]:


max_depth = 6   # change this to the best depth from Cell 30

scores = {}

for w in [1, 5, 10, 30]:
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=max_depth,
        min_child_weight=w,
        eval_metric='auc',
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)], verbose=False)
    scores[w] = model.evals_result()['validation_0']['auc']

for w, aucs in scores.items():
    plt.plot(aucs, label=f'min_child_weight={w}')

plt.xlabel('number of trees')
plt.ylabel('Validation AUC')
plt.legend()
plt.show()

for w, aucs in scores.items():
    print(f'min_child_weight={w}: best AUC = {max(aucs):.4f}, AUC at tree 200 = {aucs[199]:.4f}')


# In[98]:


scores = {}

for d in [6, 8, 10, 12]:
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=d,
        min_child_weight=1,
        eval_metric='auc',
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)], verbose=False)
    scores[d] = model.evals_result()['validation_0']['auc']

for d, aucs in scores.items():
    plt.plot(aucs, label=f'max_depth={d}')

plt.xlabel('number of trees')
plt.ylabel('Validation AUC')
plt.legend()
plt.show()

for d, aucs in scores.items():
    print(f'max_depth={d}: best AUC = {max(aucs):.4f}, AUC at tree 200 = {aucs[-1]:.4f}')


# In[99]:


max_depth = 6

configs = [(1.0, 1.0), (0.8, 1.0), (0.8, 0.8), (0.6, 0.8), (0.6, 0.6)]
scores = {}

for sub, col in configs:
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=max_depth,
        min_child_weight=1,
        subsample=sub,
        colsample_bytree=col,
        eval_metric='auc',
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)], verbose=False)
    scores[(sub, col)] = model.evals_result()['validation_0']['auc']

for (sub, col), aucs in scores.items():
    plt.plot(aucs, label=f'subsample={sub}, colsample={col}')

plt.xlabel('number of trees')
plt.ylabel('Validation AUC')
plt.legend(fontsize=8)
plt.show()

for (sub, col), aucs in scores.items():
    print(f'subsample={sub}, colsample={col}: best = {max(aucs):.4f}, at tree 200 = {aucs[-1]:.4f}')


# In[100]:


xgb_final = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    min_child_weight=1,
    eval_metric='auc',
    random_state=42,
    n_jobs=-1,
)
xgb_final.fit(X_train, y_train)

results.append({'model': 'XGBoost', **evaluate(xgb_final, X_val, y_val)})
pd.DataFrame(results).round(3)


# K-FOld Cross validation

# In[101]:


from sklearn.model_selection import StratifiedKFold, cross_validate

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'LogReg': lr,
    'RandomForest': best_rf,
    'XGBoost': xgb_final,
}

cv_rows = []
for name, model in models.items():
    out = cross_validate(model, X_train, y_train, cv=cv,
                         scoring={'roc_auc': 'roc_auc', 'pr_auc': 'average_precision'})
    cv_rows.append({
        'model': name,
        'roc_auc_mean': out['test_roc_auc'].mean(),
        'roc_auc_std': out['test_roc_auc'].std(),
        'pr_auc_mean': out['test_pr_auc'].mean(),
        'pr_auc_std': out['test_pr_auc'].std(),
    })

pd.DataFrame(cv_rows).round(3)


# In[102]:


from sklearn.metrics import precision_recall_curve

y_val_prob = best_rf.predict_proba(X_val)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, y_val_prob)

f1 = 2 * precision * recall / (precision + recall + 1e-9)

plt.plot(thresholds, precision[:-1], label='precision')
plt.plot(thresholds, recall[:-1], label='recall')
plt.plot(thresholds, f1[:-1], label='F1')
plt.xlabel('threshold')
plt.legend()
plt.show()

best_idx = np.argmax(f1[:-1])
print(f'F1-optimal threshold: {thresholds[best_idx]:.3f}')
print(f'precision={precision[best_idx]:.3f}, recall={recall[best_idx]:.3f}, F1={f1[best_idx]:.3f}')


# In[103]:


rows = []
for t in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
    rows.append({'threshold': t, **evaluate(best_rf, X_val, y_val, threshold=t)})

pd.DataFrame(rows)[['threshold', 'precision', 'recall', 'f1']].round(3)


# In[104]:


threshold = 0.32


# In[105]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

test_results = []
for name, model in [('LogReg', lr), ('RandomForest', best_rf), ('XGBoost', xgb_final)]:
    # t = threshold if name == 'RandomForest' else 0.5
    t = threshold
    test_results.append({'model': name, 'threshold': t,
                         **evaluate(model, X_test, y_test, threshold=t)})

pd.DataFrame(test_results).round(3)


# In[106]:


y_test_prob = best_rf.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_prob >= threshold).astype(int)

cm = confusion_matrix(y_test, y_test_pred)
ConfusionMatrixDisplay(cm, display_labels=['not good', 'good']).plot()
plt.show()


# In[107]:


importances = best_rf.feature_importances_
feature_names = dv.get_feature_names_out()

imp = pd.Series(importances, index=feature_names).sort_values()

plt.figure(figsize=(7, 5))
imp.plot(kind='barh', color='skyblue')
plt.xlabel('Feature importance')
plt.title('Random Forest feature importance')
plt.tight_layout()
plt.show()

imp.sort_values(ascending=False).round(3)


# In[108]:


xgb_imp = pd.Series(xgb_final.feature_importances_, index=feature_names).sort_values()

plt.figure(figsize=(7, 5))
xgb_imp.plot(kind='barh', color='orange')
plt.xlabel('Feature importance')
plt.title('XGBoost feature importance')
plt.tight_layout()
plt.show()

xgb_imp.sort_values(ascending=False).round(3)


# # Selecting the best model

# In[109]:


cv_table = pd.DataFrame(cv_rows).sort_values('pr_auc_mean', ascending=False)
print(cv_table.round(3))

best_name = cv_table.iloc[0]['model']
print(f'\nSelected model: {best_name} (highest cross-validated PR-AUC)')


# In[110]:


final_params = grid_search.best_params_
print(final_params)
threshold = 0.32

y_full_train = df_full_train.good.values
full_train_dicts = df_full_train.drop(columns=['good', 'quality']).to_dict(orient='records')

dv_final = DictVectorizer(sparse=False)
X_full_train = dv_final.fit_transform(full_train_dicts)
X_test_final = dv_final.transform(test_dicts)

final_model = RandomForestClassifier(**final_params, random_state=42, n_jobs=-1)
final_model.fit(X_full_train, y_full_train)


# In[111]:


final_scores = evaluate(final_model, X_test_final, y_test, threshold=threshold)
pd.Series(final_scores).round(3)


# In[112]:


output_file = 'model.bin'

with open(output_file, 'wb') as f_out:
    pickle.dump((dv_final, final_model, threshold), f_out)

print(f'Saved to {output_file}')


# Testing the model

# In[113]:


with open(output_file, 'rb') as f_in:
    dv_loaded, model_loaded, threshold_loaded = pickle.load(f_in)

sample = df_test.iloc[5].to_dict()
X_sample = dv_loaded.transform([sample])
prob = model_loaded.predict_proba(X_sample)[0, 1]

print('Probability of good:', round(prob, 3))
print('Prediction:', 'good' if prob >= threshold_loaded else 'not good')
print('Actual:', 'good' if y_test[0] == 1 else 'not good')

