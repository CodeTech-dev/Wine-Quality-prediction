import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

RED_PATH = 'data/winequality-red.csv'
WHITE_PATH = 'data/winequality-white.csv'
OUTPUT_FILE = 'model.bin'
THRESHOLD = 0.32


def load_data():
    red = pd.read_csv(RED_PATH, sep=';')
    white = pd.read_csv(WHITE_PATH, sep=';')
    red['wine_type'] = 'red'
    white['wine_type'] = 'white'

    df = pd.concat([red, white], ignore_index=True)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df = df.drop_duplicates().reset_index(drop=True)
    df['good'] = (df.quality >= 7).astype(int)
    return df


def prepare(df):
    y = df.good.values
    features = df.drop(columns=['good', 'quality'])
    return features.to_dict(orient='records'), y


def train(dicts, y):
    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(dicts)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return dv, model


if __name__ == '__main__':
    df = load_data()

    df_full_train, df_test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df.good)

    train_dicts, y_train = prepare(df_full_train)
    test_dicts, y_test = prepare(df_test)

    print('Training the final model on train + validation data...')
    dv, model = train(train_dicts, y_train)

    X_test = dv.transform(test_dicts)
    y_prob = model.predict_proba(X_test)[:, 1]
    print(f'Test ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}')
    print(f'Test PR-AUC:  {average_precision_score(y_test, y_prob):.3f}')

    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump((dv, model, THRESHOLD), f)

    print(f'Model saved to {OUTPUT_FILE}')