from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

def build_pca(df, n_components=10):
    df_train = df[df['dataset']=="Train"]
    xtrain = df_train.drop(columns=['cancer_type', 'uuid', 'dataset'])
    ytrain = df_train['cancer_type']
    pca = PCA(n_components=min(xtrain.shape[1], n_components))
    pca.fit_transform(xtrain)
    return pca

def build_minmax_scaler(df, mmrange=(-5,5)):
    df_train = df[df['dataset']=='Train']
    xtrain = df_train.drop(columns=['cancer_type', 'uuid', 'dataset'])
    mm = MinMaxScaler(feature_range = mmrange)
    mm.fit(xtrain)
    return mm

def build_standard_scaler(df):
    df_train = df[df['dataset']=='Train']
    xtrain = df_train.drop(columns=['cancer_type', 'uuid', 'dataset'])
    mm = StandardScaler()
    mm.fit(xtrain)
    return mm

def build_logreg(df, pca, mm):
    df_train = df[df['dataset']=="Train"]
    xtrain = df_train.drop(columns=['cancer_type', 'uuid', 'dataset'])
    ytrain = df_train['cancer_type']
    xtrain = pca.transform(xtrain)
    xtrain = mm.transform(xtrain)
    
    logreg = LogisticRegression(max_iter=10000)
    logreg.fit(xtrain, ytrain)
    return logreg

def build_classifier(df, pca, mm, clf_class=LogisticRegression, clf_kws={'max_iter':10000}):
    df_train = df[df['dataset']=="Train"]
    xtrain = df_train.drop(columns=['cancer_type', 'uuid', 'dataset'])
    ytrain = df_train['cancer_type']
    xtrain = pca.transform(xtrain)
    xtrain = mm.transform(xtrain)
    
    model = clf_class(**clf_kws)
    model.fit(xtrain, ytrain)
    return model

def add_predictions(df, pca, logreg, mm):
    x = df.drop(columns=['cancer_type', 'uuid', 'dataset'])
    x = pca.transform(x)
    x = mm.transform(x)
    df = df.assign(logreg_prediction = logreg.predict(x))
    return df