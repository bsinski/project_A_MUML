import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
import warnings
warnings.filterwarnings("ignore")

def conditional_mutual_information(X, Y, Z):
    cmi = 0
    n = len(Z)
    for z in np.unique(Z):
        proba = np.sum(Z == z)/n
        cmi += proba*mutual_info_score(X[Z == z], Y[Z == z])
    return cmi

def interaction_information(X, Y, Z):
    return conditional_mutual_information(X, Y, Z) - mutual_info_score(X, Y)

def jmi(X,Y,j,S):
    n = len(S)
    Xj = X[:,j]
    if n == 0:
        return mutual_info_score(Xj, Y)
    sum_term = 0
    for i in range(n):
        sum_term += interaction_information(Xj, X[:,S[i]] , Y)
    return mutual_info_score(Xj, Y) + sum_term/n

def forward_selection_jmi(X,Y,k):
    n,p = X.shape
    selected_features = []
    features = list(range(p))
    for _ in range(k):
        jmi_scores = np.array([jmi(X, Y,j, selected_features) if j not in selected_features else -1 for j in features])
        next_feature = np.argmax(jmi_scores)
        selected_features.append(next_feature)
    return np.array(selected_features) 

def cife(X,Y,j,S):
    n = len(S)
    Xj = X[:,j]
    if n == 0:
        return mutual_info_score(Xj, Y)
    sum_term = 0
    for i in range(n):
        sum_term += interaction_information(Xj, X[:,S[i]] , Y)
    return mutual_info_score(Xj, Y) + sum_term 

def forward_selection_cife(X,Y,k):
    n,p = X.shape
    selected_features = []
    features = list(range(p))
    for _ in range(k):
        jmi_scores = np.array([cife(X, Y,j, selected_features) if j not in selected_features else -1 for j in features])
        next_feature = np.argmax(jmi_scores)
        selected_features.append(next_feature)
    return np.array(selected_features)

def rfc_selection(X,Y,k):
    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(X,Y)
    return np.argsort(rfc.feature_importances_)[::-1][:k]

def boruta_selection(X,Y):
    np.int = np.int32
    np.float = np.float64
    np.bool = np.bool_
    rfc = RandomForestClassifier(n_estimators=100)
    feature_selector = BorutaPy(rfc, n_estimators='auto', random_state=1)
    feature_selector.fit(X,Y)
    return np.arange(X.shape[1])[feature_selector.support_]
