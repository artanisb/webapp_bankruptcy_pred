import joblib as jl
import tensorflow as tf
import numpy as np
from bfl import BinaryFocalLoss
from featfactory import FeatureEngineer

# Generiert Features die für die Erstellung der Featur-Presets benötigt werden.
def fill_data(s):
    s['X16'] = s['X9']
    s['X4'] = s['X12'] + s['X3']
    s['X13'] = s['X9'] - s['X2']
    return s

# Baut Feature-Presets basierend auf der Modell-Konfuguration.
# Löscht dabei die drei vorher generierten Features.
def build_features(features, s):
    fe = FeatureEngineer()
    fe.build_dfs(s)
    s = fe.generate_features(features)
    return s

# Skaliert die Features basierend auf dem Scaler der Trainingsdaten.
def scale(scaled, s):
    if scaled['scaled']:
        scalers = jl.load('frontend/GRU_scalers.pkl')
        s_scaled = s.copy()
        for col in s.columns:
            if col in scalers:
                s_scaled[col] = scalers[col].transform(s[[col]].values).flatten()
        return s_scaled
    else:
        return s

# Erstellt Sequences aus den Rohdaten, damit diese in das GRU gegeben werden können.
def build_sequence(s, year_col='year'):
    feature_cols = [c for c in s.columns if c != year_col]
    s_sorted = s.sort_values(year_col)
    features = s_sorted[feature_cols].values
    return features[np.newaxis, :, :]

# Konfiguriert die Loss-Funktion und lädt das Model.
def build_gru_model(loss):
    gamma = loss['gamma']
    alpha = loss['alpha']
    model = tf.keras.models.load_model(f"frontend/GRU_model.h5", custom_objects={'BinaryFocalLoss': BinaryFocalLoss(gamma=gamma, alpha=alpha)})
    return model

# Führt die Vorhersage anhand des Modells aus.
def predict(model, s):
    prediction = model.predict(s, verbose=0)
    return prediction

# Hyperparameter zur Feature-Erstellung und Modellkonfiguration
hp_gru = {
            'lag1': False,
            'lag2': False,
            'lag3': True,
            'sum3': False,
            'avg3': True,
            'flags': False,
            'ratios': False,
            'oscore': False,
            'ocomp': True,
            'zscore': True,
            'zcomp': True,
            'gamma': 2,
            'alpha': 0.2,
            'scaled': True,
}

# Liest die Hyperparameter aus und speichert sie in Dictonaries.
def get_configs(hp):
    f_keys = ['lag1', 'lag2', 'lag3', 'sum3', 'avg3', 'flags', 'ratios', 'oscore', 'ocomp', 'zscore', 'zcomp']
    l_keys = ['gamma', 'alpha']
    a_keys = ['scaled']
    f = {k: hp[k] for k in f_keys}
    l = {k: hp[k] for k in l_keys}
    s = {k: hp[k] for k in a_keys}
    return f, l, s

# Wrapper-Funktion die im Frontend aufgerufen wird. 
def compute_score(hp, sample):
    features, loss, scaled = get_configs(hp)
    sample = fill_data(sample)
    sample = build_features(features, sample)
    sample = scale(scaled, sample)
    sample_for_model = build_sequence(sample)
    model = build_gru_model(loss)
    pred = predict(model, sample_for_model)
    return pred