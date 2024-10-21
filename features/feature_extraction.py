import pandas as pd

def extract_features(data):
    features = data[['H_and_X_ar', 'electronegativity_Pauling_pow2']]
    return features

def extract_target(data):
    target = data[['H_length']]
    return target


