import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
import joblib
import os
from .data_loader import load_and_clean_data

DATA_PATH = 'data/dataset.csv'
MODEL_DIR = 'models/'

# data label weighting
# popularity bias neutralized
FEATURE_WEIGHTS = {
    'track_genre': 2.2,    # High 
    'loudness': 2.5,       # CRITICAL: Era Indicator (Older songs are generally quieter/more dynamic vs modern loudness).
    'acousticness': 2.0,   # CRITICAL: Distinguishes between organic/acoustic bands and synthesized music.
    
    # vibe identity
    'valence': 1.8,     
    'energy': 1.5,
    'tempo': 1.5,
    'instrumentalness': 1.2,
    
    'danceability': 1.0,
    'liveness': 0.8,
    'explicit': 3.0,      

    # win weight
    'popularity': 0.1    
}

def train():
    df = load_and_clean_data(DATA_PATH)
    joblib.dump(df, os.path.join(MODEL_DIR, 'full_dataframe.pkl'))
    metadata = df[['track_id', 'artists', 'track_name', 'track_genre', 'popularity']]
    
    print(f"Training model with {len(df)} tracks... (Mode: Pure Audio Focus)")

    # define Feature Groups
    features_audio = ['danceability', 'energy', 'valence', 'tempo', 'acousticness', 'instrumentalness', 'liveness', 'loudness']
    feature_pop = ['popularity']
    feature_explicit = ['explicit']
    feature_genre = ['track_genre']

    # preprocessing
    # audio features (Standard Scaling)
    scaler_audio = StandardScaler()
    X_audio = scaler_audio.fit_transform(df[features_audio])
    
    # popularity (MinMax scaling)
    scaler_pop = MinMaxScaler() 
    X_pop = scaler_pop.fit_transform(df[feature_pop])
    
    # explicit content
    X_expl = df[feature_explicit].values
    
    # genres
    encoder_genre = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_genre = encoder_genre.fit_transform(df[feature_genre])
    
    # apply feature weights
    X_audio_weighted = X_audio * [FEATURE_WEIGHTS[f] for f in features_audio]
    X_pop_weighted = X_pop * FEATURE_WEIGHTS['popularity']
    X_expl_weighted = X_expl * FEATURE_WEIGHTS['explicit']
    X_genre_weighted = X_genre * FEATURE_WEIGHTS['track_genre']
    
    # feature combination
    X_final = np.hstack([X_audio_weighted, X_pop_weighted, X_expl_weighted, X_genre_weighted])
    
    # model Training
    model = NearestNeighbors(n_neighbors=60, algorithm='brute', metric='cosine')
    model.fit(X_final)
    
    # save artifacts
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    print("Saving model artifacts to 'models/' directory...")
    joblib.dump(model, os.path.join(MODEL_DIR, 'knn_model.pkl'))
    joblib.dump(X_final, os.path.join(MODEL_DIR, 'features_final.pkl'))
    
    print("Training Complete. Model is now optimized for audio fidelity over popularity.")

if __name__ == "__main__":
    train()