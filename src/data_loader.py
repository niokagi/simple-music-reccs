import pandas as pd
import os

def load_and_clean_data(filepath):
    """
    Loads music dataset from CSV, handles missing values, removes duplicates,
    and selects relevant audio features for the model.
    """
    # validate file existence
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    print("Loading and preprocessing data...")
    df = pd.read_csv(filepath)
    
    # schema init
    if 'genre' in df.columns and 'track_genre' not in df.columns:
        df = df.rename(columns={'genre': 'track_genre'})
    
    # cleaning
    df = df.drop_duplicates(subset=['track_id'])
    df = df.dropna()
    
    # selection columns
    relevant_columns = [
        'track_id', 'artists', 'track_name', 
        'danceability', 'energy', 'valence', 'tempo', 
        'acousticness', 'instrumentalness', 'liveness',
        'track_genre', 
        'popularity',
        'explicit',
        'loudness'
    ]
    
    existing_cols = [col for col in relevant_columns if col in df.columns]
    
    if 'track_genre' not in existing_cols:
        raise ValueError("Critical Error: 'track_genre' column missing in the dataset.")
    
    df_clean = df[existing_cols].reset_index(drop=True)
    
    # convert boolean 'explicit' field to integer (true=1, false=0) for math processing
    if 'explicit' in df_clean.columns:
        df_clean['explicit'] = df_clean['explicit'].astype(int)
        
    return df_clean