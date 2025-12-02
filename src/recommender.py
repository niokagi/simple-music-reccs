import joblib
import os
import pandas as pd
import numpy as np

class MusicRecommender:
    def __init__(self, model_dir='models/'):
        try:
            self.model = joblib.load(os.path.join(model_dir, 'knn_model.pkl'))
            self.features_final = joblib.load(os.path.join(model_dir, 'features_final.pkl'))
            self.df = joblib.load(os.path.join(model_dir, 'full_dataframe.pkl')) 
        except FileNotFoundError:
            raise FileNotFoundError("Required model artifacts are missing. Please execute the training phase (Option 1) first.")

    def find_song_index(self, song_name):
        exact = self.df[self.df['track_name'].str.lower() == song_name.lower()]
        if not exact.empty:
            return exact.sort_values('popularity', ascending=False).index[0]
        
        contains = self.df[self.df['track_name'].str.contains(song_name, case=False, na=False)]
        if not contains.empty:
            return contains.sort_values('popularity', ascending=False).index[0]
        return None

    def recommend(self, song_name, n_final=10):
        idx = self.find_song_index(song_name)
        if idx is None: 
            return f"Song '{song_name}' was not found in the database."
        
        # candidate generation
        target_vector = self.features_final[idx].reshape(1, -1)
        distances, indices = self.model.kneighbors(target_vector, n_neighbors=50)
        
        # retrieve input 
        input_song = self.df.iloc[idx]
        input_bpm = input_song['tempo']
        input_loudness = input_song['loudness']
        input_genre = input_song['track_genre']
        input_artist = input_song['artists']
        input_title = input_song['track_name'].lower()
        
        candidates = []
        artist_counter = {input_artist: 0} 
        seen_titles = {input_title}

        # filtering & normalized
        for i in range(1, len(indices[0])): 
            neighbor_idx = indices[0][i]
            candidate = self.df.iloc[neighbor_idx]
            cand_artist = candidate['artists']
            cand_title = candidate['track_name'].lower()
            
            # filters
            if cand_title in seen_titles: continue
            if artist_counter.get(cand_artist, 0) >= 2: continue 
            if abs(candidate['tempo'] - input_bpm) > (input_bpm * 0.20): continue
            if abs(candidate['loudness'] - input_loudness) > 4.5: continue 

            # calc
            raw_audio_score = (1 - distances[0][i]) * 100
            weighted_audio = raw_audio_score * 0.90
            weighted_genre = 5.0 if candidate['track_genre'] == input_genre else 0.0
            weighted_artist = 5.0 if cand_artist == input_artist else 0.0
            final_score = weighted_audio + weighted_genre + weighted_artist
            final_score = min(final_score, 100.0)
            
            candidates.append({
                'artist': cand_artist,
                'title': candidate['track_name'],
                'genre': candidate['track_genre'],
                'loudness': candidate['loudness'],
                'score': final_score
            })
            
            artist_counter[cand_artist] = artist_counter.get(cand_artist, 0) + 1
            seen_titles.add(cand_title)
            
        # sorting
        candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
        top_picks = candidates[:n_final]
        
        results = [f"Input: '{input_song['track_name']}' - {input_artist}"]
        results.append(f"Spec: {input_genre} | {input_loudness}dB | {input_bpm:.0f} BPM")
        results.append("============================================================")
        
        if not top_picks: 
            return "No recommendations passed the strict audio filtering criteria."

        for i, song in enumerate(top_picks, 1):
            artist_mark = "★" if song['artist'] == input_artist else ""
            
            results.append(
                f"{i}. {song['title']} - {song['artist']} {artist_mark}\n"
                f"   [Gen: {song['genre']} | Vol: {song['loudness']}dB] -> Match: {song['score']:.1f}%"
            )
            
        return "\n".join(results)