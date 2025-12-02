# Simple Music Recommendation

An experimental project exploring music recommendations based on "Audio DNA" rather than just popularity
Most recommendation systems are heavily biased towards what's trending. This project takes a different approach: it treats music as **mathematical data**.

By analyzing intrinsic audio features (like `loudness`, `energy`, `valence`, and `acousticness`), this engine attempts to find songs that *feel* the same, regardless of whether they are global hits or hidden gems. It uses a **Weighted K-Nearest Neighbors (KNN)** algorithm with a custom logic layer to ensure the recommendations make sense musically and historically.

## Getting Started

Follow these steps to get the engine running on your local machine.

### 1. Prerequisites
Make sure you have Python installed. Then, install the required libraries:

```bash
pip install -r requirements.txt
```

### 2. Prepare Data
Download the **Spotify Tracks Dataset** [here(kaggle)](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) and place the `dataset.csv` file inside the `data/` folder.

### 3. Train the Model
Before asking for recommendations, the system needs to learn the audio patterns. Run the app and select **Option 1**:

```bash
python main.py
```
> Select: **1. Train New Model**

### 4. Get Recommendations
Once trained, you can start exploring! Run the app again and select **Option 2**:

```bash
python main.py
```
> Select: **2. Generate Music Recommendations**
> Input: *Psychosocial* (or any song you like)

## Visual Analysis
Curious about how the data looks? Check out `notebooks/exploration.ipynb` to see Radar Charts comparing the "Audio DNA" of different tracks.

