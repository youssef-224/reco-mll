import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, jsonify, request

# Set up Spotify API credentials
client_id = 'f0ab2ded1b8a482188532d1f3441d07c'
client_secret = 'db0e7338b4ed487b91b48654b93dd9ec'
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)





# Define a function to get recommended tracks for an artist
def get_recommendations(artist_name):
    # Collect data for the artist's top tracks
    results = sp.search(q='artist:' + artist_name, type='artist')
    items = results['artists']['items']
    if len(items) == 0:
        return None
    artist_id = items[0]['id']
    top_tracks = sp.artist_top_tracks(artist_id)['tracks']
    tracks = []
    for track in top_tracks:
        track_info = sp.track(track['id'])
        tracks.append([artist_name, track['id'], track['name'], track['popularity'], track['album']['name'], track['album']['release_date'],track_info['album']['images'][0]['url'],track_info['id'],track_info['preview_url']])
    # Store data in a Pandas dataframe
    df = pd.DataFrame(tracks, columns=['artist', 'id', 'name', 'popularity', 'album', 'release_date', 'image_url','track_id','preview_url'])
    df.set_index('id', inplace=True)
    # Feature engineering
    df['year'] = pd.DatetimeIndex(df['release_date']).year
    df['decade'] = (df['year'] // 10) * 10
    df['popularity_scaled'] = StandardScaler().fit_transform(df[['popularity']])
    df['decade_encoded'] = pd.factorize(df['decade'])[0]
    # Build a recommendation model using content-based filtering
    features = ['popularity_scaled', 'decade_encoded']
    X = df[features].values
    similarity_matrix = cosine_similarity(X)
    
    
    # Get recommended tracks for the artist
    artist_tracks = df[df['artist'] == artist_name]
    if len(artist_tracks) == 0:
        return None
    else:
        track_ids = artist_tracks.index.values
        track_indices = [list(df.index).index(id) for id in track_ids]
        similar_tracks_indices = similarity_matrix[track_indices].argsort()[:, ::-1][:, 1:4].flatten()
        similar_track_ids = [list(df.index)[i] for i in similar_tracks_indices]
        similar_tracks = df.loc[similar_track_ids]
        return similar_tracks[['artist', 'name', 'popularity', 'album', 'release_date', 'image_url','track_id','preview_url']].to_dict('records')
    
    
    
app = Flask(__name__)

@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    # Get artists from query parameters
    artist_names = request.args.get('artist').split(',')
    # Collect data for each artist's top tracks and store in a list
    tracks = []
    for artist_name in artist_names:
        results = sp.search(q='artist:' + artist_name, type='artist')
        items = results['artists']['items']
        if len(items) > 0:
            artist_id = items[0]['id']
            top_tracks = sp.artist_top_tracks(artist_id)['tracks']
            for track in top_tracks:
                track_info = sp.track(track['id'])
                tracks.append([artist_name, track['id'], track['name'], track['popularity'], track['album']['name'], track['album']['release_date'],track_info['album']['images'][0]['url'],track_info['id'],track_info['preview_url']])
    if len(tracks) == 0:
        return jsonify({'error': 'No tracks found for these artists.'}), 404
    # Store data in a Pandas dataframe
    df = pd.DataFrame(tracks, columns=['artist', 'id', 'name', 'popularity', 'album', 'release_date', 'image_url','track_id','preview_url'])
    df.set_index('id', inplace=True)
    # Feature engineering
    df['year'] = pd.DatetimeIndex(df['release_date']).year
    df['decade'] = (df['year'] // 10) * 10
    df['popularity_scaled'] = StandardScaler().fit_transform(df[['popularity']])
    df['decade_encoded'] = pd.factorize(df['decade'])[0]
    # Build a recommendation model using content-based filtering
    features = ['popularity_scaled', 'decade_encoded']
    X = df[features].values
    similarity_matrix = cosine_similarity(X)
    
    # Get recommended tracks for the artists
    recommended_tracks = []
    for artist_name in artist_names:
        artist_tracks = df[df['artist'] == artist_name]
        if len(artist_tracks) > 0:
            track_ids = artist_tracks.index.values
            track_indices = [list(df.index).index(id) for id in track_ids]
            similar_tracks_indices = similarity_matrix[track_indices].argsort()[:, ::-1][:, 1:4].flatten()
            similar_track_ids = [list(df.index)[i] for i in similar_tracks_indices]
            similar_tracks = df.loc[similar_track_ids]
            recommended_tracks.extend(similar_tracks[['artist', 'name', 'popularity', 'album', 'release_date', 'image_url','track_id','preview_url']].to_dict('records'))
    if len(recommended_tracks) == 0:
        return jsonify({'error': 'No recommendations found for these artists.'}), 404
    else:
        return jsonify(recommended_tracks)