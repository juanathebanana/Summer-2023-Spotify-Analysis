import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import lyricsgenius

# Set up Spotify client
client_id = "YOUR CLIENT ID "
client_secret = "YOUR CLIENT SECRET ID"
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)

# Set up Genius client
genius_access_token = "ACCES TOKEN "
genius = lyricsgenius.Genius(genius_access_token)

sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Get tracks from the Global Top 50 playlist
playlist_id = 'LINK OF THE PLAYLIST'
results = sp.playlist(playlist_id)

tracks = results['tracks']
track_info = [{'id': track['track']['id'],
               'name': track['track']['name'],
               'artist': track['track']['artists'][0]['name']}
               for track in tracks['items']]

# Get audio features, lyrics and genres for these tracks
data = []
for track in track_info:
    features = sp.audio_features([track['id']])[0]
    song = genius.search_song(track['name'], track['artist'])
    if song:
        lyrics = song.lyrics
    else:
        lyrics = None
    artist = sp.search(track['artist'], type='artist')
    genres = artist['artists']['items'][0]['genres']
    track_data = track.copy()
    track_data.update(features)
    track_data['lyrics'] = lyrics
    track_data['genres'] = genres
    data.append(track_data)

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('spotify_data1.csv', index=False)#data won't be shares in this repostory due to spotify legal terms
