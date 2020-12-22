import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import seaborn as sns
import numpy as np

#initialise a client credentials manager
cid ='05607c4ff03849df9d2b0c05e392ab19'
secret = 'b8c59c18f96c4e059e6c3ec544af9e58'
username='1198480425'
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret) 
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
playlists = sp.user_playlists(username)

# playlists
playlist1 = "1ZmU9yhNg0Oee9LH3E9sOR"
playlist2 = "0xBmP4bvMO7CbnAaKeGvb2"

#features = ['danceability','acousticness','energy','instrumentalness','speechiness','tempo','valence', 'loudness']

def get_playlist_tracks(username, playlist_id):
  tracks_list = []
  results = sp.user_playlist(username, playlist_id,
  fields="tracks,next")
  tracks = results['tracks']
  while tracks:
     tracks_list += [ item['track'] for (i, item) in
     enumerate(tracks['items']) ]
     tracks = sp.next(tracks)
  return tracks_list

def get_playlist_URIs(username, playlist_id):
  return [t["uri"] for t in get_playlist_tracks(username,
  playlist_id)]

#modified get features function
def get_audio_features(track_URIs) :
  features = []
  r = splitlist(track_URIs,5)
  for pack in range(len(r)):
     features = features + (sp.audio_features(r[pack]))
  df = pd.DataFrame.from_dict(features)
  df["uri"] = track_URIs
  return df

def splitlist(track_URIs,step):
    return [track_URIs[i::step] for i in range(step)]

list_tracks1 = get_playlist_URIs(username, playlist1)
list_tracks2 = get_playlist_URIs(username, playlist2)

audio_features_df1 = get_audio_features(list_tracks1)
audio_features_df2 = get_audio_features(list_tracks2)

audio_features_df1.to_csv('christmasSongs.csv')
audio_features_df2.to_csv('ordinarySongs.csv')