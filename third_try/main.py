import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#initialise a client credentials manager
cid ='05607c4ff03849df9d2b0c05e392ab19'
secret = 'b8c59c18f96c4e059e6c3ec544af9e58'
username='1198480425'
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret) 
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
playlists = sp.user_playlists(username)



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
  r = splitlist(track_URIs,2)
  for pack in range(len(r)):
     features = features + (sp.audio_features(r[pack]))
  df = pd.DataFrame.from_dict(features)
  df["uri"] = track_URIs
  return df

def splitlist(track_URIs,step):
    return [track_URIs[i::step] for i in range(step)]

playlist1 = "37i9dQZF1DXaZMjKCB7m2q"
playlist2 = "37i9dQZF1DX7FV7CCq9byu"
list_tracks1 = get_playlist_URIs(username, playlist1)
list_tracks2 = get_playlist_URIs(username, playlist2)

audio_features1 = get_audio_features(list_tracks1)
audio_features2 = get_audio_features(list_tracks2)
print(audio_features1)


# [‘danceability’,’acousticness’,’energy’,’instrumentalness’,’speechiness’,’tempo’,’valence’]
sns.distplot(audio_features1[['acousticness']],color='indianred',axlabel='Tempo')
sns.distplot(audio_features2[['acousticness']],color='mediumslateblue')
plt.show()


#training_data = pd.concat([audio_features1,audio_features2], axis=0, join='outer', ignore_index=True)