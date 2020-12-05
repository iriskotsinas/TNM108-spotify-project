import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
# import visualize
import functions
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import operator
import numpy as np

client_id ='05607c4ff03849df9d2b0c05e392ab19'
client_secret = 'b8c59c18f96c4e059e6c3ec544af9e58'
redirect_uri = 'http://localhost:7777/callback'

username='1198480425'
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret) 
scope = 'user-library-read playlist-read-private'
try:
    token = util.prompt_for_user_token(username, scope,client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri)  
    sp=spotipy.Spotify(auth=token)
except:
    print('Token is not accesible for ' + username)

playlistDictionary={"Get Turnt": "37i9dQZF1DX5GuMRZBomNE",
                   "Classical Essentials": "37i9dQZF1DX5GuMRZBomNE", #37i9dQZF1DXaXB8fQg7xif
                    "Rock Save the Queen" : "37i9dQZF1DX5GuMRZBomNE",
                   "Coffee Table Jazz" : "37i9dQZF1DX5GuMRZBomNE"
                   }

getTurntId=playlistDictionary["Get Turnt"]
dancePartyId=playlistDictionary["Classical Essentials"]
jazzyRomanceId=playlistDictionary["Coffee Table Jazz"]
rockSaveTheQueenId=playlistDictionary["Rock Save the Queen"]
spotifyUsername='Spotify'
getTurnt=sp.user_playlist(spotifyUsername, playlist_id=getTurntId)
danceParty=sp.user_playlist(spotifyUsername, playlist_id=dancePartyId)
jazzyRomance=sp.user_playlist(spotifyUsername, playlist_id=jazzyRomanceId)
rockSaveTheQueen=sp.user_playlist(spotifyUsername, playlist_id=rockSaveTheQueenId)

column=('songName', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'valence', 'instrumentalness','tempo')
playlistList=[getTurnt, danceParty, jazzyRomance, rockSaveTheQueen]
n_bins=20
fig, axs = plt.subplots(figsize=(24, 8), nrows=2, ncols=4)
axs=axs.flatten()
dfTrivialList=functions.getTrivialInfo(playlistList)
songIdsList=functions.getSongList(dfTrivialList)
songIds1=songIdsList[0]
songIds2=songIdsList[1]
songIds3=songIdsList[2]
songIds4=songIdsList[3]
audioFeatures1 = sp.audio_features(tracks=songIds1)
audioFeatures2 = sp.audio_features(tracks=songIds2)
audioFeatures3 = sp.audio_features(tracks=songIds3)
audioFeatures4 = sp.audio_features(tracks=songIds4)
for song1, song2, song3, song4 in zip(audioFeatures1, audioFeatures2, audioFeatures3, audioFeatures4):
    if song1 is None:
        audioFeatures1.remove(song1)
    if song2 is None:
        audioFeatures2.remove(song2)
    if song3 is None:
        audioFeatures3.remove(song3)
    if song4 is None:
        audioFeatures4.remove(song4)
            
index=0
for feature in column[1:]:
    ax=axs[index]
    ax.set_title(feature)
    featureListToPlot1 = []
    featureListToPlot2 = []
    featureListToPlot3 = []
    featureListToPlot4 = []
    for song1, song2, song3, song4 in zip(audioFeatures1, audioFeatures2, audioFeatures3, audioFeatures4):
        featureListToPlot1.append(song1[feature])
        featureListToPlot2.append(song2[feature])
        featureListToPlot3.append(song3[feature])
        featureListToPlot4.append(song4[feature])
    sns.histplot(featureListToPlot1, bins=n_bins, color = 'red', ax=ax)
    sns.histplot(featureListToPlot2, bins=n_bins, color = 'blue', ax=ax)
    sns.histplot(featureListToPlot3, bins=n_bins, color = 'orange', ax=ax)
    sns.histplot(featureListToPlot4, bins=n_bins, color = 'yellow', ax=ax)
    index+=1
    
plt.show()

# songLibrary = sp.current_user_saved_tracks()
# playlist = sp.user_playlist(username, playlist_id='0yCSEXNXnGEVzf030s93ps')
# playlists = sp.user_playlists(username)

# #0yCSEXNXnGEVzf030s93ps 456yv9FB1Tx5z0OFDTsAU9
# # visualize.visualization(playlist, sp)


# column=('songName', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'valence', 'instrumentalness','tempo')
# n_bins=20
# fig, axs = plt.subplots(figsize=(24, 8), nrows=2, ncols=4)
# axs=axs.flatten()
# dfTrivialList=functions.getTrivialInfo(playlist)
# songIds=functions.getSongList(dfTrivialList)
# audioFeatures1 = sp.audio_features(tracks=songIds)
# # audioFeatures1 = sp.audio_features([songIds[0]])
# for song in zip(audioFeatures1):
#   if song is None:
#       audioFeatures1.remove(song) 
# # for song in songIds:
# #   try:
# #     audioFeatures1 = sp.audio_features([song])
# #   except:
# #     audioFeatures1.remove(song) 

# index=0
# for feature in column[1:]:
#     ax=axs[index]
#     ax.set_title(feature)
#     featureListToPlot1 = []
#     for song in zip(audioFeatures1):
#         featureListToPlot1.append(song[feature])
#     sns.distplot(featureListToPlot1, hist=True, bins=n_bins, color = 'red', hist_kws={'edgecolor':'red'}, kde_kws={'linewidth': 4}, ax=ax)
#     index+=1