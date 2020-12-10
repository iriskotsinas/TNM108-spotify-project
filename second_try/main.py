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
import pandas as pd

client_id ='05607c4ff03849df9d2b0c05e392ab19'
client_secret = 'b8c59c18f96c4e059e6c3ec544af9e58'
redirect_uri = 'http://localhost:7777/callback'

username='1198480425'
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret) 
scope = 'user-library-read playlist-read-private'
try:
    token = util.prompt_for_user_token(username, scope,client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri)  
    sp = spotipy.Spotify(auth=token)
except:
    print('Token is not accesible for ' + username)

playlistDictionary={"God Jul": "37i9dQZF1DXaZMjKCB7m2q",
                   "Julklassiker": "37i9dQZF1DWStE8VkQFPzG", #37i9dQZF1DXaXB8fQg7xif
                    "HipHop" : "37i9dQZF1DX5cpU86I7OAy",
                   "Pop" : "37i9dQZF1DX7FV7CCq9byu"
                   }

getTurntId=playlistDictionary["God Jul"]
dancePartyId=playlistDictionary["Julklassiker"]
jazzyRomanceId=playlistDictionary["HipHop"]
rockSaveTheQueenId=playlistDictionary["Pop"]
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
# for feature in column[1:]:
#     ax=axs[index]
#     ax.set_title(feature)
#     featureListToPlot1 = []
#     featureListToPlot2 = []
#     featureListToPlot3 = []
#     featureListToPlot4 = []
#     for song1, song2, song3, song4 in zip(audioFeatures1, audioFeatures2, audioFeatures3, audioFeatures4):
#         featureListToPlot1.append(song1[feature])
#         featureListToPlot2.append(song2[feature])
#         featureListToPlot3.append(song3[feature])
#         featureListToPlot4.append(song4[feature])
#     sns.histplot(featureListToPlot1, bins=n_bins, color = 'red', ax=ax)
#     sns.histplot(featureListToPlot2, bins=n_bins, color = 'blue', ax=ax)
#     sns.histplot(featureListToPlot3, bins=n_bins, color = 'orange', ax=ax)
#     sns.histplot(featureListToPlot4, bins=n_bins, color = 'yellow', ax=ax)
#     index+=1
#plt.show()
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
    sns.distplot(featureListToPlot1, hist=True, bins=n_bins, color = 'red', hist_kws={'edgecolor':'red'}, kde_kws={'linewidth': 4}, ax=ax)
    sns.distplot(featureListToPlot2, hist=True, bins=n_bins, color = 'blue', hist_kws={'edgecolor':'blue'}, kde_kws={'linewidth': 4}, ax=ax)
    sns.distplot(featureListToPlot3, hist=True, bins=n_bins, color = 'orange', hist_kws={'edgecolor':'orange'}, kde_kws={'linewidth': 4}, ax=ax)
    sns.distplot(featureListToPlot4, hist=True, bins=n_bins, color = 'yellow', hist_kws={'edgecolor':'yellow'}, kde_kws={'linewidth': 4}, ax=ax)
    index+=1    
# show plot 
plt.show()



categories = column[1:]
tempoFeaturesTogether=[]
loudnessFeaturesTogether=[]

for song1, song2, song3, song4 in zip(audioFeatures1, audioFeatures2, audioFeatures3, audioFeatures4):
    tempoFeaturesTogether.append(song1['tempo'])
    tempoFeaturesTogether.append(song2['tempo'])
    tempoFeaturesTogether.append(song3['tempo'])
    tempoFeaturesTogether.append(song4['tempo'])
    loudnessFeaturesTogether.append(song1['loudness'])
    loudnessFeaturesTogether.append(song2['loudness'])
    loudnessFeaturesTogether.append(song3['loudness'])
    loudnessFeaturesTogether.append(song4['loudness'])
    
minimumTempo=min(tempoFeaturesTogether)
maximumTempo=max(tempoFeaturesTogether)
minimumLoudness=min(loudnessFeaturesTogether)
print(minimumLoudness)
maximumLoudness=max(loudnessFeaturesTogether)
print(maximumLoudness)

for song1, song2, song3, song4 in zip(audioFeatures1, audioFeatures2, audioFeatures3, audioFeatures4):
    song1['tempo']=(song1['tempo']-minimumTempo)/(maximumTempo-minimumTempo)
    song2['tempo']=(song2['tempo']-minimumTempo)/(maximumTempo-minimumTempo)
    song3['tempo']=(song3['tempo']-minimumTempo)/(maximumTempo-minimumTempo)
    song4['tempo']=(song4['tempo']-minimumTempo)/(maximumTempo-minimumTempo)
    song1['loudness']=(song1['loudness']-minimumLoudness)/(maximumLoudness-minimumLoudness)
    song2['loudness']=(song2['loudness']-minimumLoudness)/(maximumLoudness-minimumLoudness)
    song3['loudness']=(song3['loudness']-minimumLoudness)/(maximumLoudness-minimumLoudness)
    song4['loudness']=(song4['loudness']-minimumLoudness)/(maximumLoudness-minimumLoudness)

dfFeaturesList=[]
audioFeaturesList=[audioFeatures1, audioFeatures2, audioFeatures3, audioFeatures4]

for audioFeatures in audioFeaturesList:
    dfFeatures = pd.DataFrame(columns=categories, index=np.arange(0, len(audioFeatures)))
    for i, song in enumerate(audioFeatures[:99]):
        print(song['loudness'])
        dfFeatures.loc[i]=[song['danceability'], song['energy'], song['loudness'], song['speechiness'], song['acousticness'], song['valence'], song['instrumentalness'], song['tempo']]
    
    dfFeaturesList.append(dfFeatures)

for i in range(len(dfFeaturesList)):
    dfFeaturesList[i]=dfFeaturesList[i].mean()

dfFeaturesList=pd.concat(dfFeaturesList, axis=1)

# ------ 

N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

fig = plt.figure(figsize=(8,8))
ax = plt.subplot(111, polar=True)
ax.set_theta_offset(pi)
ax.set_theta_direction(-1)
 
plt.xticks(angles, categories)

ax.set_rlabel_position(0)
plt.yticks([0, 0.5, 1], ["0", "0.5", "1"], color="grey", size=7)
plt.ylim(0, 1)
 

# Ind1
colors=['r', 'b', 'orange', 'y']
for i, key in enumerate(playlistDictionary.keys()):
    if i <4:
        values=list(dfFeaturesList[i])
        print(values)
        values += values[:1]
        ax.plot(angles, values, color=colors[i], linewidth=1, linestyle='solid', label=key)
 
# Add legend
plt.legend(bbox_to_anchor=(0.1, 0.1))
# ----- Map ------
plt.show()

playlistList=[getTurnt, danceParty, rockSaveTheQueen, jazzyRomance]

dfTrivialList=functions.getTrivialInfo(playlistList)
songIdsList=functions.getSongList(dfTrivialList)
dfFeaturesList=functions.getFeaturesList(dfTrivialList, songIdsList, sp, columns=('SongName', 'Danceability', 'Energy', 'Loudness', 'Speechiness', 'Acousticness', 'Valence', 'Instrumentalness','Tempo'))
print(dfFeaturesList[1].iloc[48])
featuresToUseList=functions.getFeaturesToUse(dfFeaturesList, categories=['Danceability','Energy', 'Speechiness', 'Acousticness', 'Valence','Instrumentalness'])
print(featuresToUseList[1])

f1= featuresToUseList[0][:35]
f2= featuresToUseList[1][:35]
f3= featuresToUseList[2][:35]
f4= featuresToUseList[3][:35]
trainingSet = {'1': f1, '2': f2, '3': f3, '4': f4}
for i in range(len(featuresToUseList)):
    print(i)
    accurate=0
    total=0
    for j in range(35,50):
        distanceList=functions.distances(trainingSet, featuresToUseList[i].iloc[j], [1,1,1,1,1,1])
        print(dfFeaturesList[i].iloc[j])
        sortedDict={}
        for key in distanceList.keys():
            sortedDict[key]=sorted(distanceList[key], key=operator.itemgetter(0))
        counter, neighborKeyAndId = functions.knn(sortedDict, 25)
        prediction=max(counter.items(), key=operator.itemgetter(1))[0]
        if(i+1 == int(prediction)):
            accurate+=1
        total+=1
        print(counter)
    print(float(accurate/total))


