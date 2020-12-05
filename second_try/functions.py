import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np

def getTrivialInfo(playlistList):
    dfTrivialList=[]
    for playlist in playlistList:
        index=0
        dfTrivial = pd.DataFrame(columns=('SongName', 'SongId', 'SongArtist'), index=np.arange(0, len(playlist['tracks']['items'])))
        for item in playlist['tracks']['items']:
            track = item['track']
            dfTrivial.loc[index]=[track['name'], track['id'], track['artists'][0]['name']]
            index+=1
        dfTrivialList.append(dfTrivial)
    return dfTrivialList

def getSongList(dfTrivialList):
    songIdsList=[]
    for dfTrivial in dfTrivialList:
        songIds=list(dfTrivial['SongId'])
        songIdsList.append(songIds)
    return songIdsList

def getFeaturesList(dfTrivialList, songIdsList, sp, columns=('SongName', 'Danceability', 'Energy', 'Loudness', 'Speechiness', 'Acousticness', 'Valence', 'Instrumentalness','Tempo')):
    dfFeaturesList=[]
    for dfTrivial, songIds in zip(dfTrivialList, songIdsList):
        index=0
        audioFeatures = sp.audio_features(tracks=songIds)
        categories=columns[1:]
        dfFeatures = pd.DataFrame(columns=columns, index=np.arange(0, len(songIds)))
        for i, song in enumerate(audioFeatures):
            dfFeatures.loc[index]=[list(dfTrivial['SongName'])[i], song['danceability'], song['energy'], song['loudness'], song['speechiness'], song['acousticness'], song['valence'], song['instrumentalness'], song['tempo']]
            index+=1
        dfFeaturesList.append(dfFeatures)
    return dfFeaturesList

def getFeaturesToUse(dfFeaturesList, categories=['Danceability','Energy', 'Speechiness', 'Acousticness', 'Valence']):
    featuresToUseList=[]
    for dfFeatures in dfFeaturesList:
        features = dfFeatures[categories]
        featuresToUseList.append(features)
    return featuresToUseList

def featurePreprocessing(song, categories=['Danceability','Energy', 'Speechiness', 'Acousticness', 'Valence']):
    return song[categories]


def get_features(track_id, token):
    sp = spotipy.Spotify(auth=token)
    try:
        features = sp.audio_features([track_id])
        return features[0]
    except:
        return None


def euclideanDistance(data1, data2, weight, length):
    distance = 0
    for x in range(length):
        distance += np.square(weight[x]*(data1[x] - data2[x]))
    return np.sqrt(distance)

def distances(trainingSet, testSong, weight):
    distanceDict = {}
    length = testSong.shape[0]
    for genre, features in trainingSet.items():
        dist = [[euclideanDistance(features.iloc[x], testSong, weight, length), x] for x in range(len(features))]
        distanceDict[genre] = dist
    return distanceDict 

def knn(sortedDistances, k):
    counter={}
    for key in sortedDistances.keys():
        counter[key] =0 
    minKey=''
    minId=0
    neighborKeyAndId=[]
    for i in range(k):
        minValue=5
        for key, value in sortedDistances.items():
            if value[0][0]<minValue:
                minId=value[0][1]
                minKey=key
                minValue = value[0][0]
        del(sortedDistances[minKey][0])
        counter[minKey]=counter[minKey]+1
        neighborKeyAndId.append([minKey, minId])
    return counter, neighborKeyAndId         