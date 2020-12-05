import functions
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import operator
import numpy as np

def visualization(playList, sp):
  column=('songName', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'valence', 'instrumentalness','tempo')
  n_bins=20
  fig, axs = plt.subplots(figsize=(24, 8), nrows=2, ncols=4)
  axs=axs.flatten()
  dfTrivialList=functions.getTrivialInfo(playList)
  songIds=functions.getSongList(dfTrivialList)
  audioFeatures1 = sp.audio_features(tracks=songIds)
  for song in zip(audioFeatures1):
    if song is None:
        audioFeatures1.remove(song)  
  index=0
  for feature in column[1:]:
      ax=axs[index]
      ax.set_title(feature)
      featureListToPlot1 = []
      for song in zip(audioFeatures1):
          featureListToPlot1.append(song[feature])
      sns.distplot(featureListToPlot1, hist=True, bins=n_bins, color = 'red', hist_kws={'edgecolor':'red'}, kde_kws={'linewidth': 4}, ax=ax)
      index+=1