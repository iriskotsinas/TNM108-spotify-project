import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import seaborn as sns
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

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

playlistDictionary={"Christmas": "1ZmU9yhNg0Oee9LH3E9sOR",
                    "Other genres": "0xBmP4bvMO7CbnAaKeGvb2" 
                  }

features = ['danceability','acousticness','energy','instrumentalness','speechiness','tempo','valence', 'loudness']
column=('songName', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'valence', 'instrumentalness','tempo')

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

def add_to_star(playlist, color, label=None):
    values = df2.loc[playlist].tolist()
    values += values[:1]
    if label != None:
        ax.plot(angles, values, color=color, linewidth=1, label=label)
    else:
        ax.plot(angles, values, color=color, linewidth=1, label=playlist)
    ax.fill(angles, values, color=color, alpha=0.25)


list_tracks1 = get_playlist_URIs(username, playlist1)
list_tracks2 = get_playlist_URIs(username, playlist2)

audio_features_df1 = get_audio_features(list_tracks1)
audio_features_df2 = get_audio_features(list_tracks2)

audio_features_df1.to_csv('christmasSongs.csv')
audio_features_df2.to_csv('ordinarySongs.csv')

# dfFeaturesList=[]
# audioFeaturesList=[audio_features_df1, audio_features_df2]
# categories = column[1:]

# for audioFeatures in audioFeaturesList:
#     dfFeatures = pd.DataFrame(columns=categories, index=np.arange(0, len(audioFeatures)))
#     for i, song in enumerate(audioFeatures[:99]):
#         print(song[1])
#         print(audioFeatures[1])
#         dfFeatures.loc[i]=[song['danceability'], song['energy'], song['loudness'], song['speechiness'], song['acousticness'], song['valence'], song['instrumentalness'], song['tempo']]
    
#     dfFeaturesList.append(dfFeatures)


# for i in range(len(dfFeaturesList)):
#     dfFeaturesList[i]=dfFeaturesList[i].mean()
#     # print(dfFeaturesList[i])

# dfFeaturesList=pd.concat(dfFeaturesList, axis=1)

# N = len(categories)
# angles = [n / float(N) * 2 * pi for n in range(N)]
# angles += angles[:1]

# fig = plt.figure(figsize=(8,8))
# ax = plt.subplot(111, polar=True)
# ax.set_theta_offset(pi)
# ax.set_theta_direction(-1)
 
# plt.xticks(angles, categories)

# ax.set_rlabel_position(0)
# plt.yticks([0, 0.5, 1], ["0", "0.5", "1"], color="grey", size=7)
# plt.ylim(0, 1)

# colors=['r', 'b']
# for i, key in enumerate(playlistDictionary.keys()):
#     if i < 4:
#         values=list(audioFeaturesList[i])
#         print(values)
#         values += values[:1]
#         ax.plot(angles, values, color=colors[i], linewidth=1, linestyle='solid', label=key)
 
# Add legend
# plt.legend(bbox_to_anchor=(0.1, 0.1))
# # ----- Map ------
# plt.show()

# set label with true or false 
audio_features_df1["target"] = 1
audio_features_df2["target"] = 0



## Convert all rankings and contiguous data to scale between 0-100
# new_max = 100
# new_min = 0
# new_range = new_max - new_min
# ## Create Scaled Columns
# for factor in features:
#     max_val = df[factor].max()
#     min_val = df[factor].min()
#     val_range = max_val - min_val
#     df[factor + '_Adj'] = df[factor].apply(lambda x: (((x - min_val) * new_range) / val_range) + new_min)

# points = len(features)
# angles = np.linspace(0, 2 * np.pi, points, endpoint=False).tolist()
# angles += angles[:1]

# ## Create plot object   
# fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
# ## Plot a new diamond with the add_to_star function
# add_to_star(27749, '#1aaf6c', "Most Expensive Diamond")
# add_to_star(0, '#429bf4', "Least Expensive A")

training_data = pd.concat([audio_features_df1,audio_features_df2], axis=0, join='outer', ignore_index=True)


features = ['danceability','acousticness','energy','instrumentalness','speechiness','tempo','valence', 'loudness']

sns.distplot(audio_features_df1[['loudness']],color='#D05555',axlabel='Loudness')
sns.distplot(audio_features_df2[['loudness']],color='#7CE475')
# fixa legend 
# fixa subplot 

plt.show()

# The selected features to use in the model: 
selected_features = ['danceability','acousticness','energy','instrumentalness','speechiness', 'valence']



# split the data training / test
train, test = train_test_split(training_data, test_size = 0.2)
x_train = train[selected_features]
y_train = train['target']
x_test = test[selected_features]
y_test = test['target']


# Decision tree classifier
dtc = DecisionTreeClassifier()
dt = dtc.fit(x_train,y_train)
y_pred = dtc.predict(x_test)
score_dt = accuracy_score(y_test, y_pred) * 100 # fix this
print('Decision tree classifier: ',score_dt)

# cvs = cross_val_score(dtc, test, test.target, cv=10)
# print(cvs)

# how to select K ??
knc = KNeighborsClassifier(5)
knc.fit(x_train,y_train)
knn_pred = knc.predict(x_test)
score_knn = accuracy_score(y_test, knn_pred) * 100
print('KNN Classifier: ', score_knn)

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)
pca = PCA(n_components=3)
classifier = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("Accuracy using the PCA model is: ", accuracy_score(y_test, y_pred)*100, "%")
