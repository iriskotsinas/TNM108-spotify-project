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
#playlists = sp.user_playlists(username)

# playlists
playlist1 = "1ZmU9yhNg0Oee9LH3E9sOR"
playlist2 = "0xBmP4bvMO7CbnAaKeGvb2"

playlist_test1 = "12BOdc8cGvGUJXxrfFYaPw"
playlist_test2 = "65b6BivMuORf2MymcrgmaX"

features = ['danceability','acousticness','energy','instrumentalness','speechiness','tempo','valence', 'loudness']

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

#audio_features_df1.to_csv('christmasSongs.csv')
#audio_features_df2.to_csv('ordinarySongs.csv')

# set label with true or false 
audio_features_df1["target"] = 1
audio_features_df2["target"] = 0

training_data = pd.concat([audio_features_df1,audio_features_df2], axis=0, join='outer', ignore_index=True)

# Extract info and create set with test songs: 
list_tracks_test1 = get_playlist_URIs(username, playlist_test1)
list_tracks_test2 = get_playlist_URIs(username, playlist_test2)
audio_features_df_test1 = get_audio_features(list_tracks_test1)
audio_features_df_test2 = get_audio_features(list_tracks_test2)
audio_features_df_test1["target"] = 1
audio_features_df_test2["target"] = 0
test_data = pd.concat([audio_features_df_test1,audio_features_df_test2], axis=0, join='outer', ignore_index=True)

features = ['danceability','acousticness','energy','instrumentalness','speechiness','tempo','valence', 'loudness']

sns.distplot(audio_features_df1[['acousticness']],color='#D05555',axlabel='Acousticness', label='Christmas songs')
sns.distplot(audio_features_df2[['acousticness']],color='#7CE475', label='Ordinary songs')
plt.legend(loc='upper right')
#plt.show()



score_dt_=[]
score_knn_=[]
score_pca_=[]

# The selected features to use in the model: 
selected_features = ['danceability','acousticness','energy','speechiness']

for i in range(30): 
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
  score_dt = accuracy_score(y_test, y_pred) * 100 
  score_dt_.append(score_dt)
  

  # KNN classifier
  knc = KNeighborsClassifier(5)
  knc.fit(x_train,y_train)
  knn_pred = knc.predict(x_test)
  score_knn = accuracy_score(y_test, knn_pred) * 100
  score_knn_.append(score_knn)
  
  # PCA and Random forest classifier
  sc = StandardScaler()
  X_train = sc.fit_transform(x_train)
  X_test = sc.transform(x_test)
  pca = PCA(n_components=3)
  classifier = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
  classifier.fit(X_train, y_train)
  y_pred = classifier.predict(X_test)
  score_pca = accuracy_score(y_test, y_pred)*100
  score_pca_.append(score_pca)
  


min_m = min(min(score_dt_), min(score_knn_), min(score_pca_))
max_m = max(max(score_dt_), max(score_knn_), max(score_pca_))
print(min_m)
print(max_m)

plt.clf()
plt.plot(score_dt_, label='Decision tree', color = '#3FE603')
plt.plot(score_knn_, label='KNN', color="#2FAD03")
plt.plot(score_pca_, label='Random forest', color="#217802")
plt.plot(np.ones(30)*min_m, '--', color='#A0F980', linewidth=1)
plt.plot(np.ones(30)*max_m, '--', color='#A0F980', linewidth=1)
plt.legend(loc='lower right')
plt.ylabel('Accuracy (%)')
plt.xlabel('Iteration')
plt.ylim(top=100)
plt.ylim(bottom=0) 
plt.show()  

# Test 
x_test_test = test_data[selected_features]
y_test_test = test_data['target']

y_pred_test = dtc.predict(x_test_test)
score_dt_test = accuracy_score(y_test_test, y_pred_test) * 100 # fix this
print('Decision tree classifier: ',score_dt_test)

knn_pred_test = knc.predict(x_test_test)
score_knn_test = accuracy_score(y_test_test, knn_pred_test) * 100
print('KNN Classifier: ', score_knn_test)

X_test_test = sc.transform(x_test_test)
y_pred_test = classifier.predict(X_test_test)
score_pca_test = accuracy_score(y_test_test, y_pred_test)*100
print("Accuracy using the PCA model is: ", score_pca_test, "%")
