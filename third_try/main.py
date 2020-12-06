import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import seaborn as sns
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

# set label with true or false 
audio_features_df1["target"] = 1
audio_features_df2["target"] = 0

#audio_features_df1.to_csv('streaming_history1.csv')
#audio_features_df2.to_csv('streaming_history2.csv')

training_data = pd.concat([audio_features_df1,audio_features_df2], axis=0, join='outer', ignore_index=True)





# import matplotlib.pyplot as plt

# f, axes = plt.subplots(2, 4, figsize=(7, 7), sharex=True)
# f.suptitle('Vertically stacked subplots')
# plot(x, y)
# axs[1].plot(x, -y)

features = ['danceability','acousticness','energy','instrumentalness','speechiness','tempo','valence']

# for i in range(1,len(features)):
#   sns.distplot(audio_features_df1[[features[i]]],color='g',axlabel=features[i])
#   sns.distplot(audio_features_df2[[features[i]]],color='indianred')

# plt.show()


sns.distplot(audio_features_df1[['danceability']],color='g',axlabel='danceability')
sns.distplot(audio_features_df2[['danceability']],color='indianred')
# fixa legend 
# fixa subplot 

plt.show()

# The selected features to use in the model: 
selectec_features = ['danceability','acousticness','energy','instrumentalness','speechiness', 'valence']


# split the data training / test
train, test = train_test_split(training_data, test_size = 0.2)
x_train = train[selectec_features]
y_train = train['target']
x_test = test[selectec_features]
y_test = test['target']


# Decision tree classifier
dtc = DecisionTreeClassifier()
dt = dtc.fit(x_train,y_train)
y_pred = dtc.predict(x_test)
score_dt = accuracy_score(y_test, y_pred) * 100 # fix this
print('Decision tree classifier: ',score_dt)

# cvs = cross_val_score(dtc, y_pred, y_train, cv=10)
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
