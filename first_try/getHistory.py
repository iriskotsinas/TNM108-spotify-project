import ast
from typing import List
from os import listdir
import spotipy.util as util
import requests
import spotipy

def get_token() -> str:
  
  username = '1198480425'
  client_id ='657ab2b1cdb14c95a95677ed1225a328'
  client_secret = '99dd9d7a96b84b0097af3404a3c3fa2f'
  redirect_uri = 'http://localhost:7777/callback'
  scope = 'user-read-recently-played'

  token = util.prompt_for_user_token(username=username, 
                                   scope=scope, 
                                   client_id=client_id,   
                                   client_secret=client_secret,     
                                   redirect_uri=redirect_uri)
  return token

def get_streamings(path: str = 'MyData') -> List[dict]:
    
    files = ['MyData/' + x for x in listdir(path)
             if x.split('.')[0][:-1] == 'StreamingHistory']
    
    all_streamings = []
    
    for file in files: 
        with open(file, 'r', encoding='UTF-8') as f:
            new_streamings = ast.literal_eval(f.read())
            all_streamings += [streaming for streaming 
                               in new_streamings]
    return all_streamings

def get_id(track_name: str, token: str) -> str:
    headers = {
      'Accept': 'application/json',
      'Content-Type': 'application/json',
      'Authorization': f'Bearer ' + token,
    }
    params = [
      ('q', track_name),
      ('type', 'track'),
    ]
    try:
        response = requests.get('https://api.spotify.com/v1/search', 
                    headers = headers, params = params, timeout = 5)
        json = response.json()
        first_result = json['tracks']['items'][0]
        track_id = first_result['id']
        return track_id
    except:
        return None

def get_features(track_id: str, token: str) -> dict:
    sp = spotipy.Spotify(auth=token)
    try:
        features = sp.audio_features([track_id])
        return features[0]
    except:
        return None