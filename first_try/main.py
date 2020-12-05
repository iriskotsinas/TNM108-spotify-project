import getHistory
import pandas as pd

def main():

    #recover streamings history
    token = getHistory.get_token()
    
    streamings = getHistory.get_streamings()
    print(f'Recovered {len(streamings)} streamings.')

    track_id = getHistory.get_id('Lucy', token)
    print(track_id)

    lucy_features = getHistory.get_features('28fuXrmmF9dYWx25dMW9dP', token)
    print(lucy_features)

    # unique_tracks = list(set([for streaming in streamings]))

    # unique_tracks = set([f"{streaming[2]}___{streaming[2]}" for streaming in streamings])
    # print(f'Discovered {len(unique_tracks)} unique tracks.')

    # all_features = {}
    # for track in unique_tracks:
    #   track_id = getHistory.get_id(track, token)
    #   features = getHistory.get_features(track_id, token)
    #   if features:
    #     all_features[track] = features
        
    # with_features = []
    # for track_name, features in all_features.items():
    #   with_features.append({'name': track_name, **features})

    # df = pd.DataFrame(with_features)
    # df.to_csv('streaming_history.csv')


if __name__ == '__main__':
    main()












# http://localhost:7777/callback?code=AQC4-DpT-1Y4yj-4cFM5o6Mlx3ON4uKVSHRxT8FsdjdFInDxINp8xd2YLD7cjF4PXPlcG1MmD_xsBxBGPuka0_wSVEptZDuqPgH6C0QWEatMnhovmIVHT-T2snHNJMmBOFEdBO0pRKcZ622lWFUSSISxS4qyD8vez-wx8KErkm0HX5LJjWoW9BV0-SYXVpS3wbwMOVM62LVB_OrxNA