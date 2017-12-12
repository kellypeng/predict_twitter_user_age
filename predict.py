# Python 3.6
from model import *


if __name__ == '__main__':
    # Use random forest model to redict on final test set
    ages_test = pd.read_csv('assignment_package/ages_test.csv')
    age_profiles = pd.read_json('assignment_package/age_profiles.json')
    ap = feature_engineering(age_profiles)
    ages_test_joined = joined_age(ap, ages_test)
    df_final = ages_test_joined[['emoji_cnt', 'source_android','statuses_count',
             'has_description','followers_count','favourites_count','source_ios',
             'has_emoji', 'older_group_words','profile_background_tile']]

    with open('rf_model.pkl', 'rb') as f:
        rf = pickle.load(f, encoding='utf-8')
    predictions = rf.predict(df_final)
    df_final['predicted'] = predictions
    df_final['predicted'].to_csv('prediction.csv')
