import numpy as np
import pandas as pd

def clean_data():
    pass

def extract_emojis(s):
    '''Extract emojis from text'''
    return ''.join(c for c in str(s) if c in emoji.UNICODE_EMOJI)

def age_profiles_prep(age_profiles):
    '''Process age_profiles for modeling'''
    age_profiles_new = pd.concat([age_profiles,
                                  pd.DataFrame(age_profiles['status'].apply(pd.Series) \
                                  .rename(columns = lambda x: 'status_' + str(x)))], axis=1)
    age_profiles_new['status_emoji'] = age_profiles_new['status_text'].map(lambda x: extract_emojis(x))
    age_profiles_new['emoji_cnt'] = age_profiles_new['status_emoji'].map(lambda x: len(x))
    return age_profiles_new

    age_profiles_new.set_index('id', inplace=True)
    ages_new.set_index('ID', inplace=True)
    age_joined = age_profiles_new.join(ages_new, how='inner')
    return age_joined

def design_matrix(age_profiles_new, ages_train):
    '''
    '''


if __name__ == '__main__':
    age_profiles = pd.read_json('assignment_package/age_profiles.json')
