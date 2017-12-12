# Python 3.6
import emoji
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler


def clean_data(ages_train):
    """
    Remove users age older than 80

    Input: pandas dataframe
    Output: pandas dataframe
    """
    ages_train = ages_train[ages_train['Age']<80]
    return ages_train


def extract_emojis(s):
    """
    Extract emojis from text

    Input: string
    Output: a string of emojis that are extracted from the text
    """
    return ''.join(c for c in str(s) if c in emoji.UNICODE_EMOJI)

# define ios devices:
ios = ['Twitter for iPhone', 'Twitter for iPad', 'iOS', 'Twitter for Mac', 'Tweetbot for Mac',
       'TweetCaster for iOS', 'Tweetbot for iΟS', 'Osfoora for iOS', 'UberSocial for iPhone', 'OS X',
       'Instagram on iOS', '8 Ball Pool™ on iOS', 'Photos on iOS']
# define Android devices:
android = ['Twitter for Android', 'TweetCaster for Android', 'Plume for Android',
          'Twitter for Android Tablets', 'Twitter for  Android', 'UberSocial for Android',
          'Vine for Android', 'Fenix for Android', 'Echofon  Android']

high_prob_25_word_list = ['married','producer','engineer','mother','30','family','woman','old','work','working'
                         'writer','gallery','mom','wife','kids','retired','c.e.o','nurse','lady','business','employed'
                         'photographer','mommy','ceo','hairstylist','journalist']


def feature_engineering(age_profiles):
    """
    Process age_profiles for modeling

    Input: pandas dataframe
    Output: pandas dataframe
    """
    ap_new = pd.concat([age_profiles,
                        pd.DataFrame(age_profiles['status'].apply(pd.Series) \
                        .rename(columns = lambda x: 'status_' + str(x)))], axis=1)
    ap_new['status_emoji'] = ap_new['status_text'].map(lambda x: extract_emojis(x))
    ap_new['has_emoji'] = ap_new['status_emoji'].map(lambda x: int(x != ''))
    ap_new['emoji_cnt'] = ap_new['status_emoji'].map(lambda x: len(x))
    # parse source to get device type:
    ap_new['source_parsed'] = ap_new['status_source'].apply(lambda x: BeautifulSoup(str(x),'html.parser').text)
    ap_new['source_ios'] = (ap_new['source_parsed'].isin(ios)).astype(int)
    ap_new['source_android'] = (ap_new['source_parsed'].isin(android)).astype(int)

    ap_new['has_description'] = ap_new['description'].map(lambda x: x != '') # has profile description
    ap_new['older_group_words'] = ap_new['description'].apply(lambda x: int(any(term in x for term in high_prob_25_word_list)))
    return ap_new


def joined_age(ap_new, ages_train):
    """
    Join ages_train with age_profiles, return a joined dataframe

    Input:
    -------
    pandas dataframe, pandas dataframe
    Output:
    -------
    pandas dataframe
    """
    ap_new = ap_new.set_index('id')
    ages_train = ages_train.set_index('ID')
    age_joined = ap_new.join(ages_train, how='inner')
    return age_joined


def get_age_group(age_joined):
    """
    Create a new column as age_decade

    Input:
    -------
    pandas dataframe
    Output:
    -------
    pandas dataframe with a new column 'age_decade'
    """
    bins = [17,25,35,45,55,120]
    group_names = ['18-25','26-35','36-45','46-55','>55']
    age_joined['age_decade'] = pd.cut(age_joined['Age'], bins, labels=group_names)
    return age_joined


def design_matrix(age_joined):
    """
    Create design matrix with multiclass labels

    Input:
    -------
    pandas DataFrame
    Output:
    -------
    pandas DataFrame can be put into model directly
    with a new column age group index as label
    """
    age_joined['age_idx'] = 1 # for age group 16-25
    age_joined['age_idx'][age_joined['age_decade'] == '26-35'] = 2
    age_joined['age_idx'][age_joined['age_decade'] == '36-45'] = 3
    age_joined['age_idx'][age_joined['age_decade'] == '46-55'] = 4
    age_joined['age_idx'][age_joined['age_decade'] == '>55'] = 5
    df = age_joined[['emoji_cnt', 'source_android','statuses_count',
                     'has_description','followers_count','favourites_count',
                     'source_ios', 'has_emoji', 'older_group_words',
                     'profile_background_tile', 'age_idx']]
    return df


def plot_feature_importance(estimator, X):
    """
    Plot the feature importances of random forest

    Input:
    -------
    model estimator, design matrix
    Output:
    -------
    feature importance plot
    """
    importances = estimator.feature_importances_
    indices = np.argsort(importances)
    plt.title("Feature importances")
    plt.barh(range(X.shape[1]), importances[indices],
           color="r", alpha=.7, align="center")
    feature_names = X.columns
    plt.yticks(range(X.shape[1]), feature_names[indices])
    plt.ylim([-1, X.shape[1]])
    plt.tight_layout()
    plt.show()


def write_pickle(filename, model):
    """
    Write the final model to a pickle file

    Input:
    -------
    filename: String
    model: sklearn model instance
    Output:
    -------
    Nothing
    """
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    age_profiles = pd.read_json('assignment_package/age_profiles.json')
    ages_train = pd.read_csv('assignment_package/ages_train.csv')
    ages_train = clean_data(ages_train) # remove age outliers
    ap = feature_engineering(age_profiles)
    age_joined = joined_age(ap, ages_train)
    age_joined = get_age_group(age_joined)
    # get design matrix
    df = design_matrix(age_joined)
    y = df.pop('age_idx')
    X = df
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=42,
                                                    stratify=y)
    # Apply SMOTE to deal with imbalanced data
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_sample(X_train, y_train)
    # Also try undersampling and oversampling
    ros = RandomOverSampler(ratio='minority',random_state=42)
    X_train, y_train = ros.fit_sample(X_train, y_train)
    # A dataset without over/undersampling is giving the same performance
    # thus I'm not using over/undersampling when building models

    # logistic regression
    lr = LogisticRegression(solver='sag',
                            multi_class='multinomial',
                            C=1,
                            penalty='l2',
                            fit_intercept=True,
                            random_state=42)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    score = lr.score(X_test, y_test)
    print('Logistic regression accuracy score:', score)
    cm = confusion_matrix(y_test, lr_pred)
    print('Logistic regression confusion matrix:')
    print(cm)
    class_report = classification_report(y_test, lr_pred)
    print('Logistic regression classification report')
    print(class_report)


    # random forest:
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    score = rf.score(X_test, y_test)
    print('Accuracy score:', score)
    cm = confusion_matrix(y_test, rf_pred)
    print('Random forest confusion matrix:')
    print(cm)
    class_report = classification_report(y_test, rf_pred)
    print('Random forest classification report:')
    print(class_report)
    plot_feature_importance(rf, X)


    print('pickle final model to file...')
    write_pickle('rf_model.pkl', rf)
