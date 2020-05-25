#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import pandas as pd
import os
from scipy.sparse import hstack
from sklearn.naive_bayes import MultinomialNB
import pickle

# The name of the folder were the tweets are held.
TWEETS_FOLDER = "tweets"
COUNTRIES_FILE = "countries.txt"


def parse_data(data_folder):
    """
    Read all the csv files in the given folder and merge them to a single
    pandas dataframe. Note that this function drops the 'user' column and
    keeps the tweets owner id number at the head of each column.
    :param data_folder: 
    :return: 
    """
    tweets = list()
    for filename in os.listdir(data_folder):
        # Only choose files with CSV ending and ignore the test file.
        if filename.endswith(".csv") and filename != 'tweets_test_demo.csv':
            current_tweet = pd.read_csv(TWEETS_FOLDER + '/' + filename)
            tweets.append(current_tweet)
            continue
        else:
            continue

    # Join all the tweets to a single pandas array.
    tweets = pd.concat(tweets, sort=True, axis=0)
    return tweets


def len_feature(tweets):
    feature_vec = np.zeros((len(tweets), 1))
    for i, tweet in enumerate(tweets):
        feature_vec[i] = len(tweet.split())
    return feature_vec


def country_parse():
    """
    Generates an np.array of all the countries.
    :param countries_file: The file of countries location.
    :return: An array of all countries in lower case!
    """
    f = open(COUNTRIES_FILE, "r")
    countries = list()
    for country in f:
        countries.append(country.replace('\n', '').lower())
    np.array(countries)
    return countries


def countries_tweets_count(tweets):
    """
    Receives an array of tweets and counts the appearances of countries
    :param tweets: an array of tweets
    :return: an array of integers
    """
    country_count = np.zeros((len(tweets), 1))
    for i, tweet in enumerate(tweets):
        tweet_country_count = 0
        for country in country_parse():
            tweet_country_count = tweet_country_count + tweet.lower().count(country)
        country_count[i] = tweet_country_count
    for i in range(len(country_count)):
        if country_count[i] != 0:
            country_count[i] = 1
    return country_count


def find_tags(tweets):
    """
    Receives an array of tweets and counts the appearances of countries
    :param tweets: an array of tweets
    :return: an array of integers
    """
    tag_count = np.zeros((len(tweets), 1))
    for i, tweet in enumerate(tweets):
        if tweet.lower().count('@') > 2:
            tag_count[i] = 1
    return tag_count


def classify(tweets):
    """
    Classifies tweets to various twitter stars.
    :param tweets:
    :return:
    """

    vc_file = open("vc.pickle", "rb")
    vc = pickle.load(vc_file)
    vc_file.close()

    vcTags_file = open("vcTags.pickle", "rb")
    vcTags = pickle.load(vcTags_file)
    vcTags_file.close()

    vcExclamationMark_file = open("vcExclamationMark.pickle", "rb")
    vcExclamationMark = pickle.load(vcExclamationMark_file)
    vcExclamationMark_file.close()

    vcQuestionMark_file = open("vcQuestionMark.pickle", "rb")
    vcQuestionMark = pickle.load(vcQuestionMark_file)
    vcQuestionMark_file.close()

    vcDotsMark_file = open("vcDotsMark.pickle", "rb")
    vcDotsMark = pickle.load(vcDotsMark_file)
    vcDotsMark_file.close()

    train_avg_len_file = open("train_avg_len.pickle", "rb")
    train_avg_len = pickle.load(train_avg_len_file)
    train_avg_len_file.close()


    print("HEY")

    tweets = tweets['tweet']

    print(tweets)
    print(tweets.shape)

    # VC transform
    hist = vc.transform(tweets)
    print("hist shape is " + str(hist.shape))

    # Lengths
    test_tweets_lengths = len_feature(tweets)
    test_tweets_lengths = test_tweets_lengths / train_avg_len

    # Find all tags
    tags = vcTags.transform(tweets)

    # Find all marks
    exclamationMarks = np.sum(vcExclamationMark.transform(tweets), axis=1)

    questionMarks = np.sum(vcQuestionMark.transform(tweets), axis=1)

    dotsMarks = np.sum(vcDotsMark.transform(tweets), axis=1)

    countries = countries_tweets_count(tweets)

    tweets = hstack([hist, tags, exclamationMarks, questionMarks, dotsMarks])
    print(test_tweets_lengths)
    print(tweets.shape)
    print(test_tweets_lengths.shape)
    tweets = hstack([tweets, test_tweets_lengths])
    tweets = hstack([tweets, countries])



    classifier_file = open("best_classifier_ever.pickle", "rb")
    classifier = pickle.load(classifier_file)
    classifier_file.close()

    print(classifier.predict(tweets))

    return classifier.predict(tweets)



def train_it():
    # generate train and test data
    data = parse_data(TWEETS_FOLDER)
    test_size = 1
    test_data = data.sample(test_size)
    train_data = data.drop(test_data.index)

    #####################
    # Generate Features #
    #####################

    # Transform all tweets to matrix of features per tweet
    vc = CountVectorizer(ngram_range=(1, 2), stop_words='english')
    hist = vc.fit_transform(train_data['tweet'])

    # Generate X length feature and join it with the previous features
    train_tweets_lengths = len_feature(train_data['tweet'])
    train_avg_len = sum(train_tweets_lengths) / len(train_tweets_lengths)
    train_tweets_lengths = train_tweets_lengths / train_avg_len

    # Generate the length feature of the train set
    test_tweets_lengths = len_feature(test_data['tweet'])
    test_tweets_lengths = test_tweets_lengths / train_avg_len

    # Find all tags
    vcTags = CountVectorizer(token_pattern=r'\b@\b')
    # tags = find_tags(train_data['tweet'])
    tags = vcTags.fit_transform(train_data['tweet'])

    # Find all marks
    vcExclamationMark = CountVectorizer(token_pattern=r'(?u)(?:[!]+)')
    exclamationMarks = np.sum(vcExclamationMark.fit_transform(train_data['tweet']), axis=1)

    vcQuestionMark = CountVectorizer(token_pattern=r'(?u)(?:[?]+)')
    questionMarks = np.sum(vcQuestionMark.fit_transform(train_data['tweet']), axis=1)

    vcDotsMark = CountVectorizer(token_pattern=r'(?u)(?:[.]{2,})')
    dotsMarks = np.sum(vcDotsMark.fit_transform(train_data['tweet']), axis=1)

    # Find Countries
    countries = countries_tweets_count(train_data['tweet'])

    features = hstack([hist, tags, exclamationMarks, questionMarks, dotsMarks])
    features = hstack([features, train_tweets_lengths])
    features = hstack([features, countries])

    # Generate Test Features
    hist_test = vc.transform(test_data['tweet'])
    # tags_test = find_tags(test_data['tweet'])
    tags_test = vcTags.transform(test_data['tweet'])
    exclamation_marks_test = np.sum(vcExclamationMark.transform(test_data['tweet']), axis=1)
    question_marks_test = np.sum(vcQuestionMark.transform(test_data['tweet']), axis=1)
    dots_marks = np.sum(vcDotsMark.transform(test_data['tweet']), axis=1)
    countries_test = countries_tweets_count(test_data['tweet'])

    test_features = hstack([hist_test, tags_test, exclamation_marks_test, question_marks_test, dots_marks])
    test_features = hstack([test_features, test_tweets_lengths])
    test_features = hstack([test_features, countries_test])
    # test_features = hstack([test_features, tags_test])


    # Transform features
    transformer = TfidfTransformer()
    freq_all = transformer.fit_transform(features)
    freq_test = transformer.fit_transform(test_features)

    clf2 = MultinomialNB().fit(freq_all, train_data['user'])
    # predicted2 = clf2.predict(freq_test)

    # Success of prediction on test_data.
    # print(len(predicted2[predicted2 == test_data['user']]) / test_size)


    save_classifier = open("best_classifier_ever.pickle", "wb")
    pickle.dump(clf2, save_classifier)
    save_classifier.close()

    save_vc = open("vc.pickle", "wb")
    pickle.dump(vc, save_vc)
    save_vc.close()

    save_vcTags = open("vcTags.pickle", "wb")
    pickle.dump(vcTags, save_vcTags)
    save_vcTags.close()

    save_vcExclamationMark = open("vcExclamationMark.pickle", "wb")
    pickle.dump(vcExclamationMark, save_vcExclamationMark)
    save_vcExclamationMark.close()

    save_vcQuestionMark = open("vcQuestionMark.pickle", "wb")
    pickle.dump(vcQuestionMark, save_vcQuestionMark)
    save_vcQuestionMark.close()

    save_vcDotsMark = open("vcDotsMark.pickle", "wb")
    pickle.dump(vcDotsMark, save_vcDotsMark)
    save_vcDotsMark.close()

    save_train_avg = open("train_avg_len.pickle", "wb")
    pickle.dump(train_avg_len, save_train_avg)
    save_train_avg.close()


if __name__ == '__main__':


    file ="tweets_test_demo.csv"
    current_tweet = pd.read_csv(file)

    classify(current_tweet)
