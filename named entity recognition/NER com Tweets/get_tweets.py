#!/usr/bin/env python
# coding: utf8

from TwitterSearch import TwitterOrder, TwitterSearch, TwitterSearchOrder, TwitterSearchException
import pandas as pd


def coleta_tweets():

    try:
    
        ts = TwitterSearch(
            consumer_key = '',
            consumer_secret = '',
            access_token = '',
            access_token_secret = ''
        )
    
        tso = TwitterSearchOrder()
        tso.set_keywords(['Harry potter'])
        tso.set_language('pt')
        df = []
        for tweet in ts.search_tweets_iterable(tso):
            df.append('@%s tweeted: %s' % ( tweet['user']['screen_name'], tweet['text'])+',')
            #print( '@%s tweeted: %s' % ( tweet['user']['screen_name'], tweet['text']) )
        print('Coleta finalizada!')
        
        df = pd.DataFrame(df)
        #df.to_csv('tweets.txt')
        #print('Arquivo salvo.')
        return df
    except TwitterSearchException as e:
        print(e)