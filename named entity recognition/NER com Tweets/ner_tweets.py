#!/usr/bin/env python
# coding: utf8

from get_tweets import coleta_tweets
import spacy 

def get_entities(frase):
    nlp = spacy.load('modelo_treinado')
    df = []

    doc = nlp(frase)
    for ent in doc.ents:
        df.append(ent.text, ent.labels_)
        print(ent.text, ent.labels_)
    return df