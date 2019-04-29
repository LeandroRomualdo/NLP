#!/usr/bin/env python
# -*- coding: utf-8 -*-

import spacy
spacy.require_gpu()
from pathlib import Path
import random
from spacy.util import minibatch, compounding
#import xx_ent_wiki_sm
from spacy.lang.pt import Portuguese
from ast import literal_eval
import datetime
import time 

output_dir = "./sky_ner"

modelDir = Path(output_dir)

nlp = spacy.blank('pt')    
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
print (st)
if modelDir.exists() is True:

    # training data
    TRAIN_DATA = open('dataset_new.txt', 'r').read()
    print('Dados carregados')
    try:
        TRAIN_DATA = literal_eval(TRAIN_DATA)
        print('literal eval aplicado')
    except: 
        print('Falha ao aplicar eval')     
    
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    else:
        ner = nlp.get_pipe('ner')
        ner.add_label('ner')

    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    n_iter=10

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            ts = time.time()
            st=  datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            print (st)
            print("Starting iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            #batches = minibatch(TRAIN_DATA, size=compounding(4., 32., 1.001))
            batches = minibatch(TRAIN_DATA, size=compounding(1000., 8000., 1.25))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,
                    annotations,
                    drop=0.3,
                    sgd=optimizer,
                    losses=losses)
                print('Losses', losses)
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    print (st)
    print ("CONCLUIDO") 

    # test the trained model
    for text, _ in TRAIN_DATA:
        doc = nlp(text)
        print('Entities', [(ent.text, ent.label_) for ent in doc.ents])

doc = nlp('O dia depois de amanhã')
print(doc.text)

for entity in doc.ents:
    print(entity.text, entity.label_)

if modelDir.exists() is False:
    print("Salvando modelo!!")
    nlp.to_disk(output_dir)
