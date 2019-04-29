#!/usr/bin/env python
# coding: utf8

from __future__ import unicode_literals, print_function
import plac
import spacy
from pathlib import Path
import random
from spacy.util import minibatch, compounding
from spacy.lang.pt import Portuguese
from ast import literal_eval
import datetime
import time 
import xx_ent_wiki_sm

TRAIN_DATA = [('adoro',{'entities':[(0,5,'critica')]}),
            ('odeio',{'entities':[(0,5,'critica')]}),
            ('gosto',{'entities':[(0,5,'critica')]}),
            ('perfeito',{'entities':[(0,8,'critica')]}),
            ('pessimo',{'entities':[(0,7,'critica')]}),
            ('bom',{'entities':[(0,3,'critica')]}),
            ('ruim',{'entities':[(0,4,'critica')]}),
            ('maravilhoso',{'entities':[(0,11,'critica')]}),
            ('ridiculo',{'entities':[(0,8,'critica')]}),
            ('porcaria',{'entities':[(0,8,'critica')]}),
            ('maratona',{'entities':[(0,8,'evento')]}),
            ('episódio',{'entities':[(0,8,'programação')]}),
            ('amanhã',{'entities':[(0,6,'programação')]}),
            ('hoje',{'entities':[(0,4,'programação')]}),
            ('sábado',{'entities':[(0,6,'programação')]}),
            ('ontem',{'entities':[(0,5,'programação')]}),
            ('segunda',{'entities':[(0,6,'programação')]}),
            ('terça',{'entities':[(0,6,'programação')]}),
            ('quarta',{'entities':[(0,6,'programação')]}),
            ('quinta',{'entities':[(0,6,'programação')]}),
            ('sexta',{'entities':[(0,5,'programação')]}),
            ('domingo',{'entities':[(0,7,'programação')]})]


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model=None, output_dir=None, n_iter=100):

    if model is not None:
        nlp = spacy.load(model) 
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")
        print("Created blank 'en' model")

    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    else:
        ner = nlp.get_pipe("ner")

    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes): 
        if model is None:
            nlp.begin_training()
        for itn in range(n_iter):
            ts = time.time()
            st=  datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            print (st)
            print("Starting iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}

            batches = minibatch(TRAIN_DATA, size=compounding(1000., 8000., 1.25))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts, 
                    annotations, 
                    drop=0.5, 
                    losses=losses,
                )
            print("Losses", losses)
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    print (st)
    print ("CONCLUIDO") 

    for text, _ in TRAIN_DATA:
        doc = nlp(text)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
        print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        for text, _ in TRAIN_DATA:
            doc = nlp2(text)
            print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
            print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])


if __name__ == "__main__":
    plac.call(main)