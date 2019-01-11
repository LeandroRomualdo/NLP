import spacy
from pathlib import Path
import random
from spacy.util import minibatch, compounding
from spacy.lang.pt import Portuguese
from ast import literal_eval
import datetime
import time 

output_dir = "./sky_get_intent"

modelDir = Path(output_dir)

nlp = spacy.blank('pt')
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
print (st)  
if modelDir.exists() is False:

    TRAIN_DATA = open('dataset.txt', 'r').read()
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

    n_iter=100

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
            batches = minibatch(TRAIN_DATA, size=compounding(1000., 8000., 1.25)) #aws
            #batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001)) #default
            #batches = minibatch(TRAIN_DATA, size=compounding(1000., 6000., 1.75)) #ajustada
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  
                    annotations,  
                    drop=0.55,  # Droput rate, inibe o modelo de memorizar os dados e crie um viés. Nosso modelo tem 1/4 de taxa da abandono.
                    sgd=optimizer,  
                    losses=losses)
                print('Losses', losses) 
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    print (st)
    print ("CONCLUIDO") 

    for text, _ in TRAIN_DATA:
        doc = nlp(text)
        print('Entities', [(ent.text, ent.label_) for ent in doc.ents]) 

doc = nlp('Quero trocar a porcaria do controle remoto')
print(doc.text)

for entity in doc.ents:
    print(entity.text, entity.label_)

if modelDir.exists() is False:
    print("Salvando modelo!!")
    nlp.to_disk(output_dir)
