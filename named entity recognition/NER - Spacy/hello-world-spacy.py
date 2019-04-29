import spacy
from pathlib import Path
import random
from spacy.util import minibatch, compounding
#import xx_ent_wiki_sm
from spacy.lang.pt import Portuguese
from ast import literal_eval

output_dir = "./trained_model"

modelDir = Path(output_dir)

'''
if modelDir.exists():
    nlp = spacy.load(modelDir)
    print('Saved model loaded')
else:
    #nlp = spacy.load('pt')
    nlp = xx_ent_wiki_sm.load()
    nlp = Portuguese(pipeline=['tensorizer', 'ner'])
'''
nlp = spacy.blank('pt')    
if modelDir.exists() is False:

    # training data
    TRAIN_DATA = open('tests/df_titulos.txt', 'r').read()#.replace('\n,'')
    print('Dados carregados')
    try:
        TRAIN_DATA = literal_eval(TRAIN_DATA)
        print('literal eval aplicado')
    except:
        print('Falha ao aplicar eval')     
    
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
        # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe('ner')
        ner.add_label('ner')

        # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    n_iter=100

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            print("Starting iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.55,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
                print('Losses', losses)

    # test the trained model
    for text, _ in TRAIN_DATA:
        doc = nlp(text)
        print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
        #print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])

doc = nlp('O dia depois de amanhã')
print(doc.text)

# Find named entities, phrases and concepts
for entity in doc.ents:
    print(entity.text, entity.label_)

if modelDir.exists() is False:
    print("Salvando modelo!!")
    nlp.to_disk(output_dir)
