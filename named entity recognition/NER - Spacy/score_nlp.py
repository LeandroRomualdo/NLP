from collections import defaultdict
import spacy
nlp = spacy.load('sky_ner_cold')

def get_scores(text):

    with nlp.disable_pipes('ner'):
        doc = nlp(text)

    threshold = 0.3
    (beams, somethingelse) = nlp.entity.beam_parse([ doc ], beam_width = 16, beam_density = 0.0001)

    entity_scores = defaultdict(float)
    for beam in beams:
        for score, ents in nlp.entity.moves.get_beam_parses(beam):
            for start, end, label in ents:
                entity_scores[(start, end, label)] += score

    for key in entity_scores:
        start, end, label = key
        score = entity_scores[key]
        if ( score > threshold):
            print ('Tipo de entidade: {}, Texto: {}, Score: {}'.format(label, doc[start:end], score))