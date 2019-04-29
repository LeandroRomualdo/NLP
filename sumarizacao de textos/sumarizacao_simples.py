# -*- coding: utf-8 -*- 

import math
import nltk
import networkx as nx

class Sumarize:
    def __init__(self, raw_text, qtd):
        self.raw_text = raw_text
        self._sentences = None
        self._graph = None
        self.qtd = qtd

    def resumo(self):
        sentencas = sorted(
            self.sentences, key=lambda s: s.pontuacao, reverse=True)[:self.qtd]
        ordenadas = sorted(sentencas, key=lambda s: self.sentences.index(s))

        return ' '.join([s.raw_text for s in ordenadas])

    @property
    def sentences(self):
        if self._sentences is not None:
            return self._sentences
        self._sentences = [Sentenca(self, s)
                           for s in nltk.sent_tokenize(self.raw_text)]
        return self._sentences

    @property
    def graph(self):
        if self._graph is not None:
            return self._graph

        graph = nx.Graph()
        for s in self.sentences:
            graph.add_node(s)
        for node in graph.nodes():
            for n in graph.nodes():
                if node == n:
                    continue

                semelhanca = self._calculate_similarity(node, n)
                if semelhanca:
                    graph.add_edge(node, n, weight=semelhanca)

        self._graph = graph
        return self._graph

    def _calculate_similarity(self, sentence1, sentence2):
        w1, w2 = set(sentence1.palavras), set(sentence2.palavras)
        repeticao = len(w1.intersection(w2))
        semelhanca = repeticao / (math.log(len(w1)) + math.log(len(w2)))

        return semelhanca


class Sentenca:
    def __init__(self, texto, raw_text):
        self.texto = texto
        self.raw_text = raw_text
        self._palavras = None
        self._pontuacao = None

    @property
    def palavras(self):
        if self._palavras is not None:
            return self._palavras
        self._palavras = nltk.word_tokenize(self.raw_text)
        return self._palavras

    @property
    def pontuacao(self):
        if self._pontuacao is not None:
            return self._pontuacao
        pontuacao = 0.0
        for n in self.texto.graph.neighbors(self):
            pontuacao += self.texto.graph.get_edge_data(self, n)['weight']

        self._pontuacao = pontuacao
        return self._pontuacao

    def __hash__(self):
        return hash(self.raw_text)