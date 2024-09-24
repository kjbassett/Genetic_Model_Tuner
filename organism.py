from copy import deepcopy
import os
import json


class Organism:

    def __init__(self, dna=None):
        if dna is None:
            self.dna = []
        else:
            self.dna = dna
        self.score = 0
        self.fitness = 0
        self.knowledge = None

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __eq__(self, other):
        return dna2str(self.dna) == dna2str(other.dna)

    def __hash__(self):
        return hash(dna2str(self.dna))

    def __str__(self):
        return dna2str(self.dna)

    def __repr__(self):
        return self.__str__()

    def add_gene(self, gene):
        self.dna.append(gene)

    def mate(self, other):
        pass

    def reproduce(self):
        return Organism(deepcopy(self.dna))

    def save(self):

        dna = deepcopy(self.dna)
        print(dna)
        # remove 'func' keys from dna
        for gene in dna:
            for key in ['train', 'inference']:
                if gene[key] is None:
                    continue
                gene[key].pop('func')
        print(dna)
        if not os.path.exists('organisms'):
            os.makedirs('organisms')
        with open('organisms/config.json', 'w') as f:
            json.dump(dna, f, cls=StrEncoder, indent=4)

    def reset(self):
        self.score = 0
        self.fitness = 0
        self.knowledge = None


def dna2str(dna):
    dna_str = ''
    for gene in dna:
        dna_str += gene['name'] + '('
        if gene['train']:
            dna_str += ', '.join([str(a) for a in gene['train']['args']])
            for key, value in gene['train']['kwargs'].items():
                dna_str += f', {key}={value}'
        dna_str += ')'
    return dna_str


class StrEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)
