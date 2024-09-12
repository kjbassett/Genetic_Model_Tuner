from copy import deepcopy


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

    def reset(self):
        self.score = 0
        self.fitness = 0
        self.knowledge = None


def dna2str(dna):
    dna_str = ''
    for gene in dna:
        dna_str += gene['name'] + '('
        dna_str += ', '.join([str(a) for a in gene['args']])
        for key, value in gene['kwargs'].items():
            dna_str += f', {key}={value}'
        dna_str += ')'
    return dna_str
