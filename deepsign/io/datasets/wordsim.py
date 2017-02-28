

class WordSim353Reader:

    def __init__(self,sim_file,rel_file,lowercase=True):
        """

        :param sim_file: file with similarity scores
        :param rel_file: file with relatedness scores
        """
        self.words = set()
        self.sim = []
        self.rel = []

        for line in sim_file:
            (w1,w2,score) = line.split()
            if lowercase:
                w1 = w1.lower()
                w2 = w2.lower()

            self.words.add(w1)
            self.words.add(w2)

            self.sim.append((w1,w2,float(score)/10))

        for line in rel_file:
            (w1,w2,score) = line.split()
            if lowercase:
                w1 = w1.lower()
                w2 = w2.lower()

            self.words.add(w1)
            self.words.add(w2)

            self.rel.append((w1,w2,float(score)/10))


