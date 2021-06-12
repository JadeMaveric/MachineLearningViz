# Prior Probabilities -> P(target==True) & P(target==False)
# Conditional Probabilities
# Outlook  | True | False
# -----------------------
# sunny    | 2/9  | 3/5
# overcast | 4/9  | 0
# rain     | 3/9  | 2/5
#
# P(Y=k|{x1,x2,x3...}) = P(x1|Y=k) * P(x2|Y=k) * ... * P(Y=k)
#                               P(x1) * P(x2) * P(x3) ...
# PosteriorProb = ProbLikihood * PriorProb / ProbEvidence

class NaiveBayes:
    def __init__(self):
        self.data = None
        self.target = ""
        self.prior = {}
        self.evidence = {}
        self.likelihood = {}
        self.attribs = []
        self.labels = []
        self.probability = None
        self.m = 1

    def fit(self, data, probability='simple', m=1):
        self.data = data
        self.attribs = data.columns[:-1]
        self.target = data.columns[-1]
        target = self.target
        self.labels = data[target].unique()
        prob_function = {
            'simple': self.simple_probability,
            'laplace': self.laplace_estimate,
            'm-estimate': self.m_estimate
        }
        self.probability = prob_function[probability]
        self.m = 1

        self.prior = {
            value: sum(data[target]==value)/len(data[target]) for value in data[target].unique()
        }
        self.evidence = {
            attrib: {
                value: sum(data[attrib]==value)/len(data[attrib]) for value in data[attrib].unique()
            } for attrib in data.columns[:-1]
        }
        self.likelihood = {
            outcome: {
                attrib: {
                    value: self.probability(attrib, value, outcome)
                        for value in data[attrib].unique()
                } for attrib in self.attribs
            } for outcome in data[target].unique()
        }

    def predict(self, record):
        if len(record) != len(self.attribs):
            raise ValueError(f"Expected tuple of size {len(self.attribs)}, but got {len(record)}")
        df = self.data
        target = self.target
        posterior = {}

        for label in self.labels:
            P_prior = self.prior[label]
            P_evidence = 1
            P_likelihood = 1
            for attrib, value in zip(self.attribs, record):
                P_evidence *= self.evidence[attrib][value]
                P_likelihood *= self.likelihood[label][attrib][value] # P(label|attrib==value)
            posterior[label] = P_likelihood * P_prior / P_evidence

        return posterior

    def evaluate(self, data):
        dict_max = lambda dict_: max(dict_, key=dict_.get)
        max_attrib = lambda record: dict_max(self.predict(record))
        prediction = list(map(max_attrib, data[self.attribs].values))
        actual = self.data[self.target].values
        return {True: sum(prediction==actual), False: sum(prediction!=actual)}

    ### Helper Functions ###
    
    def simple_probability(self, attrib, value, label):
        target = self.target
        df = self.data
        N_ic = sum((df[attrib]==value)&(df[target]==label))
        N_c = sum(df[target]==label)
        return N_ic / N_c

    def m_estimate(self, attrib, value, label):
        target = self.target
        df = self.data
        m = self.m
        p = 1 / df[attrib].nunique()
        N_ic = sum((df[attrib]==value)&(df[target]==label))
        N_c = sum(df[target]==label)
        return (N_ic + m*p) / (N_c + m)

    def laplace_estimate(self, attrib, value, label):
        target = self.target
        df = self.data
        c = df[attrib].nunique()
        N_ic = sum((df[attrib]==value)&(df[target]==label))
        N_c = sum(df[target]==label)
        return (N_ic + 1) / (N_c + c)
        
