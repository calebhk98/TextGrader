from .common import result
def measure(text,nlp=None,**_):
 if nlp is None:return [result('nlp.passive_voice','Passive voice',None,'%',warning='NLP model unavailable')]
 doc=nlp(text); clauses=[t for t in doc if t.dep_ in ('nsubj','nsubjpass')]; passive=sum(t.dep_=='nsubjpass' or any(c.dep_=='auxpass' for c in t.head.children) for t in clauses)
 return [result('nlp.passive_voice','Passive voice',100*passive/len(clauses) if clauses else None,'%')]
