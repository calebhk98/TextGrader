from collections import Counter
from .common import result
KEEP={'NOUN','PROPN','VERB','ADJ','ADV'}
def measure(text,nlp=None,**_):
 if nlp is None:return [result('nlp.pos_distribution','POS distribution',None,warning='NLP model unavailable')]
 c=Counter(t.pos_ for t in nlp(text) if t.is_alpha); total=sum(c.values()); return [result('nlp.pos_'+p.lower(),p.title()+' share',100*c[p]/total if total else None,'%') for p in sorted(KEEP)]
