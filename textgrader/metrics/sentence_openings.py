from collections import Counter
from textgrader.text import sentences
from .common import tokens,result
def measure(text,config=None,**_):
 n=(config or {}).get('words',3); openings=[' '.join(tokens(s)[:n]) for s in sentences(text) if tokens(s)]; c=Counter(openings); repeated=sum(v for v in c.values() if v>1)
 return [result('style.sentence_opening_repetition', 'Repeated sentence openings',100*repeated/len(openings) if openings else None,'%', [{'opening':k,'count':v} for k,v in c.most_common() if v>1])]
