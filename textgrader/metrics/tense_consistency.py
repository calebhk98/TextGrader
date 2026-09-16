from collections import Counter
from .common import result
def measure(text,nlp=None,**_):
 if nlp is None:return [result('nlp.tense_drift','Tense consistency',None,warning='NLP model unavailable')]
 blocks=[]
 for sent in nlp(text).sents:
  c=Counter(v.morph.get('Tense')[0] for v in sent if v.pos_ in ('VERB','AUX') and v.morph.get('Tense')); blocks.append(c.most_common(1)[0][0] if c else None)
 changes=sum(a!=b for a,b in zip(blocks,blocks[1:]) if a and b); return [result('nlp.tense_drift','Sentence-to-sentence tense changes',changes,'changes')]
