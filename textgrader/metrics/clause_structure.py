from .common import result
def measure(text,nlp=None,**_):
 if nlp is None:return [result('nlp.parse_depth','Mean dependency parse depth',None,'levels',warning='NLP model unavailable')]
 vals=[]
 for sent in nlp(text).sents:
  for token in sent: vals.append(sum(1 for _ in token.ancestors))
 return [result('nlp.parse_depth','Mean dependency parse depth',sum(vals)/len(vals) if vals else None,'levels')]
