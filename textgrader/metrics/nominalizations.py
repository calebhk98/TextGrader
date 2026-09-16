from .common import result
def measure(text,nlp=None,**_):
 if nlp is None:return [result('nlp.nominalization_density','Nominalization density',None,warning='NLP model unavailable')]
 doc=nlp(text); nouns=[t for t in doc if t.pos_=='NOUN']; hits=[t for t in nouns if t.text.lower().endswith(('tion','sion','ment','ness','ity','ance','ence'))]
 return [result('nlp.nominalization_density','Nominalization density',100*len(hits)/len(nouns) if nouns else None,'%',[{'text':t.text,'offset':t.idx} for t in hits])]
