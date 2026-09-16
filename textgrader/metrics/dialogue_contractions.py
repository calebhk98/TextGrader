from .common import result
from textgrader.text import split_dialogue,words
def measure(text,**_):
 d=split_dialogue(text); t=words(d.spoken); count=sum("'" in x or '’' in x for x in t)
 return [result('style.dialogue_contraction_rate','Dialogue contraction rate',100*count/len(t) if t else None,'%',warning='; '.join(d.warnings) or None)]
