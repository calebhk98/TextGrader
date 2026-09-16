from collections import defaultdict
from .common import tokens,result
def measure(text,config=None,**_):
 t=tokens(text); sizes=(config or {}).get('sizes',[3,4,5,6]); hits=[]
 for n in sizes:
  seen=defaultdict(list)
  for i in range(len(t)-n+1): seen[tuple(t[i:i+n])].append(i)
  for gram,positions in seen.items():
   if len(positions)>1: hits.append({'n':n,'text':' '.join(gram),'count':len(positions),'positions':positions,'distances':[b-a for a,b in zip(positions,positions[1:])]})
 return [result('style.repeated_ngrams','Repeated n-gram types',len(hits),'types',hits)]
