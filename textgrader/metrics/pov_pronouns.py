from collections import Counter
from .common import tokens,result
GROUPS={'first':['i','me','my','mine','we','us','our','ours'],'second':['you','your','yours'],'third':['he','him','his','she','her','hers','they','them','their','theirs']}
def measure(text,config=None,**_):
 t=tokens(text); c=Counter(t); block=(config or {}).get('block_words',500); out=[]
 for group,items in GROUPS.items(): out.append(result('style.pov_'+group,f'{group.title()}-person pronouns',1000*sum(c[x] for x in items)/len(t) if t else None,'per 1,000 words'))
 drift=[]
 for i in range(0,len(t),block):
  cc=Counter(t[i:i+block]); drift.append({'start':i,**{g:sum(cc[x] for x in xs) for g,xs in GROUPS.items()}})
 out.append(result('style.pov_drift','POV block drift',len({max(GROUPS,key=lambda g:row[g]) for row in drift})-1 if drift else None,'changes',drift)); return out
