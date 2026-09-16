from collections import Counter
from .common import tokens,result
COMMON={'the','a','an','and','or','but','of','to','in','on','for','with','is','was','it','he','she','they','i','you'}
def measure(text,config=None,**_):
 t=tokens(text); windows=(config or {}).get('windows',[50,100,250]); details=[]; score=0
 for size in windows:
  repeated=0
  for start in range(0,len(t),size): repeated+=sum(v-1 for k,v in Counter(t[start:start+size]).items() if v>1 and k not in COMMON and len(k)>3)
  details.append({'window':size,'repeat_excess':repeated}); score+=repeated
 return [result('style.local_lexical_repetition','Local lexical repetition',100*score/(len(t)*len(windows)) if t else None,'%',details)]
