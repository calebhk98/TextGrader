from collections import Counter
from .common import tokens,result
FUNCTION='a an and are as at be been but by for from had has have he her him his i if in is it its me my nor not of on or our she so than that the their them then there they this to us was we were what when which who will with you your'.split()
def vector(text):
 t=tokens(text); c=Counter(t); return {w:1000*c[w]/len(t) for w in FUNCTION} if t else {}
def measure(text,profile=None,**_):
 v=vector(text); rows=(profile or {}).get('feature_profiles',{}).get('function_words',[])
 if not rows: return [result('style.function_word_delta','Function-word Burrows Delta',None,'delta',warning='No corpus function-word profiles available')]
 means={w:sum(r.get(w,0) for r in rows)/len(rows) for w in FUNCTION}; sd={w:(sum((r.get(w,0)-means[w])**2 for r in rows)/len(rows))**.5 for w in FUNCTION}
 delta=sum(abs(v.get(w,0)-means[w])/sd[w] for w in FUNCTION if sd[w])/max(1,sum(bool(sd[w]) for w in FUNCTION))
 return [result('style.function_word_delta','Function-word Burrows Delta',delta,'delta')]
