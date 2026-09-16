"""Small dependency-free helpers shared by optional style metrics."""
from collections import Counter
import math
import statistics
from textgrader.text import words, sentences, paragraphs, split_dialogue

def tokens(text): return [w.lower().replace("’", "'") for w in words(text)]
def lengths(items): return [len(words(x)) for x in items if words(x)]
def quantile(values, q):
    if not values: return None
    ordered=sorted(values); pos=(len(ordered)-1)*q; lo=int(pos); hi=min(lo+1,len(ordered)-1)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(pos-lo)
def cosine_distance(a,b):
    keys=set(a)|set(b); dot=sum(a.get(k,0)*b.get(k,0) for k in keys)
    na=math.sqrt(sum(v*v for v in a.values())); nb=math.sqrt(sum(v*v for v in b.values()))
    return 1-dot/(na*nb) if na and nb else None
def rates(counter,total,scale=100): return {k: scale*v/total for k,v in counter.items()} if total else {}
def result(metric_id,name,value,unit=None,details=None,warning=None):
    return {"metric_id":metric_id,"name":name,"value":value,"unit":unit,"details":details or [],"warning":warning}
