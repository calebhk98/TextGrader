from collections import Counter
from textgrader.text import transcript_lines,words
from .common import cosine_distance,result
from .function_words import FUNCTION
def _vector(text):
 t=[x.lower() for x in words(text)]; c=Counter(t); total=len(t) or 1
 v={f'f:{w}':c[w]/total for w in FUNCTION}; v.update({'contractions':sum("'" in x or '’' in x for x in t)/total,'questions':text.count('?')/total,'exclamations':text.count('!')/total}); return v
def measure(text,config=None,**_):
 groups={}
 for m in transcript_lines(text): groups.setdefault(m.groupdict().get('username','unknown'),[]).append(m.groupdict().get('message',''))
 pairs=[]
 for i,a in enumerate(sorted(groups)):
  for b in sorted(groups)[i+1:]: pairs.append({'a':a,'b':b,'distance':cosine_distance(_vector(' '.join(groups[a])),_vector(' '.join(groups[b]))),'lines_a':len(groups[a]),'lines_b':len(groups[b])})
 avg=sum(x['distance'] for x in pairs if x['distance'] is not None)/len(pairs) if pairs else None
 return [result('style.character_voice_distance','Mean pairwise character voice distance',avg,'cosine distance',pairs,warning=None if pairs else 'Need transcript-style speaker lines for at least two characters')]
