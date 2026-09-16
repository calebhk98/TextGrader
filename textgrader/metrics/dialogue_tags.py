import re
from textgrader.text import parse_quotations
from .common import result
BASIC={'said','asked'}; TAGS=BASIC|{'replied','whispered','shouted','murmured','cried','called','answered','exclaimed'}
def measure(text,**_):
 quotes=parse_quotations(text); found=[]
 for start,end,_ in quotes:
  context=text[max(0,start-100):start]+' '+text[end:end+100]; match=re.search(r'\b('+('|'.join(sorted(TAGS)))+r')\b',context,re.I)
  if match:found.append(match.group(1).lower())
 details=[{'tag':x} for x in found]; elaborate=sum(x not in BASIC for x in found)
 return [result('style.dialogue_tag_density','Dialogue tag density',100*len(found)/len(quotes) if quotes else None,'% of quotations',details),result('style.elaborate_tag_rate','Elaborate dialogue tags',100*elaborate/len(found) if found else None,'%',details)]
