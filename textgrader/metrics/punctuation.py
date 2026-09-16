from collections import Counter
from .common import tokens,result
MARKS={';':'semicolon',':':'colon','(':'parenthesis','…':'ellipsis','!':'exclamation','?':'question','—':'em_dash','–':'en_dash'}
def vector(text):
 total=len(tokens(text)); c=Counter(text); return {name:1000*c[ch]/total for ch,name in MARKS.items()} if total else {}
def measure(text,**_):
 v=vector(text); return [result('style.punctuation_'+k,'Punctuation: '+k.replace('_',' '),x,'per 1,000 words') for k,x in v.items()]
