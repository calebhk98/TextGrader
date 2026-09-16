from .common import tokens,result
def value(text,window=100):
 t=tokens(text)
 if not t:return None
 if len(t)<=window:return len(set(t))/len(t)
 return sum(len(set(t[i:i+window]))/window for i in range(len(t)-window+1))/(len(t)-window+1)
def measure(text,config=None,**_):
 w=(config or {}).get('window',100); return [result('style.mattr','Moving-average type-token ratio',value(text,w)*100 if value(text,w) is not None else None,'%')]
