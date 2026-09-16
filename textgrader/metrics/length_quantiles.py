from textgrader.text import sentences,paragraphs
from .common import lengths,quantile,result
def measure(text,**_):
 out=[]
 for kind,vals in [('sentence',lengths(sentences(text))),('paragraph',lengths(paragraphs(text)))]:
  for label,q in [('p10',.1),('p25',.25),('p50',.5),('p75',.75),('p90',.9)]: out.append(result(f'style.{kind}_words_{label}',f'{kind.title()} length {label}',quantile(vals,q),'words'))
 return out
