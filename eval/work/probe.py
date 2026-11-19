import sys,re
from pathlib import Path
from codicil.application.index import load_index
idx=load_index(Path('build/index-bge-small-en-v1.5'))
for q in sys.argv[1:]:
    key,_,query=q.partition(':')
    hits=idx.bm25.score(query,30)
    n=0
    for i,_s in hits:
        c=idx.chunks[i]
        if not c.doc_id.startswith({'kyc':'commercial-banks-know','dep':'commercial-banks-interest-rate-on','dps':'commercial-banks-digital','emandate':'digital-payments','auth':'authentication','ppi':'on-prepaid','aa':'non-banking-financial-companies-account','p2p':'non-banking-financial-companies-peer','nbfcrbc':'non-banking-financial-companies-resp','ombud':'commercial-banks-internal','psl':'priority','da':'deposits-and','cir':'commercial-banks-credit-inf','adv':'commercial-banks-interest-rates','cc':'commercial-banks-credit-cards','rbc':'commercial-banks-responsible','nbfccc':'non-banking-financial-companies-credit'}[key]): continue
        print(f"[{key}] {query[:30]}| {c.path[-1][:30]}: {re.sub(chr(10),' / ',c.text)[:300]}"); n+=1
        if n>=1: break
