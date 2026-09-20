from pathlib import Path
import hashlib,json,re,sys
from contextlib import closing
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'healthcare-agentic-ai'))
OUT=ROOT/'outputs/gpt-migration'
OUT.mkdir(exist_ok=True)
pattern=re.compile(r'ollama|11434|llama-server|qwen|gemma|local LLM|local model|LLMProvider|/api/chat|/api/generate',re.I)
scan=[]
for p in ROOT.rglob('*'):
    if not p.is_file() or any(x in p.relative_to(ROOT).parts for x in ('.git','.venv','__pycache__','data')) or p.name=='.env' or 'gpt-migration' in p.parts or p==Path(__file__): continue
    try: text=p.read_text(encoding='utf8')
    except (UnicodeError,OSError): continue
    hits=[i for i,line in enumerate(text.splitlines(),1) if pattern.search(line)]
    if hits: scan.append({'path':p.relative_to(ROOT).as_posix(),'lines':hits})
mode=sys.argv[1]
for entry in scan:
    name=entry['path']
    if name.startswith('prompts/'):
        entry['classification']='Original user task specification; not healthcare runtime'
    elif '/outputs/' in name:
        entry['classification']='Historical result/provenance artifact; non-executable'
    elif name.endswith('rag/embeddings.py'):
        entry['classification']='Local embedding model only; generation-independent'
    elif name.endswith('docs/phase4-validation.md'):
        entry['classification']='Required migration audit status labels; no runtime setup'
    else:
        entry['classification']='REVIEW'
(OUT/f'search-{mode}.json').write_text(json.dumps(scan,indent=2))
from qdrant_client import QdrantClient
from rag.patient_ingestion import validate_document
from rag.medical_ingestion.models import validate_chunk
snap={}
for rel,collection,validate in [('data/qdrant/patient_cases','ddxplus_patient_cases',validate_document),('data/medical_knowledge/qdrant','medical_knowledge',validate_chunk)]:
    path=(ROOT/'healthcare-agentic-ai' if collection=='ddxplus_patient_cases' else ROOT)/rel
    assert path.is_dir(),path
    with closing(QdrantClient(path=str(path))) as client:
        assert client.collection_exists(collection)
        records=[];offset=None;keys=set()
        while True:
            points,offset=client.scroll(collection,limit=128,offset=offset,with_payload=True,with_vectors=True)
            for p in points:
                validate(p.payload); keys.add(tuple(sorted(p.payload)))
                records.append({'id':str(p.id),'payload':p.payload,'vector':p.vector})
            if offset is None: break
        records.sort(key=lambda p:p['id'])
        snap[collection]={'count':client.count(collection,exact=True).count,'payload_keys':[list(k) for k in sorted(keys)],'content_sha256':hashlib.sha256(json.dumps(records,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
protected={}
for rel in ['rag/patient_parser.py','rag/models.py','rag/embeddings.py','rag/medical_embeddings.py','rag/patient_rag.py','rag/patient_retriever.py','rag/medical_retriever.py','rag/vector_store.py','rag/medical_vector_store.py','rag/patient_ingestion.py','rag/patient_audit.py','rag/agents/grounding.py','rag/agents/models.py','rag/phase4_evaluation.py']:
    p=ROOT/'healthcare-agentic-ai'/rel
    protected[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
snap['protected_source_hashes']=protected
(OUT/f'integrity-{mode}.json').write_text(json.dumps(snap,indent=2))
print(json.dumps({k:v for k,v in snap.items() if k!='protected_source_hashes'},indent=2))
print('Files with search matches:',len(scan))
if mode=='after': print('Integrity unchanged:',snap==json.loads((OUT/'integrity-before.json').read_text()))
