"""Offline final migration checks; never calls generation."""
from pathlib import Path
import ast
import json
import os
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'healthcare-agentic-ai'
OUT=Path(__file__).resolve().parent
os.chdir(ROOT)
env=dict(os.environ)
env.pop('RUN_GPT_INTEGRATION',None)
command=[sys.executable,'-m','unittest','discover','-s','healthcare-agentic-ai/tests','-t','healthcare-agentic-ai','-v']
result=subprocess.run(command,env=env,capture_output=True)
text=(result.stdout+result.stderr).decode('utf8',errors='replace')
(OUT/'unit-tests.log').write_text(text,encoding='utf8')
count=int(re.search(r'Ran (\d+) tests',text).group(1))
skipped=int(re.search(r'OK \(skipped=(\d+)\)',text).group(1)) if result.returncode==0 else 0
legacy=re.compile(r'ollama|gemma|qwen|11434|/api/chat|/api/generate|local LLM',re.I)
runtime=[]
for sub in ['rag','scripts']:
 for p in (R/sub).rglob('*.py'):
  source=p.read_text(encoding='utf8');ast.parse(source)
  if legacy.search(source): runtime.append(str(p.relative_to(R)))
syntax_files=list((R/'rag').rglob('*.py'))+list((R/'scripts').rglob('*.py'))+list((R/'tests').rglob('*.py'))
for p in syntax_files: ast.parse(p.read_text(encoding='utf8'))
before=json.loads((OUT/'integrity-before.json').read_text())
after=json.loads((OUT/'integrity-after.json').read_text())
search=json.loads((OUT/'search-after.json').read_text())
checks={'offline_tests_passed':count-skipped if result.returncode==0 else None,
        'offline_test_exit_code':result.returncode,'skipped':skipped,
        'runtime_legacy_references':runtime,'syntax_files_checked':len(syntax_files),
        'rag_integrity_unchanged':before==after,
        'unclassified_search_matches':[r for r in search if r['classification']=='REVIEW']}
for name,command in [('diff_check',['git','diff','--check']),('dependency_check',['uv','pip','check'])]:
 r=subprocess.run(command,capture_output=True)
 checks[name]={'exit_code':r.returncode,'output':(r.stdout+r.stderr).decode('utf8',errors='replace')}
# Ensure new artifacts/source contain no configured secret values; never print keys.
keys=[]
for line in (ROOT/'.env').read_text(encoding='utf8').splitlines():
 if '=' in line:
  k,v=line.split('=',1)
  if 'KEY' in k:
   v=v.strip().strip('\"\'')
   if len(v)>12: keys.append(v)
leaks=[]
for p in [*OUT.rglob('*'),*(R/'rag').rglob('*.py'),*(R/'docs').rglob('*.md'),*(R/'outputs/phase4/gpt-migration-compatible').rglob('*.json')]:
 if p.is_file():
  try: s=p.read_text(encoding='utf8')
  except (UnicodeError,OSError): continue
  if any(key in s for key in keys): leaks.append(str(p.relative_to(ROOT)))
checks['secret_exposure_files']=leaks
(OUT/'final-checks.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
print(json.dumps(checks,indent=2))
assert result.returncode==0 and not runtime and before==after and not leaks
assert not checks['unclassified_search_matches']
assert checks['diff_check']['exit_code']==checks['dependency_check']['exit_code']==0
