"""Task 7.7-only protected-file verification; no historical manifest writes.

The single exception is byte-pinned, not a blanket exemption for config.py.
Benchmark/data files are hashed as opaque bytes, never parsed or imported.
"""
import hashlib
import json
from pathlib import Path

HISTORICAL_CONFIG = '1247b27604da291857636b313284da08f36f59da4a287b6a67e841efbae60a24'
AUTHORIZED_CONFIG = 'd9567f1c69a8e3e754e6cea2b765d2121add300145a248f00e3f932e3e51da58'
ADDITION = b'OKF_ENABLED = True\r\n\r\n'


def digest(path, sample=lambda: None):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            sample()
            block = stream.read(1024 * 1024)
            if not block:
                return result.hexdigest()
            result.update(block)


def config_exception(path, historical):
    content = Path(path).read_bytes()
    actual = hashlib.sha256(content).hexdigest()
    reconstructed = hashlib.sha256(content.replace(ADDITION, b'', 1)).hexdigest()
    if (historical != HISTORICAL_CONFIG or actual != AUTHORIZED_CONFIG
            or content.count(ADDITION) != 1 or reconstructed != HISTORICAL_CONFIG):
        raise ValueError('Config differs from the exact Task 7.7.1 authorized addition')
    return {'path': str(Path(path).resolve()), 'historical_sha256': historical,
            'authorized_sha256': actual, 'in_memory_reconstruction_sha256': reconstructed,
            'addition': 'OKF_ENABLED = True followed by a blank line; CRLF preserved',
            'scope': 'Task 7.7 only; historical manifests unchanged',
            'authorization': 'prompts/task7.7.1',
            'evidence': 'Git blob 63b030c3b852c596ee183d8938905d441df31662 with CRLF matches historical SHA-256; removing only the added flag and blank line in memory gives the same hash. Prior blocked documentation records user-requested addition.'}


def verify_historical(root, sample=lambda: None):
    root = Path(root).resolve()
    manifests = sorted((root / 'outputs/task7.5').rglob('frozen-files-*.json'))
    manifests += sorted((root / 'outputs/task7.6').rglob('frozen-files-*.json'))
    if len(manifests) != 6:
        raise ValueError('Exactly six historical Task 7.5/7.6 manifests required')
    expected, manifest_hashes = {}, {}
    for manifest in manifests:
        manifest_hashes[str(manifest)] = digest(manifest, sample)
        for name, sha in json.loads(manifest.read_text(encoding='utf8')).items():
            expected.setdefault(name, []).append({'manifest': str(manifest), 'sha256': sha})
    current, exceptions = {}, []
    config = root / 'rag/config.py'
    for i, (name, entries) in enumerate(sorted(expected.items()), 1):
        path = Path(name)
        if not path.is_file():
            raise ValueError(f'Missing protected file: {name}')
        current[name] = digest(path, sample)
        for entry in entries:
            if current[name] != entry['sha256']:
                if path.resolve() != config:
                    raise ValueError(f'Unexpected protected-file mismatch: {name}; {entry}')
                evidence = config_exception(config, entry['sha256'])
                exceptions.append({**evidence, 'manifest': entry['manifest']})
        if i % 25 == 0 or i == len(expected):
            print(f'Protected integrity: {i}/{len(expected)}', flush=True)
    if len(exceptions) != 6:
        raise ValueError('Expected exactly the authorized config exception in all six manifests')
    current.update(manifest_hashes)
    return current, {'protected_paths': len(expected), 'strict_matches': len(expected) - 1,
                     'authorized_exception_path_count': 1, 'exceptions': exceptions,
                     'historical_manifest_sha256': manifest_hashes}


def capture_additional(root, output, current, sample=lambda: None):
    """Freeze old runs, implementations, contracts and opaque benchmark artifacts."""
    root, output = Path(root).resolve(), Path(output).resolve()
    paths = set()
    for directory in ('rag', 'orchestration', 'safety', 'demo', 'scripts', 'tests'):
        paths.update((root / directory).rglob('*.py'))
    for directory in ('outputs/task7.4', 'outputs/task7.5', 'outputs/task7.6',
                      'outputs/task7.7', 'outputs/task7a', 'evaluation'):
        paths.update(p for p in (root / directory).rglob('*') if p.is_file())
    for path in sorted(paths):
        if output in path.parents or '__pycache__' in path.parts:
            continue
        name = str(path.resolve())
        if name not in current:
            current[name] = digest(path, sample)
    return current


def verify_unchanged(before, sample=lambda: None):
    after = {}
    for name, expected in before.items():
        path = Path(name)
        actual = digest(path, sample) if path.is_file() else None
        after[name] = actual
        if actual != expected:
            raise ValueError(f'Protected file changed during Task 7.7: {name}')
    return after
