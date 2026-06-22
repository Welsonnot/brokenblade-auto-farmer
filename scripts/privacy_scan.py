"""privacy_scan.py -- Find leaked personal info before publishing."""

# Run from project root so relative paths resolve.
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

import os, re

PATTERNS = [
    ('Windows user path',   re.compile(r'C:[\\/]Users[\\/][^\\/"\'\s]+', re.IGNORECASE)),
    ('Email address',       re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')),
    ('API key / secret',    re.compile(r'(api[_-]?key|secret|password|token)\s*[=:]\s*["\'][A-Za-z0-9_\-]{8,}', re.IGNORECASE)),
    ('Discord webhook',     re.compile(r'https?://discord(?:app)?\.com/api/webhooks', re.IGNORECASE)),
    ('Roblox cookie',       re.compile(r'\.ROBLOSECURITY|_\|WARNING_')),
    ('IPv4 address',        re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')),
    ('Name: wilez/wilhelm', re.compile(r'wilez|wilhelm', re.IGNORECASE)),
    ('Name: sanpedro',      re.compile(r'sanpedro', re.IGNORECASE)),
]

skip_dirs = {'.git', '__pycache__', 'recordings', 'yolo_data', 'logs', 'runs', 'models', 'Ollama'}
binary_exts = ('.png', '.jpg', '.jpeg', '.gif', '.pth', '.pt', '.zip', '.pyc', '.so', '.dll', '.exe', '.mp4', '.mkv')

results = {}
for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if d not in skip_dirs]
    for f in files:
        # Skip privacy_scan.py itself -- the patterns are in it by definition
        if f == 'privacy_scan.py':
            continue
        # Skip known binaries
        if any(f.lower().endswith(ext) for ext in binary_exts):
            continue
        path = os.path.join(root, f).replace(os.sep, '/')
        try:
            with open(path, encoding='utf-8') as fh:
                content = fh.read()
        except Exception:
            continue
        for name, pat in PATTERNS:
            for m in pat.finditer(content):
                ln = content[:m.start()].count('\n') + 1
                results.setdefault(name, []).append((path, ln, m.group()))

if not results:
    print('CLEAN -- no personal-data patterns found.')
else:
    for name, hits in results.items():
        print(f'\n=== {name} ({len(hits)} match) ===')
        for path, ln, match in hits:
            print(f'  {path}:{ln}  ->  {match[:100]}')
