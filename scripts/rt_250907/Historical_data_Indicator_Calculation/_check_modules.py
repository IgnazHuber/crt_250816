import subprocess, sys

missing = []
for mod in ('scipy', 'sklearn'):
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)

if missing:
    print('Missing modules:', ', '.join(missing))
    sys.exit(1)
print('All required packages present.')
