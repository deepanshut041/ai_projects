"""Restore the captured project into a new directory; never overwrite a checkout.

Usage: python restore_source.py /path/to/a/new-directory
Place source-bundle.json beside this script, or retain the course folder layout.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys


def restore(destination):
    destination=Path(destination).resolve()
    if destination.exists():raise ValueError('Choose a new destination; existing paths are never overwritten.')
    script=Path(__file__).resolve()
    bundle_path=script.parent/'source-bundle.json'
    if not bundle_path.is_file():bundle_path=script.parents[1]/'code/source-bundle.json'
    bundle=json.loads(bundle_path.read_text())
    files=[]
    for name,item in bundle['files'].items():
        relative=PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '.git' in relative.parts:
            raise ValueError('Unsafe captured source path')
        raw=item['text'].encode('utf-8')
        if hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('Source hash mismatch: '+name)
        files.append((relative,raw))
    destination.mkdir(parents=True)
    for relative,raw in files:
        path=destination/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    print(json.dumps({'path':str(destination),'files':len(files),'restored':True}))


if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit(__doc__)
    restore(sys.argv[1])
