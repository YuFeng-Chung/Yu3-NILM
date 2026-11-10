"""Fetch the published PLAID 2014 archive; verify its publisher-supplied checksum."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    url = 'https://api.figshare.com/v2/articles/11605074'
    with urllib.request.urlopen(url, timeout=60) as response:
        metadata = json.load(response)
    assert metadata['title'] == 'PLAID 2014'
    file = next(f for f in metadata['files'] if f['id'] == 21003648)
    (args.output / 'figshare_metadata.json').write_text(json.dumps(metadata, indent=2))
    target = args.output / 'plaid_2014_published.zip'
    expected = file['computed_md5']
    if target.exists():
        assert hashlib.md5(target.read_bytes()).hexdigest() == expected
        print('Published archive already verified.', flush=True); return
    partial = target.with_suffix('.zip.part'); total = 0; digest = hashlib.md5()
    with urllib.request.urlopen(file['download_url'], timeout=90) as source, partial.open('wb') as dest:
        while block := source.read(1024 * 1024):
            dest.write(block); digest.update(block); total += len(block)
            if total % (20 * 1024 * 1024) == 0:
                print(f'Downloaded {total // (1024*1024)} MiB / {file["size"] // (1024*1024)} MiB', flush=True)
    assert total == file['size'], (total, file['size'])
    assert digest.hexdigest() == expected
    partial.rename(target)
    print(f'PASS published archive MD5 {expected}; {total} bytes.', flush=True)


if __name__ == '__main__': main()
