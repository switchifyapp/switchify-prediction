#!/usr/bin/env python3
"""Verify every file listed in the accompanying SHA256SUMS on any supported OS."""
import hashlib
from pathlib import Path, PurePosixPath


def verify(root):
    root = root.resolve()
    seen = set()
    for line in (root / 'SHA256SUMS').read_text(encoding='utf-8').splitlines():
        expected, name = line.split('  ', 1)
        relative = PurePosixPath(name)
        if (len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected)
                or relative.is_absolute() or '..' in relative.parts or '\\' in name
                or ':' in name or name in seen):
            raise ValueError('Invalid checksum entry')
        seen.add(name)
        file = (root / name).resolve()
        if root not in file.parents:
            raise ValueError('Checksum path escapes bundle')
        digest = hashlib.sha256()
        with file.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f'Checksum mismatch: {name}')
    if not seen:
        raise ValueError('Empty checksum list')
    return len(seen)


if __name__ == '__main__':
    print(f'Verified {verify(Path(__file__).parent)} files.')
