"""Build an unaccepted, payload-preserving decord metadata candidate for qualification.

This does not modify an installed distribution or RTP-LLM dependency requirements.
Linux native decoding and a normally distributed URL are required before adoption.
"""
import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

from elftools.elf.elffile import ELFFile
from packaging.tags import Tag
from packaging.utils import parse_wheel_filename

UPSTREAM_SHA256 = '51997f20be8958e23b7c4061ba45d0efcd86bffd5fe81c695d0befee0d442976'
OLD_TAG = 'cp36-cp36m-manylinux2010_x86_64'
NEW_TAG = 'py3-none-manylinux2010_x86_64'
BUILD = '1rtpfix'

def digest(data):
    return hashlib.sha256(data).hexdigest()

def record_digest(data):
    return 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('=')

def build(source, output):
    raw = source.read_bytes()
    if digest(raw) != UPSTREAM_SHA256:
        raise ValueError('input must be the exact original PyPI wheel')
    with zipfile.ZipFile(io.BytesIO(raw)) as original:
        infos = original.infolist()
        names = [x.filename for x in infos]
        if len(names) != len(set(names)) or any(Path(n).is_absolute() or '..' in Path(n).parts for n in names):
            raise ValueError('unsafe or duplicate wheel member')
        payload = {n: original.read(n) for n in names}
    wheel = 'decord-0.6.0.dist-info/WHEEL'
    record = 'decord-0.6.0.dist-info/RECORD'
    record_rows = list(csv.reader(io.StringIO(payload[record].decode())))
    records = {row[0]: row[1:] for row in record_rows}
    file_names = [info.filename for info in infos if not info.is_dir()]
    if len(records) != len(record_rows) or set(records) != set(file_names):
        raise ValueError('original RECORD member inventory mismatch')
    record_mismatches = []
    for name in file_names:
        data = payload[name]
        if name == record:
            if records[name] != ['', '']:
                raise ValueError('original RECORD self-reference must be empty')
        elif records[name] != [record_digest(data), str(len(data))]:
            record_mismatches.append({'path': name, 'recorded': records[name],
                                      'actual': [record_digest(data), str(len(data))]})
    expected_record_mismatch = [{'path': 'decord-0.6.0.dist-info/top_level.txt',
        'recorded': ['sha256=8TBMC8W9caRfSBphoy47j2wFImKqCOgWKD3JVELo5e0', '17'],
        'actual': ['sha256=2gcXRGxvur2Z1iLmIE4fxL6cmywVZYD__8rzDGIEZDM', '7']}]
    if record_mismatches != expected_record_mismatch:
        raise ValueError('original RECORD mismatch differs from the pinned upstream defect')
    original_wheel = payload[wheel].decode()
    if original_wheel.count('Tag: ') != 1 or 'Tag: ' + OLD_TAG not in original_wheel:
        raise ValueError('unexpected original wheel tags')
    native = []
    for name, data in payload.items():
        if not data.startswith(b'\x7fELF'):
            continue
        elf = ELFFile(io.BytesIO(data))
        if elf.elfclass != 64 or not elf.little_endian or elf['e_machine'] != 'EM_X86_64':
            raise ValueError('unexpected ELF architecture: ' + name)
        dynsym = elf.get_section_by_name('.dynsym')
        python_symbols = [x.name for x in dynsym.iter_symbols() if re.match(r'^_?Py(?:[A-Z_]|Init_)', x.name)]
        if python_symbols:
            raise ValueError('Python ABI references found: ' + repr(python_symbols))
        dynamic = elf.get_section_by_name('.dynamic')
        needed = [t.needed for t in dynamic.iter_tags() if t.entry.d_tag == 'DT_NEEDED']
        if any('python' in n.lower() for n in needed):
            raise ValueError('Python shared-library dependency found')
        versions = sorted({x.decode() for x in re.findall(rb'GLIBC(?:XX)?_[0-9]+(?:\.[0-9]+)+', data)})
        native.append({'path': name, 'sha256': digest(data), 'bytes': len(data), 'needed': needed,
                       'version_strings': versions, 'Python_ABI_symbols': python_symbols,
                       'machine': elf['e_machine'], 'ELF_class': elf.elfclass})
    if len(native) != 18:
        raise ValueError('native payload count changed')
    base = payload['decord/_ffi/base.py'].decode()
    if 'ctypes.CDLL' not in base or any('/_cy' in n and n.endswith('.so') for n in names):
        raise ValueError('ctypes-only wheel assumption not satisfied')
    payload[wheel] = (original_wheel.replace('Tag: ' + OLD_TAG, 'Tag: ' + NEW_TAG).rstrip('\n')
                      + '\nBuild: ' + BUILD + '\n').encode()
    rows = io.StringIO(newline='')
    writer = csv.writer(rows, lineterminator='\n')
    for name in file_names:
        if name == record:
            writer.writerow([name, '', ''])
        else:
            writer.writerow([name, record_digest(payload[name]), str(len(payload[name]))])
    payload[record] = rows.getvalue().encode()
    output.mkdir(parents=True, exist_ok=True)
    candidate = output / ('decord-0.6.0-' + BUILD + '-' + NEW_TAG + '.whl')
    if candidate.exists():
        raise FileExistsError(candidate)
    with zipfile.ZipFile(candidate, 'w') as archive:
        for info in infos:
            archive.writestr(info, payload[info.filename])
    with zipfile.ZipFile(io.BytesIO(raw)) as original, zipfile.ZipFile(candidate) as rebuilt:
        changed = [name for name in names if original.read(name) != rebuilt.read(name)]
        if set(changed) != {wheel, record}:
            raise ValueError('non-metadata payload changed')
    _, _, build_tag, tags = parse_wheel_filename(candidate.name)
    if tags != {Tag(*NEW_TAG.split('-'))} or build_tag != (1, 'rtpfix'):
        raise ValueError('candidate filename mismatch')
    evidence = {'accepted': False, 'status': 'CANDIDATE_REQUIRES_LINUX_VIDEO_AND_DISTRIBUTION_VALIDATION',
                'original_sha256': digest(raw), 'candidate_sha256': digest(candidate.read_bytes()),
                'candidate': str(candidate.resolve()), 'changed_members': changed,
                'unchanged_members': len(names) - len(changed), 'native_payload': native,
                'original_WHEEL': original_wheel, 'candidate_WHEEL': payload[wheel].decode(),
                'metadata_and_requirements_unchanged': True, 'original_RECORD_verified': False, 'original_RECORD_known_mismatches': record_mismatches,
                'other_original_RECORD_entries_verified': True,
                'all_original_member_names_preserved': True, 'Linux_runtime_executed': False,
                'limitations': ['ELF symbols do not replace a Linux load/decode test',
                               'manylinux platform suffix inherited from original filename; full auditwheel policy not yet certified',
                               'RTP-LLM dependency remains original; no installed WHEEL file edited']}
    (output / 'candidate-integrity.json').write_text(json.dumps(evidence, indent=2) + '\n')
    return evidence

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = build(args.source, args.output)
    print(json.dumps({k: result[k] for k in ['accepted', 'candidate', 'candidate_sha256', 'changed_members', 'unchanged_members']}))
