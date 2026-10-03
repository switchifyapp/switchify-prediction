#!/usr/bin/env python3
"""Prepare pinned FUTO sources locally; never vendor or redistribute the decoder/model."""
import argparse
from pathlib import Path
import zipfile
import re

from aac_experiment import sha

ARCHIVE_SHA = '465677ea63dcd53ed7ba58da9353c0d1dadf744d55a9e4b7978ccf6da2c79b6d'
COMMIT = '70a5d390c505a6bbcc4e14966e5628e43ca3f1fc'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sha(args.archive) != ARCHIVE_SHA:
        raise ValueError('FUTO archive checksum mismatch')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError('Use an empty output directory')
    prefix = f'android-keyboard-{COMMIT}/native/jni/'
    with zipfile.ZipFile(args.archive) as archive:
        (root / 'UPSTREAM-LICENSE.md').write_bytes(archive.read(f'android-keyboard-{COMMIT}/LICENSE.md'))
        for name in archive.namelist():
            if not name.startswith(prefix) or name.endswith('/'):
                continue
            relative = Path(name[len(prefix):])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe archive path')
            dest = root / 'source' / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(archive.read(name))
    source = root / 'source'
    flags = source / 'src/third_party/absl/flags/flag.cc'
    flag_text = flags.read_text(encoding='utf-8').replace('"src/common.h"', '"sentencepiece/common.h"').replace('"src/util.h"', '"sentencepiece/util.h"')
    flags.write_bytes(flag_text.encode())
    text = (source / 'org_futo_inputmethod_latin_xlm_LanguageModel.cpp').read_text(encoding='utf-8')
    # Keep the native class and sampling implementation, omit Android/JNI entry points.
    text = text[:text.index('struct SuggestionItemToRescore')]
    text = '\n'.join(line for line in text.splitlines() if not line.startswith('#include'))
    # Mechanical portability/safety fixes only, no ranking or sampling changes.
    text = text.replace('int seq_id_use_count[n_results];', 'std::vector<int> seq_id_use_count(n_results);')
    text = text.replace('auto start = s.begin();', 'if (s.empty()) return {};\n    auto start = s.begin();')
    text = text.replace('auto end = s.end();', 'if (start == s.end()) return {};\n    auto end = s.end();')
    text = text.replace('std::isspace(*start)', 'std::isspace(static_cast<unsigned char>(*start))').replace('std::isspace(*end)', 'std::isspace(static_cast<unsigned char>(*end))')
    # Upstream sometimes returns default/empty state on errors, and one decode
    # call is unchecked. A benchmark must not count those as successful queries.
    head, state = text.split('struct LanguageModelState {', 1)
    state = re.sub(r'return\s*\{\s*\};', 'throw std::runtime_error("FUTO decoder failure or unsupported model");', state)
    if state.count('llama_decode(ctx,') != 4:
        raise ValueError('Pinned decode call layout changed')
    state = state.replace('llama_decode(ctx,', 'checked_llama_decode(ctx,')
    header = '''#include <cmath>
#include <cstring>
#include <stdexcept>
#include "ggml/LanguageModel.h"
using std::isnan;
static int checked_llama_decode(llama_context *ctx, llama_batch batch) {
    const int result = llama_decode(ctx, batch);
    if (result != 0) throw std::runtime_error("FUTO llama_decode failed");
    return result;
}
'''
    text = header + head + 'struct LanguageModelState {' + state
    (root / 'decoder.inc').write_bytes(text.encode())
    print(source)


if __name__ == '__main__':
    main()
