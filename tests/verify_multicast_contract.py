from pathlib import Path
import subprocess
import tempfile
import shutil

root = Path.cwd()
evidence = root / 'docs/exploration/evidence/multicast-atomic-rejection'
evidence.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix='tusim-multicast-') as tmp:
    tmp = Path(tmp)
    subprocess.run(['git', 'archive', 'HEAD'], stdout=(tmp / 'base.tar').open('wb'), check=True)
    subprocess.run(['tar', '-xf', str(tmp / 'base.tar'), '-C', str(tmp)], check=True)
    shutil.copyfile(root / 'tests/test_multicast.c', tmp / 'tests/test_multicast.c')
    shutil.copyfile(root / 'Makefile', tmp / 'Makefile')
    flags = '-O1 -g -Wall -Wextra -std=c11 -fPIC -Itu_cmodel -fsanitize=address,undefined -fno-omit-frame-pointer'
    with (evidence / 'sanitizer.log').open('w') as log:
        for command in [['make', 'clean'], ['make', 'test-multicast', 'CFLAGS=' + flags, 'LDFLAGS=-lm -fsanitize=address,undefined']]:
            log.write('$ ' + ' '.join(command) + '\n')
            log.flush()
            subprocess.run(command, cwd=tmp, stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run(['make', 'clean'], cwd=tmp, stdout=subprocess.DEVNULL, check=True)
    source = tmp / 'tu_cmodel/dma_descriptor.c'
    original = source.read_text()
    needle = 'if (!descriptor_sram_spans_valid(desc)) {'
    assert original.count(needle) == 1
    # Safe mutant: only direct validation is bypassed. The existing private
    # multicast copier skips null/out-of-range targets instead of OOB copying.
    source.write_text(original.replace(needle, 'if (false) {', 1))
    with (evidence / 'mutation.log').open('w') as log:
        log.write('MUTATION: bypass direct descriptor SRAM-span prevalidation; keep private copy bounds checks and submission validator.\n')
        log.flush()
        result = subprocess.run(['make', 'test-multicast'], cwd=tmp, stdout=log, stderr=subprocess.STDOUT)
        log.write(f'\nMutant make exit: {result.returncode}\n')
        if result.returncode == 0:
            raise SystemExit('ERROR: expanded gate accepted partial-copy mutant')
    text = (evidence / 'mutation.log').read_text()
    assert text.count('rejected direct descriptor acquired completion state') == 6, text
print('PASS: sanitizer suite and six direct partial-copy/completion mutant failures')
