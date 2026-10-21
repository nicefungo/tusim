# im2col point-query audit evidence

`commands.json` records input HEAD, complete command arguments, working
locations and return codes. `runner.py` is the exact executed orchestration
script; it ran from baseline HEAD with five explicit overlaid files. Archives
exclude external symlink fixtures, which these C tests do not consume.
`INPUT_SHA256SUMS` binds all C/header files plus overlaid build/test inputs;
`SHA256SUMS` binds every retained artifact except itself. Logs are gzip with
mtime zero; decompression restores the unedited tool output.

Initial live clean-build/quick logs are retained separately from the clean
archived baseline. New point regression source is taken from the final overlay
for both old and new code. The baseline RED subprocess exits 1 (11/11 probes
fail); the complete baseline regression exits 1 (5,530 mismatches). Final point
checks and all asserted build/quick/focused/sanitizer commands exit zero.

`make test` is recorded as an observation, not an acceptance gate: baseline and
final stop in attention with 7/9 and 6/9 passes respectively. All three baseline
attention reruns (7/9, 8/9, 7/9) are retained. No output was dropped to stabilize
that failure set; this evidence does not attribute its root cause.

`index-commands.json` and `index-*.log.gz` additionally record a clean rebuild
and focused/quick run of the exact staged snapshot before publication. The
index verification checks that the archived implementation inputs match
`INPUT_SHA256SUMS`. Its tree ID precedes insertion of its own execution logs;
those logs do not change executable inputs.

Primary reproduction after checkout:

```sh
make clean && make
make test-quick
make test-agen-im2col-point test-agen-im2col-bounds test-config test-conv test-scatter-gather
```

For the historical RED, extract regular files/directories from
`git archive 69330281ac53c1af9fb0764e5d971806e1b0ae5c`, copy only the new
`tests/test_agen_im2col_point.c` into it, build its library, then compile the
probe against its explicit static archive and execute `--red` (expect exit 1).
The full probe without `--red` also exits 1. Use the exact commands recorded in
`commands.json`; temporary working directories themselves are not retained.
The runner is a record of that dirty-baseline execution, not a command to run
blindly from a different HEAD or over an existing evidence directory.
