# Bounded CPU integration fixtures for Tessera (PrismaQuant #251)

The failed CI run [34010631211](https://github.com/RobTand/prismaquant/actions/runs/34010631211/job/101425702369)
measured two existing E4M3 seam tests at their 300-second timeout. Both used
64×512 tensors and repeated the real L=14 CPU window recurrence. The test-file
Git blob, `64f06019f90577fcce3b0f7d4844ddb8bd76f3b5`, is identical in that run's
`465fc13796abb0d5eeec1b3e93036478cd74fdaf` source and this work's base
`3cb0d359`.

These assertions compare the real renderer with the exporter/readback and
check that removing a serializable-grid commitment changes admission while
rendering still succeeds. They do not require a large throughput workload.
`_render_seam_weight` now uses a complete 256-column rate superblock and at
least two full trellis histories, expanded by the family's tuple arity.
The window history comes from the window bits and the slowest rate in the
actual schedule; the fixture also covers the exporter's TCQ memory.
For the default E4M3 case this produces 12×256 weights. It retains the L=14
window, real numerical encoder, repeated render after registry removal,
cache invalidation, actual serialized artifact, exact equality assertion and
300-second timeout.

The producer and runtime pin remain
`ba582d476a3b6db9057ebd1385dc52926f171451`. This predates Tessera's separate
encoder-identity startup optimization; this change does not depend on moving
the pin or weakening its compatibility checks. Production behavior, artifact
bytes, menus and serving gates are unchanged.

## Qualification

Matched before/after profiles and the complete touched test file were
submitted through PrismaBuild. Results and profiler/Netdata receipts will be
recorded here before this draft is finalized.

The review worktree retains its original calibration symlinks. PB refuses
external symlink targets, so submission snapshots materialized the three
tracked calibration links into their exact contents, recording paths, targets,
byte counts and hashes in the adjacent artifact. These tests consume no
calibration dataset. The materialization was restored immediately after each
snapshot was sealed and is not part of the implementation commits.
