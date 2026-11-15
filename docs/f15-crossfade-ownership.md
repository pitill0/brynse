# F-1/F-5: crossfade ownership and evidence limits

## Accepted publication contract

The radio writer accounts for every received encoded byte exactly once.
The coverage ledger is authoritative for byte ownership and uses half-open
intervals with zero gap/overlap tolerance.

- A verified hard cut partitions the encoded stream at one frame boundary.
- An independently verified silence gap has separate outgoing and incoming
  boundaries. Its intervening bytes are published as `unowned`.
- A **recognized** crossfade carries two ordered bounds. The radio writer
  publishes the outgoing-only prefix, the mixture once as `unowned`, and the
  incoming-only suffix. It never publishes the same mixed bytes in both tracks.
- Without sufficiently independent evidence for *both* crossfade bounds,
  classification must abstain from asserting an exact overlap. The routing
  trace may record `AMBIGUOUS`; a shared hard-cut partition retains byte
  continuity, **not** a proven semantic allocation of the mixed samples.

An `unowned` artifact identifies audio with no exclusive track attribution.
It is not evidence of lost or discarded source bytes. Very short MP3 artifacts
may retain the original encoded bytes without being standalone-playable.

## Why overlap bounds are not always identifiable

ICY title changes are semantic hints, not waveform timestamps. A single
encoded stream containing a mixture does not expose the individual sources.
Different source pairs and overlap envelopes can produce the same observed
mixture. A spectral progression may suggest crossfading but cannot, by itself,
prove exact starting and ending byte offsets.

Do not infer overlap duration from a fixed number of seconds or absorb errors
with a tolerance. When the evidence is insufficient, leave the semantic
classification unresolved and preserve the bytes.

## Claudio ground-truth replay (November 2026)

The original six-song harness has a hard cut near 70 s, a silent interval
`[140.0, 141.5)`, a fade at 211.5 s, a crossfade over `[275.5, 281.5)`,
and a short terminal tail. Replays with ICY deltas of -6, 0 and +6 s verified:

- Hard cut: published around 70.03–70.06 s.
- Silence: outgoing around 140.02–140.04 s; incoming around
  141.51–141.53 s, aligned to source MP3 frames.
- Independent content mapping: no missing and no duplicated audio across
  *all* published track and `unowned` artifacts.
- **Still unresolved:** the synthetic crossfade is routed as
  `AMBIGUOUS/HARD_CUT`; its true overlap bounds are not independently proven
  by the runtime. This result must not be reported as an identified crossfade.

The remaining question for Claudio is whether another independent signal
or revised acceptance semantics can justify a bounded overlap in the absence
of original source stems. Until then, do not assert precise overlap ownership.
