# FluxTuner Ripper

FluxTuner Ripper is a source-agnostic streaming ingestion and segmentation engine
that turns continuous streams into timestamped, lossless or minimally transformed
segments using pluggable boundary sources and policies.

## Current scope

- Incremental ICY metadata parsing
- MP3 and AAC/ADTS frame timelines
- Bounded in-memory analysis ring plus reclaimable disk spool retention
- Metadata semantic tracking
- Bounded acoustic-window extraction
- FFmpeg decode for analysis
- RMS acoustic profiling
- Pluggable fixed, manual, and external boundary providers
- Boundary candidate matching and split policy
- Crossfade-aware byte-range planning
- Streaming MP3 finalization
- Streaming AAC/ADTS to M4A finalization
- Atomic track persistence
- Incremental source-agnostic CLI segmentation

## Generic streaming CLI

`fluxtuner-ripper-segment` reads MP3 or AAC input incrementally from a file or
stdin. Without `--output-dir` it reports resolved boundaries as JSON. With
`--output-dir`, encoded input is retained in a reclaimable disk spool while a
bounded ring remains available for boundary analysis; completed segments are
materialized incrementally from the retained source.

Fixed-interval segmentation:

    fluxtuner-ripper-segment input.mp3 \
      --codec mp3 \
      --provider fixed \
      --interval 180 \
      --output-dir segments

Manual boundaries can be supplied by repeating `--boundary`:

    fluxtuner-ripper-segment input.aac \
      --codec aac \
      --provider manual \
      --boundary 120 \
      --boundary 245.5 \
      --output-dir segments

External boundaries are loaded from JSON or JSONL:

    fluxtuner-ripper-segment - \
      --codec mp3 \
      --provider external \
      --boundaries-file boundaries.jsonl \
      --output-dir segments

Use `--min-tail SECONDS` to merge a final tail shorter than the configured
threshold into the preceding segment. The default is 1 second.

## Development

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]'
make gate
```

FFmpeg is required for integration tests that exercise decoding and finalization.

## License

Mozilla Public License 2.0. See `LICENSE`.
