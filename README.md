# FluxTuner Ripper

Standalone native ICY radio stream ripping core extracted from FluxTuner.

> Status: pre-alpha extraction stage. The current goal is behavioral parity
> with the already-tested FluxTuner ripping core before further refactoring.

## Current scope

- Incremental ICY metadata parsing
- MP3 and AAC/ADTS frame timelines
- Encoded audio ring buffering
- Metadata semantic tracking
- Bounded acoustic-window extraction
- FFmpeg decode for analysis
- RMS acoustic profiling
- Boundary candidate matching and split policy
- Crossfade-aware byte-range planning
- MP3 finalization
- AAC/ADTS to M4A finalization
- Atomic track persistence

No public CLI is provided yet.

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
