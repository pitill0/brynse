# Examples and validated use cases

This directory contains reproducible examples of Brynse in use.

Examples are classified by evidence level.

## Evidence levels

### Implemented

The execution path exists in the current codebase.

### Automated

The behavior is covered by automated repository tests.

### Live validated

The complete workflow has also been exercised against a real external source or
environment.

Live validation is complementary to automated testing. It is not a substitute
for deterministic repository tests.

## Validated use cases

| Use case | Implemented | Automated | Live validation |
| --- | --- | --- | --- |
| [Radio stream to tracks](live-radio-track-ripping.md) | Yes | Yes | Partial (prospective boundary validation complete) |

Additional examples should only be added to this table at the evidence level
actually demonstrated by the project.

Conceptual extension ideas belong in the architecture or extension
documentation until an executable and reproducible path exists.

`Partial` live validation means that real external sources have exercised part
of the workflow, but the complete end-to-end path has not yet been recorded as
validated.

For the radio use case, the production boundary resolver has completed frozen
prospective validation against Corpus E. The `Partial` classification remains
because the complete live CLI-to-finalized-track materialization workflow has not
yet been recorded as an end-to-end validation.

A use case should only be promoted to full live validation after the exact
documented workflow has been executed against a real source and its resulting
output has been verified.
