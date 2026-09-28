# Ripper research closeout — 2026-09-28

## Scope

This document closes the current boundary-selection research cycle for
Fluxtuner Ripper.

> **Historical sequencing note:** this closeout records two successive states from
> the same research cycle. The earlier sections describe the pre-transfer state,
> when production still used `HybridCandidateResolver`. The later **Production
> transfer and prospective validation** section records the subsequent frozen
> transfer of `MULTISIGNAL-RESOLVER-v2-MINIMAX`, its untouched Corpus E evaluation,
> and its acceptance as the production boundary resolver. The later section is the
> final production status; the earlier statements are retained as historical
> evidence rather than rewritten retrospectively. The project was subsequently
> renamed Brynse; the historical project name is intentionally preserved here.

The production radio path remains based on `HybridCandidateResolver`.
No experimental selector from this research cycle is promoted directly
into production.

## Validated findings

### Candidate generation

The FULL UNION candidate generator:

- BASIN
- direct-D2

was prospectively validated and showed that candidate generation is not
the primary bottleneck.

On Corpus D, the incoming-start oracle over FULL UNION candidates reached:

- MAE: 0.107 s
- median absolute error: 0.050 s
- 23/28 within 0.25 s
- 28/28 within 0.50 s
- max error: 0.400 s

This shifts the main problem from candidate generation to candidate
selection / interpretation.

### Forward novelty

Forward novelty showed useful local onset-sensitive information but was
not strong enough as a standalone selector.

Dense evaluation over C+D:

- N=52
- peak within 0.25 s: 26
- peak within 0.50 s: 34
- peak within 1.00 s: 43
- median absolute peak offset: 0.250 s

When used to rank FULL UNION candidates, novelty alone remained weak:

- TOP1: 13/52
- TOP3: 20/52
- TOP5: 23/52

Conclusion:

Forward novelty is useful complementary evidence, not a standalone
boundary selector.

### Signal complementarity

RRF and forward novelty were shown to be materially complementary.

For oracle TOP-k membership on C+D:

- TOP1 union: 23/52
- TOP3 union: 33/52
- TOP5 union: 39/52

This supports a multi-evidence resolver architecture.

### RankSum

A parameter-free RankSum resolver was evaluated:

    rank_sum = rank_rrf + rank_novelty

Development result:

- ALL MAE: 1.034 s
- median: 0.400 s

Baseline RRF:

- ALL MAE: 1.051 s
- median: 0.403 s

Conclusion:

RankSum was marginally positive but not strong enough for prospective
promotion.

### MINIMAX

A second parameter-free resolver was evaluated:

    score(candidate) = max(rank_rrf, rank_novelty)

Tie-breaks:

1. lower rank sum
2. better RRF rank
3. better novelty rank
4. earlier candidate

Development results on C+D:

| Metric | RRF | MINIMAX |
|---|---:|---:|
| MAE | 1.051 s | 0.902 s |
| Median AE | 0.403 s | 0.388 s |
| <= 0.25 s | 15/52 | 19/52 |
| <= 0.50 s | 28/52 | 30/52 |
| <= 1.00 s | 36/52 | 36/52 |
| Max error | 5.200 s | 4.119 s |

By corpus:

- Corpus C MAE: 1.298 -> 1.002 s
- Corpus D MAE: 0.840 -> 0.816 s

Hard cuts improved strongly:

- RRF MAE: 0.764 s
- MINIMAX MAE: 0.366 s
- max error: 5.200 -> 1.594 s

Crossfades remained largely unresolved:

- RRF MAE: 1.567 s
- MINIMAX MAE: 1.529 s
- threshold counts were effectively unchanged

Exclusions regressed:

- RRF MAE: 0.311 s
- MINIMAX MAE: 0.745 s

Conclusion:

MINIMAX is the best development resolver tested in this research cycle,
but is not sufficient to justify direct production promotion.

## Regime observability

A final diagnostic tested whether a hard-cut-like regime could be
identified blind from observable ranking geometry.

No stable cross-corpus signature was found.

Aggregate differences existed, but distributions overlapped and several
observables changed materially between Corpus C and Corpus D.

Conclusion:

Do not introduce a truth-type-conditioned resolver or a heuristic
hard-cut classifier from these results.

## Fingerprint identity experiments

Shazam / fingerprint experiments confirmed that identity trajectories can
provide useful evidence about stream identity transitions.

However:

- winner identity is not exclusivity
- fingerprint switch time is not equivalent to incoming-start time
- direct fingerprint-based boundary selection failed prospectively
- exact reference matching requires a genuinely independent reference
  catalog

Fingerprint evidence therefore remains a future provider, not the
primary acoustic boundary selector.

## Architecture conclusion

The research supports the following separation:

    continuous stream
        -> retained stream
        -> boundary proposal providers
        -> descriptive evidence
        -> multi-evidence resolver
        -> segmenter
        -> sink

Different evidence types must not be treated as temporal synonyms.

Examples include:

- incoming onset
- outgoing end
- acoustic change
- identity transition
- semantic cut
- metadata boundary

The production implementation should preserve resolver replaceability
and avoid embedding domain-specific semantics into the generic core.

## Production status

Current radio runtime:

    HybridCandidateResolver

Current generic CLI:

    DefaultCandidateResolver

Current BASIN + STRUCTURAL + BoundaryReconciler path:

    shadow / experimental

The FULL UNION + RRF + forward-novelty + MINIMAX pipeline was validated
in the research harness, but production does not yet implement the same
candidate/ranking space.

For that reason, MINIMAX is not ported directly into production in this
closeout.

Doing so correctly would require a real multi-evidence ranking layer,
not an ad-hoc modification to `BoundaryReconciler`.

## Known limitation

Gradual transitions and crossfades remain substantially harder than
abrupt transitions.

The current system should be understood as estimating a useful
segmentation boundary from acoustic and semantic evidence. It does not
guarantee exact perceptual incoming onset for every transition.

## Holdout status

Corpus E remains untouched.

Corpus C and Corpus D are considered consumed development corpora.

Corpus E must remain reserved for a future genuinely frozen resolver or
a materially new evidence provider.

## Decision

Research on the current acoustic resolver is stopped here deliberately.

Do not open another tuning cycle, feature family, embedding model, or
post-hoc selector iteration without a new architectural reason.

Future work, if resumed, should focus on one of:

- a proper production multi-evidence ranking layer
- an independent fingerprint/reference provider
- metadata / external boundary providers
- a genuinely new evidence family

The current production ripper remains on the validated Hybrid resolver
path.


## Production transfer and prospective validation

The frozen MULTISIGNAL-RESOLVER-v2-MINIMAX was transferred to
production and prospectively evaluated against untouched Corpus E.

Production implementation commit:

    b68ecf8 Freeze radio multisignal acoustic window

Final pre-E preregistration:

    cfb9bc6 Finalize multisignal resolver freeze for Corpus E

Corpus E contained 18 transitions. Three historically annotated
exclusions were omitted from acoustic-resolver scoring because exclusion
handling is a session-level responsibility established before evaluation.

Resolver evaluation:

    N       = 15
    MAE     = 0.562 s
    median  = 0.200 s
    <=0.25  = 9/15
    <=0.50  = 10/15
    <=1.00  = 12/15
    max     = 2.375 s

For comparison, the frozen C+D development result was:

    N       = 46
    MAE     = 0.922 s
    median  = 0.397 s
    <=0.25  = 15/46
    <=0.50  = 26/46
    <=1.00  = 32/46
    max     = 4.119 s

Corpus E therefore did not show an out-of-sample collapse. The frozen
resolver generalized prospectively and is accepted as the production
boundary resolver.

No tuning was performed from Corpus E results.

Evaluation artifacts:

    manifest:
    afbfb9cc137c627215ebb11b7c44d2f03fbe2ce38990e7e382f9b7808caa3dd0

    evaluator:
    a669988a75308e481a5a3f239d6f7e51f8cf4c3ca1d16545b75e54cb401c8036

    frozen predictions:
    dfce6c6106955f9e2947aff27688e20173c61e6ebd7520efe6cd887c946e4b2d

    human truth:
    a4dbdc5cd73e4155831a389412e0cb6a4c3dc0cd11cda1b15af006742b6f2f4e

Status:

    research: CLOSED
    production transfer: COMPLETE
    prospective validation: PASSED
    post-E tuning: NONE
