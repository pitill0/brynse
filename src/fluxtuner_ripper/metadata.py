from __future__ import annotations

from fluxtuner_ripper.models import (
    MetadataSemanticDecision,
    SplitKind,
    TimedMetadataEvent,
    TrackCandidate,
)


class MetadataSemanticTracker:
    """Classify metadata lifetimes before any acoustic split analysis.

    A metadata title becomes durable only after it has survived at least
    ``transient_threshold_seconds``. Titles that are replaced earlier are
    classified as ``NO_BOUNDARY``.
    """

    def __init__(self, transient_threshold_seconds: float = 8.0) -> None:
        if transient_threshold_seconds <= 0:
            raise ValueError("transient_threshold_seconds must be greater than zero")

        self._threshold = transient_threshold_seconds
        self._current: TimedMetadataEvent | None = None
        self._current_confirmed = False

    @property
    def transient_threshold_seconds(self) -> float:
        return self._threshold

    @property
    def current_title(self) -> str | None:
        return self._current.title if self._current is not None else None

    def feed(
        self, event: TimedMetadataEvent
    ) -> tuple[tuple[MetadataSemanticDecision, ...], tuple[TrackCandidate, ...]]:
        """Consume one timed metadata event.

        Returns semantic decisions about completed metadata lifetimes plus any
        newly confirmed durable track candidates.
        """

        decisions: list[MetadataSemanticDecision] = []
        candidates: list[TrackCandidate] = []

        if self._current is None:
            self._current = event
            return (), ()

        if event.title == self._current.title:
            if not self._current_confirmed:
                lifetime = event.audio_time_seconds - self._current.audio_time_seconds
                if lifetime >= self._threshold:
                    self._current_confirmed = True
                    candidates.append(
                        TrackCandidate(
                            title=self._current.title,
                            start_offset=self._current.audio_offset,
                            start_time_seconds=self._current.audio_time_seconds,
                            confirmed_at_offset=event.audio_offset,
                            confirmed_at_time_seconds=event.audio_time_seconds,
                        )
                    )
            return tuple(decisions), tuple(candidates)

        lifetime = event.audio_time_seconds - self._current.audio_time_seconds
        if lifetime < 0:
            raise ValueError("metadata events must be monotonic in audio time")

        decisions.append(
            MetadataSemanticDecision(
                title=self._current.title,
                kind=(
                    SplitKind.HARD_CUT
                    if self._current_confirmed or lifetime >= self._threshold
                    else SplitKind.NO_BOUNDARY
                ),
                start_offset=self._current.audio_offset,
                start_time_seconds=self._current.audio_time_seconds,
                lifetime_seconds=lifetime,
            )
        )

        if not self._current_confirmed and lifetime >= self._threshold:
            candidates.append(
                TrackCandidate(
                    title=self._current.title,
                    start_offset=self._current.audio_offset,
                    start_time_seconds=self._current.audio_time_seconds,
                    confirmed_at_offset=event.audio_offset,
                    confirmed_at_time_seconds=event.audio_time_seconds,
                )
            )

        self._current = event
        self._current_confirmed = False
        return tuple(decisions), tuple(candidates)

    def confirm_current(
        self, *, audio_offset: int, audio_time_seconds: float
    ) -> tuple[TrackCandidate, ...]:
        """Confirm the current title after enough audio time has elapsed.

        This is used when no repeated metadata packet arrives for the same
        title but the stream has nevertheless advanced beyond the transient
        threshold.
        """

        if self._current is None or self._current_confirmed:
            return ()

        if audio_time_seconds < self._current.audio_time_seconds:
            raise ValueError("audio_time_seconds must be monotonic")

        lifetime = audio_time_seconds - self._current.audio_time_seconds
        if lifetime < self._threshold:
            return ()

        self._current_confirmed = True
        return (
            TrackCandidate(
                title=self._current.title,
                start_offset=self._current.audio_offset,
                start_time_seconds=self._current.audio_time_seconds,
                confirmed_at_offset=audio_offset,
                confirmed_at_time_seconds=audio_time_seconds,
            ),
        )
