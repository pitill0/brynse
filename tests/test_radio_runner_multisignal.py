from pathlib import Path

import brynse.integrations.radio.runner as runner_module
from brynse.integrations.radio.orchestrator import CandidateBoundaryResolver
from brynse.integrations.radio.runner import RippingRunConfig, RippingRunner


def test_radio_runner_uses_multisignal_resolver_for_aac_and_mp3(
    monkeypatch,
    tmp_path: Path,
) -> None:
    created: list[dict[str, object]] = []
    sentinels: list[object] = []

    incoming_created: list[str] = []
    incoming_sentinels: list[object] = []

    def fake_incoming_pipeline(*, ffmpeg_binary: str, on_observation=None):
        incoming_created.append((ffmpeg_binary, on_observation))
        pipeline = object()
        incoming_sentinels.append(pipeline)
        return pipeline

    monkeypatch.setattr(
        runner_module,
        "build_incoming_boundary_pipeline",
        fake_incoming_pipeline,
        raising=False,
    )

    class FakeMultiSignalResolver:
        pass

    def fake_multisignal_resolver(**kwargs):
        created.append(kwargs)
        resolver = FakeMultiSignalResolver()
        sentinels.append(resolver)
        return resolver

    monkeypatch.setattr(
        runner_module,
        "MultiSignalAcousticCandidateResolver",
        fake_multisignal_resolver,
        raising=False,
    )

    runner = RippingRunner(
        RippingRunConfig(
            url="https://example.invalid/stream",
            output_directory=tmp_path,
            ffmpeg_binary="/custom/ffmpeg",
        )
    )

    def observe_incoming(observation):
        pass

    aac = runner._build_orchestrator(
        "aac",
        on_incoming_observation=observe_incoming,
    )
    mp3 = runner._build_orchestrator(
        "mp3",
        on_incoming_observation=observe_incoming,
    )

    assert isinstance(aac, CandidateBoundaryResolver)
    assert isinstance(mp3, CandidateBoundaryResolver)

    assert aac._resolver is sentinels[0]
    assert mp3._resolver is sentinels[1]

    assert len(created) == 2

    assert incoming_created == [
        ("/custom/ffmpeg", observe_incoming),
        ("/custom/ffmpeg", observe_incoming),
    ]
    assert len(incoming_sentinels) == 2

    for index, kwargs in enumerate(created):
        assert kwargs["incoming_refiner"] is incoming_sentinels[index]
        extractor = kwargs["window_extractor"]
        decoder8 = kwargs["decoder8"]
        decoder16 = kwargs["decoder16"]

        assert extractor.search_radius_seconds == 24.0

        assert decoder8.output_sample_rate == 8000
        assert decoder16.output_sample_rate == 16000

        assert decoder8._ffmpeg_binary == "/custom/ffmpeg"
        assert decoder16._ffmpeg_binary == "/custom/ffmpeg"
