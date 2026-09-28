from pathlib import Path

import fluxtuner_ripper.integrations.radio.runner as runner_module
from fluxtuner_ripper.integrations.radio.orchestrator import CandidateBoundaryResolver
from fluxtuner_ripper.integrations.radio.runner import RippingRunConfig, RippingRunner


def test_radio_runner_uses_multisignal_resolver_for_aac_and_mp3(
    monkeypatch,
    tmp_path: Path,
) -> None:
    created: list[dict[str, object]] = []
    sentinels: list[object] = []

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

    aac = runner._build_orchestrator("aac")
    mp3 = runner._build_orchestrator("mp3")

    assert isinstance(aac, CandidateBoundaryResolver)
    assert isinstance(mp3, CandidateBoundaryResolver)

    assert aac._resolver is sentinels[0]
    assert mp3._resolver is sentinels[1]

    assert len(created) == 2

    for kwargs in created:
        extractor = kwargs["window_extractor"]
        decoder8 = kwargs["decoder8"]
        decoder16 = kwargs["decoder16"]

        assert extractor.search_radius_seconds == 24.0

        assert decoder8.output_sample_rate == 8000
        assert decoder16.output_sample_rate == 16000

        assert decoder8._ffmpeg_binary == "/custom/ffmpeg"
        assert decoder16._ffmpeg_binary == "/custom/ffmpeg"
