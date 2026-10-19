from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from brynse.integrations.radio import cli
from brynse.integrations.radio import runner as runner_module
from brynse.integrations.radio.runner import RippingRunConfig, RippingRunError, RippingRunner
from brynse.source import BinaryIOStreamSource


@pytest.mark.parametrize("order", ["auto", "preserve", "artist-title", "title-artist"])
def test_cli_forwards_metadata_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, order: str
) -> None:
    configs = []

    class FakeRunner:
        def __init__(self, config, **kwargs):
            configs.append(config)

        def run(self, **kwargs):
            return SimpleNamespace(codec="mp3", metaint=417)

    monkeypatch.setattr(cli, "RippingRunner", FakeRunner)
    result = cli.main(
        [
            "https://example.invalid/stream",
            "--output",
            str(tmp_path),
            "--metadata-order",
            order,
        ]
    )

    assert result == 0
    assert configs[0].metadata_order == order


def test_cli_uses_automatic_metadata_order_by_default() -> None:
    args = cli._build_parser().parse_args(
        [
            "https://example.invalid/stream",
            "--output",
            "/tmp/tracks",
        ]
    )
    assert args.metadata_order == "auto"


def test_cli_rejects_unknown_metadata_order() -> None:
    with pytest.raises(SystemExit) as error:
        cli._build_parser().parse_args(
            [
                "https://example.invalid/stream",
                "--output",
                "/tmp/tracks",
                "--metadata-order",
                "unknown",
            ]
        )
    assert error.value.code == 2


def test_runner_rejects_metadata_order_before_opening_stream(tmp_path: Path) -> None:
    def unexpected_open(url):
        raise AssertionError("Invalid configuration must not open the stream")

    runner = RippingRunner(
        RippingRunConfig(
            url="https://example.invalid/stream",
            output_directory=tmp_path,
            metadata_order="unknown",
        ),
        stream_opener=unexpected_open,
    )
    with pytest.raises(RippingRunError, match="metadata_order"):
        runner.run()


@pytest.mark.parametrize("order", ["preserve", "artist-title", "title-artist"])
def test_runner_passes_metadata_order_to_writer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, order: str
) -> None:
    orders = []
    original_writer = runner_module.SessionOutputWriter

    def capture_writer(**kwargs):
        orders.append(kwargs["metadata_order"])
        return original_writer(**kwargs)

    monkeypatch.setattr(runner_module, "SessionOutputWriter", capture_writer)
    stream = io.BytesIO()
    runner = RippingRunner(
        RippingRunConfig(
            url="https://example.invalid/stream",
            output_directory=tmp_path,
            metadata_order=order,
        ),
        stream_opener=lambda url: BinaryIOStreamSource(
            stream,
            metadata={"Content-Type": "audio/mpeg", "icy-metaint": "417"},
        ),
    )

    runner.run()

    assert orders == [order]
    assert stream.closed
