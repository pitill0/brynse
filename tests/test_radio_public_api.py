"""Tests for the supported public radio integration API."""


def test_radio_runner_api_is_available_from_integration_package() -> None:
    from brynse.integrations.radio import (
        RippingRunConfig,
        RippingRunError,
        RippingRunner,
        RippingRunResult,
    )

    assert RippingRunConfig is not None
    assert issubclass(RippingRunError, RuntimeError)
    assert RippingRunResult is not None
    assert RippingRunner is not None


def test_radio_public_api_is_deliberately_bounded() -> None:
    import brynse.integrations.radio as radio

    assert radio.__all__ == [
        "RippingRunConfig",
        "RippingRunError",
        "RippingRunResult",
        "RippingRunner",
    ]
