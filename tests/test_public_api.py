"""Tests for the supported top-level public API."""


def test_safe_streaming_pipeline_is_available_from_package_root() -> None:
    from brynse import SafeStreamingPipeline, create_safe_streaming_pipeline

    assert SafeStreamingPipeline is not None
    assert callable(create_safe_streaming_pipeline)
