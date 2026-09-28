from pathlib import Path


def test_readme_documents_public_generic_streaming_cli() -> None:
    readme = Path("README.md").read_text()

    assert "No public CLI is provided yet." not in readme
    assert "brynse" in readme
    assert "source-agnostic" in readme
    assert "--output-dir" in readme


def test_readme_documents_machine_oriented_api() -> None:
    readme = Path("README.md").read_text()

    assert "Machine-oriented API" in readme
    assert "segment_source" in readme
    assert "run_segment_source" in readme
    assert "run_segment_json" in readme
    assert "SegmentRequest" in readme
    assert "SegmentResult" in readme
    assert "MachineError" in readme
    assert '"codec"' in readme
    assert '"provider"' in readme
    assert '"interval_seconds"' in readme
    assert "invalid_request" in readme
    assert "io_error" in readme
    assert "finalize_error" in readme
    assert "write_error" in readme
    assert "internal_error" in readme
