from pathlib import Path


def test_readme_documents_public_generic_streaming_cli() -> None:
    readme = Path("README.md").read_text()

    assert "No public CLI is provided yet." not in readme
    assert "fluxtuner-ripper-segment" in readme
    assert "source-agnostic" in readme
    assert "--output-dir" in readme
