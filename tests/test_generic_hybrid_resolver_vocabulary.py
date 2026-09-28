from pathlib import Path


def test_hybrid_resolver_is_generic_and_not_radio_owned() -> None:
    root = Path(__file__).resolve().parents[1]

    orchestrator = (root / "src/brynse/orchestrator.py").read_text(encoding="utf-8")

    radio_orchestrator = (
        root / "src/brynse/integrations/radio/orchestrator.py"
    ).read_text(encoding="utf-8")

    assert "class HybridCandidateResolver" in orchestrator
    assert "HybridRippingOrchestrator" not in radio_orchestrator
