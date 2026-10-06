

def test_build_orchestrator_propagates_routing_observation_callback(tmp_path) -> None:
    from brynse.integrations.radio.runner import RippingRunConfig, RippingRunner

    observed = []
    callback = observed.append

    runner = RippingRunner(
        RippingRunConfig(
            url="http://example.test/stream",
            output_directory=tmp_path,
            codec="mp3",
        )
    )

    orchestrator = runner._build_orchestrator(
        "mp3",
        on_routing_observation=callback,
    )

    resolver = orchestrator._resolver

    assert resolver._on_routing_observation is callback
