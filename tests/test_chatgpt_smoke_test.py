from chatgpt_smoke_test import configure_smoke_parameters


def test_smoke_configuration_is_cpu_hybrid_without_rendering():
    cfg = configure_smoke_parameters()
    assert cfg.TEST_SET_NAME == "hybrid"
    assert cfg.TEST_SET_DIR == "DungeonMaps/test/hybrid"
    assert cfg.NUM_ROBOTS_MIN == cfg.NUM_ROBOTS_MAX == 4
    assert cfg.SAVE_GIFS is False
    assert cfg.VIZ_GRAPH_EDGES is False
    assert cfg.VIZ_GRAPH_EDGES_GROUND_TRUTH is False
    assert cfg.USE_GPU is False
