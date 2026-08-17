from profile_ir2 import profile_callable


def test_profile_callable_returns_result_and_writes_profile(tmp_path):
    output = tmp_path / "sample.prof"

    def workload():
        return sum(range(100))

    result, stats = profile_callable(workload, dump_path=output)

    assert result == 4950
    assert output.is_file()
    assert stats.total_calls > 0
