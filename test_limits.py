import main
from main import check_limits

DAY = 1_800_000_000.0  # a fixed UTC timestamp, so tests don't depend on the real clock


def reset(daily_limit: int = 200) -> None:
    main._session_times.clear()
    main._daily.update(day="", count=0)
    main.DAILY_LIMIT = daily_limit


def test_session_limit_and_gap():
    reset()
    for i in range(4):
        assert check_limits("a", DAY + i * 10) is None
    assert "used your 4 questions" in check_limits("a", DAY + 40)
    assert check_limits("b", DAY + 40) is None  # other sessions are unaffected
    assert "wait a few seconds" in check_limits("b", DAY + 42)
    assert check_limits("a", DAY + 4 * 3600 + 1) is None  # first question has aged out of the window


def test_daily_cap_resets_next_day():
    reset(daily_limit=2)
    assert check_limits("a", DAY) is None
    assert check_limits("b", DAY) is None
    assert "limit for today" in check_limits("c", DAY)
    assert check_limits("c", DAY + 86400) is None


if __name__ == "__main__":
    test_session_limit_and_gap()
    test_daily_cap_resets_next_day()
    print("ok")
