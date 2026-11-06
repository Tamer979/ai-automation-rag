from evals.run_eval import grade, passes


def test_grading():
    assert passes("It costs $3,000 to $10,000.", ["3000", "10000"])
    assert passes("Ship it in 2 weeks.", ["two weeks|2 weeks"])
    assert not passes("Between $500 and $1,000.", ["500", "3000"])
    assert grade({"unanswerable": True}, "The context does not mention a CRM.", []) == (True, None)
    assert grade({"unanswerable": True}, "HubSpot is recommended.", []) == (False, None)
    assert grade({"must": ["13"]}, "Only 13%.", ["only 13% of AI projects"]) == (True, True)


if __name__ == "__main__":
    test_grading()
    print("ok")
