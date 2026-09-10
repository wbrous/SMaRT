from smrt.eval_code import _score_task

_PROMPT = "def add(a, b):\n"
_TEST = "def check(candidate):\n    assert candidate(2, 3) == 5\n    assert candidate(-1, 1) == 0\n"


# If this fails: a correct completion is no longer scored as passing —
# either the subprocess harness itself is broken, or the assembled
# prompt+completion+test program has a syntax/wiring bug.
def test_score_task_passes_correct_completion():
    completion = "    return a + b\n"
    assert _score_task(_PROMPT, completion, _TEST, "add") is True


# If this fails: an incorrect completion is no longer scored as failing —
# the subprocess is swallowing the AssertionError instead of surfacing a
# nonzero exit code.
def test_score_task_fails_incorrect_completion():
    completion = "    return a - b\n"
    assert _score_task(_PROMPT, completion, _TEST, "add") is False


# If this fails: a completion with a syntax error is no longer scored as
# failing (e.g. the subprocess call itself raises instead of returning a
# nonzero exit code, crashing the whole eval run on one bad sample).
def test_score_task_fails_on_syntax_error():
    completion = "    return a +\n"
    assert _score_task(_PROMPT, completion, _TEST, "add") is False
