import random

import pytest

from cir.perturb import (
    NONSENSE_EN,
    PERTURBATIONS,
    comment_pool,
    language_of,
    perturb_response,
)

PYTHON = "Use this:\n\n```python\nx = 1  # one\ny = x + 1\n'''doc'''\nprint(y)\n```\nDone."


def test_language_prefix_maps_csharp_and_skips_cpp():
    assert language_of("C#_12") == "CSharp"
    assert language_of("C++_12") == "C++"
    for perturbation in PERTURBATIONS:
        assert perturb_response(PYTHON, "C++", perturbation, random.Random(0)) == PYTHON


def test_code_block_perturbations_leave_prose_untouched():
    assert perturb_response(PYTHON, "Python", "remove_comments") == (
        "Use this:\n\n```python\nx = 1  \ny = x + 1\n\nprint(y)\n```\nDone."
    )
    assert perturb_response(PYTHON, "Python", "remove_whitespace") == (
        "Use this:\n\n```python\nx=1#oney=x+1'''doc'''print(y)```\nDone."
    )
    assert perturb_response(PYTHON, "Python", "replace_variables") == (
        "Use this:\n\n```python\nvar_0 = 1  # one\nvar_1 = var_0 + 1\n'''doc'''\n"
        "print(var_1)\n```\nDone."
    )


def test_replace_variables_skips_strings_and_reserved_names():
    code = (
        '```c\nint main() {\n    int count = 0;\n    char *s = "count";\n    return count;\n}\n```'
    )
    assert perturb_response(code, "C", "replace_variables") == (
        '```c\nint main() {\n    int var_0 = 0;\n    char *var_1 = "count";\n'
        "    return var_0;\n}\n```"
    )


def test_keyword_replacement_spans_prose_but_not_comments():
    result = perturb_response(PYTHON, "Python", "replace_keywords_en", random.Random(0))
    assert result.endswith("```\nDone.")
    assert "# one" in result
    assert "x = 1  # one\ny = x + 1\n'''doc'''\nprint(y)" in result
    # "Use this" contains no keyword, but prose keywords are replaced case-insensitively.
    prose = perturb_response("Return it as is.", "Python", "replace_keywords_en", random.Random(0))
    assert prose.split()[0] in NONSENSE_EN
    assert prose.split()[2] in NONSENSE_EN


def test_comment_perturbations_need_two_distinct_comments_or_a_pool():
    code = "a = 1  # first\nb = 2  # second\nc = 3  # third\n"
    swapped = perturb_response(code, "Python", "swap_comments_local", random.Random(0))
    # Comments are substituted in sequence, so a shuffle can also restore the input.
    assert swapped == perturb_response(code, "Python", "swap_comments_local", random.Random(0))
    assert any(
        perturb_response(code, "Python", "swap_comments_local", random.Random(seed)) != code
        for seed in range(10)
    )
    assert [line.split("#")[0] for line in swapped.splitlines()] == [
        "a = 1  ",
        "b = 2  ",
        "c = 3  ",
    ]
    single = "a = 1  # first\n"
    assert perturb_response(single, "Python", "swap_comments_local", random.Random(0)) == single
    assert perturb_response(single, "Python", "replace_comments_global", random.Random(0)) == single
    pool = comment_pool([{"id": "Python_1", "response": code}, {"id": "C++_2", "response": code}])
    assert pool == {"Python_1": ["# first", "# second", "# third"]}
    replaced = perturb_response(
        single, "Python", "replace_comments_global", random.Random(0), ["# other"]
    )
    assert replaced == "a = 1  # other\n"


def test_random_perturbations_require_a_generator_and_known_names():
    with pytest.raises(ValueError):
        perturb_response(PYTHON, "Python", "swap_comments_local")
    with pytest.raises(ValueError):
        perturb_response(PYTHON, "Python", "typo")
