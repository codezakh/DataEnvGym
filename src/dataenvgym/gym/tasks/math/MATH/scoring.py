"""
This is a slightly modified version of the scoring code from
https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/minerva_math/utils.py
"""

import re
import signal
from typing import Optional
import sympy
from sympy.parsing.latex import parse_latex
from loguru import logger
import jinja2


import datasets

# Stop the parsing code below from spamming a ton of logs.
logger.disable(__name__)


# taken from
# https://github.com/wellecks/lm-evaluation-harness/blob/master/lm_eval/tasks/minerva_math.py
def doc_to_text(doc: dict) -> str:
    return "Problem:" + "\n" + doc["problem"] + "\n\n" + "Solution:"


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        last_slashboxed = find_last_boxed_string(doc["solution"])
        # The answer is expected to the be the last \boxed expression in the
        # solution. This should _always_ be true for the ground truth solutions
        # in the dataset.
        assert last_slashboxed is not None
        answer = normalize_final_answer(remove_slashbox(last_slashboxed))
        out_doc = {
            "problem": doc["problem"],
            "solution": doc["solution"],
            "answer": answer,
        }
        if getattr(doc, "few_shot", None) is not None:
            out_doc["few_shot"] = True
        return out_doc

    return dataset.map(_process_doc)


def list_fewshot_samples() -> list[dict]:
    return [
        {
            "problem": "Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}",
            "solution": "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.\nFinal Answer: The final answer is $[2,5)$. I hope it is correct.",
            "few_shot": "1",
        },
        {
            "problem": "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
            "solution": "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$\nFinal Answer: The final answer is $24$. I hope it is correct.",
            "few_shot": "1",
        },
        {
            "problem": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
            "solution": "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:\n\\begin{align*}\n30n&=480\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}\nFinal Answer: The final answer is $16$. I hope it is correct.",
            "few_shot": "1",
        },
        {
            "problem": "If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero.",
            "solution": "If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\nFinal Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct.",
            "few_shot": "1",
        },
    ]


def score_candidate_answer(ground_truth_answer: str, candidate: str) -> int:
    unnormalized_answer = get_unnormalized_answer(candidate)
    answer = normalize_final_answer(unnormalized_answer)

    return int(is_equiv(answer, ground_truth_answer))


def find_last_boxed_string(s: str) -> Optional[str]:
    r"""
    Find the last \boxed{...} or \fbox{...} expression in a string.
    """
    idx = s.rfind("\\boxed")
    if "\\boxed " in s:
        return "\\boxed " + s.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = s.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(s):
        if s[i] == "{":
            num_left_braces_open += 1
        if s[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = s[idx : right_brace_idx + 1]

    return retval


def remove_slashbox(s: str) -> str:
    r"""
    Remove the \boxed{...} or \fbox{...} expression from a string.
    """
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


class Timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def is_equiv(x1: str, x2: str) -> bool:
    """
    x1 and x2 are normalized latex string
    """
    try:
        with Timeout(seconds=5):
            try:
                parsed_x1 = parse_latex(x1)
                parsed_x2 = parse_latex(x2)
            except (
                sympy.parsing.latex.errors.LaTeXParsingError,  # type: ignore[attr-defined]
                sympy.SympifyError,
                TypeError,
            ):
                logger.debug(f"couldn't parse one of {x1} or {x2}")
                return x1 == x2

            try:
                diff = parsed_x1 - parsed_x2
            except TypeError:
                logger.debug(f"couldn't subtract {x1} and {x2}")
                return x1 == x2

            try:
                if sympy.simplify(diff) == 0:
                    return True
                else:
                    return False
            except ValueError:
                logger.debug(
                    f"Had some trouble simplifying when comparing {x1} and {x2}"
                )
                return x1 == x2
    except TimeoutError:
        logger.debug(f"Timed out comparing {x1} and {x2}")
        return x1 == x2
    except ImportError as e:
        logger.error(e)
        raise
    except Exception as e:
        logger.opt(exception=True).error(f"Failed comparing {x1} and {x2} with {e}")
        return False


EXPECTED_ANSWER_PREFIX = "Final Answer: The final answer is"

EXPECTED_ANSWER_TEMPLATE = jinja2.Template(
    """{{ chain_of_thought }}\n{{ expected_answer_prefix }} $\\boxed{ {{ answer }} }$."""
)


def render_solution_for_scoring(chain_of_thought: str, final_answer: str) -> str:
    return EXPECTED_ANSWER_TEMPLATE.render(
        chain_of_thought=chain_of_thought,
        expected_answer_prefix=EXPECTED_ANSWER_PREFIX,
        answer=final_answer,
        trim_blocks=True,
        lstrip_blocks=True,
    )


# BACKUP_PATTERN = r"\$\\boxed\{(?P<answer>[^}]+)\}\$(?![^$]*\$\\boxed)"
BACKUP_PATTERN = r"(?P<boxed_answer>\$\\boxed\{[^}]+\}\$)(?![^$]*\$\\boxed)"


def get_unnormalized_answer(text: str) -> str:
    INVALID_ANSWER = "[invalidanswer]"
    end_seq = "I hope it is correct."
    text += end_seq
    pattern = EXPECTED_ANSWER_PREFIX + r"(.*?)\s*" + re.escape(end_seq)
    match = re.search(
        # EXPECTED_ANSWER_PREFIX + r"(.*?).\s*I hope it is correct.",
        pattern,
        text,
    )
    if match:
        return match.group(1).strip()

    # Matches the Llama3 eval template.
    pattern = "Therefore, the final answer is" + r"(.*?)\s*" + re.escape(end_seq)
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()

    match = re.search(BACKUP_PATTERN, text)
    if match:
        return match.group("boxed_answer").strip()

    return INVALID_ANSWER


SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.

    Copied character for character from appendix D of Lewkowycz et al. (2022)
    """
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer
