"""
Utility constants and functions for math answer evaluation.
Used by judger.py.
"""

import re

# ─── Parenthesis map ─────────────────────────────────────────────────────────

PAREN_MAP = {
    "(": ")",
    "[": "]",
    "{": "}",
    "\\{": "\\}",
    "\\langle": "\\rangle",
    "\\left(": "\\right)",
    "\\left[": "\\right]",
    "\\left{": "\\right}",
}

# ─── Answer extraction prefixes ──────────────────────────────────────────────

GSM8K_ANS_PREFIX = "#### "
PRM800K_ANS_PRRFIX = "# Answer\n\n"

# ─── Strings to strip / replace ──────────────────────────────────────────────

SIMPLE_RM_STRS = [
    "\\!", "\\,", "\\;", "\\:",
    "\\quad", "\\qquad",
    "\\displaystyle", "\\textstyle", "\\scriptstyle",
    "\\scriptscriptstyle", "\\normalsize",
    "\\hspace{0.1in}", "\\hspace{0.2in}", "\\hspace{1em}",
    "\\medspace", "\\thinspace", "\\thickspace",
    "\\enspace", "\\negthinspace",
    "~",
]

SIMPLE_REPLACE_MAP = {
    "\\times": "*",
    "\\cdot": "*",
    "\\div": "/",
    "\\leq": "<=",
    "\\geq": ">=",
    "\\neq": "!=",
    "×": "*",
    "÷": "/",
    "π": "\\pi",
    "…": "...",
    "−": "-",
    "–": "-",
    "\u2212": "-",       # unicode minus
    "\u00b7": "*",       # middle dot
    "\u00d7": "*",       # multiplication sign
    "\u00f7": "/",       # division sign
    "&#960;": "\\pi",
    "\\uparrow": "",
    "\\downarrow": "",
    "\\nearrow": "",
    "\\searrow": "",
    "\\rightarrow": "->",
    "\\leftarrow": "<-",
    "\\Rightarrow": "=>",
    "\\Leftarrow": "<=",
    "\\longrightarrow": "->",
    "\\longleftarrow": "<-",
    "\\iff": "<=>",
    "\\to": "->",
    "\\gets": "<-",
    "\\mapsto": "->",
    "\\perp": "\\perp",
    "\\parallel": "\\parallel",
    "\\angle": "\\angle",
    "\\triangle": "\\triangle",
    "\\square": "\\square",
    "\\circ": "\\circ",
    "\\bullet": "\\bullet",
    "\\star": "*",
    "\\dagger": "",
    "\\ddagger": "",
}

# ─── LaTeX commands to remove (argument kept) ────────────────────────────────

LATEX_CMDS = [
    "\\rm", "\\bf", "\\it", "\\sl", "\\tt", "\\sf",
    "\\small", "\\large", "\\Large", "\\LARGE",
    "\\huge", "\\Huge", "\\normalsize", "\\tiny", "\\footnotesize",
    "\\scriptsize",
    "\\underline", "\\overline",
    "\\hat", "\\bar", "\\tilde", "\\vec",
    "\\widetilde", "\\widehat",
    "\\overrightarrow", "\\overleftarrow",
    "\\not", "\\cancel",
    "\\boldsymbol", "\\bm",
    "\\color{red}", "\\color{blue}", "\\color{green}",
]

# ─── LaTeX environments to strip ─────────────────────────────────────────────

LATEX_FMT_ENVS = [
    "align", "align*", "aligned",
    "equation", "equation*",
    "gather", "gather*", "gathered",
    "multline", "multline*",
    "split", "cases",
    "eqnarray", "eqnarray*",
    "flalign", "flalign*",
]

LATEX_LIST_ENVS = [
    "itemize", "enumerate", "description",
    "list",
]

# ─── Basic math function names (without backslash) ───────────────────────────

BASIC_FN_NAMES = [
    "arcsin", "arccos", "arctan", "arccot", "arcsec", "arccsc",
    "sin", "cos", "tan", "cot", "sec", "csc",
    "sinh", "cosh", "tanh", "coth", "sech", "csch",
    "log", "ln", "exp",
    "max", "min", "sup", "inf",
    "lim", "limsup", "liminf",
    "det", "tr", "dim", "deg", "ker", "im",
    "gcd", "lcm",
    "Re", "Im", "Arg",
    "sgn", "sign",
]

# ─── Physical / math units ───────────────────────────────────────────────────

UNITS = [
    # time
    r"\\text\{?s\}?", r"\\text\{?sec\}?", r"\\text\{?second\}?",
    r"\\text\{?min\}?", r"\\text\{?minute\}?", r"\\text\{?hour\}?", r"\\text\{?hr\}?",
    r"\\text\{?day\}?", r"\\text\{?week\}?", r"\\text\{?year\}?",
    # length
    r"\\text\{?m\}?", r"\\text\{?cm\}?", r"\\text\{?mm\}?", r"\\text\{?km\}?",
    r"\\text\{?ft\}?", r"\\text\{?in\}?", r"\\text\{?yd\}?",
    # mass
    r"\\text\{?kg\}?", r"\\text\{?g\}?", r"\\text\{?mg\}?", r"\\text\{?lb\}?",
    # angle
    r"°", r"\\circ",
    # misc
    r"\\%",
    r"\\text\{?\%\}?",
]

# ─── String-to-number map ────────────────────────────────────────────────────

STR2NUM = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12",
    "thirteen": "13", "fourteen": "14", "fifteen": "15",
    "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19",
    "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50",
    "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",
    "hundred": "100", "thousand": "1000", "million": "1000000",
    "half": "1/2", "third": "1/3", "quarter": "1/4", "fourth": "1/4",
    "true": "True", "false": "False",
    "yes": "True", "no": "False",
}

# ─── Punctuation stripping sets ──────────────────────────────────────────────

NO_PRECEDING_PUNCS = set(",.:;!?=+-×÷*/\\|")
NO_TRAILING_STRS   = set(",.:;!?=\\.。，、；：！？")

# ─── Weekday normalization ───────────────────────────────────────────────────

_WEEKDAY_MAP = {
    "monday": "Monday", "mon": "Monday",
    "tuesday": "Tuesday", "tue": "Tuesday", "tues": "Tuesday",
    "wednesday": "Wednesday", "wed": "Wednesday",
    "thursday": "Thursday", "thu": "Thursday", "thur": "Thursday", "thurs": "Thursday",
    "friday": "Friday", "fri": "Friday",
    "saturday": "Saturday", "sat": "Saturday",
    "sunday": "Sunday", "sun": "Sunday",
}


def norm_str2weekday(s: str):
    """Return canonical weekday name or None."""
    return _WEEKDAY_MAP.get(s.strip().lower(), None)


def norm_str2bool(s: str):
    """Return 'True'/'False' string or None."""
    _map = {
        "true": "True", "yes": "True", "t": "True", "y": "True",
        "1": "True", "correct": "True", "right": "True",
        "false": "False", "no": "False", "f": "False", "n": "False",
        "0": "False", "incorrect": "False", "wrong": "False",
    }
    return _map.get(s.strip().lower(), None)


# ─── LaTeX environment removal ───────────────────────────────────────────────

def rm_latex_env(s: str, env: str) -> str:
    """Remove LaTeX \\begin{env}...\\end{env} wrappers, keeping content."""
    begin = f"\\begin{{{env}}}"
    end   = f"\\end{{{env}}}"
    while begin in s:
        s = s.replace(begin, "")
    while end in s:
        s = s.replace(end, "")
    return s


# ─── Degree normalization ────────────────────────────────────────────────────

def norm_deg(s: str) -> str:
    """Normalize degree notation: 30° -> 30, 30\\circ -> 30, 30^{\\circ} -> 30."""
    s = re.sub(r"(\d+(?:\.\d+)?)\s*°", r"\1", s)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*\^?\{?\\circ\}?", r"\1", s)
    return s


# ─── Inverse function fix ────────────────────────────────────────────────────

_INV_FN_MAP = {
    "arcsin": "\\arcsin", "arccos": "\\arccos", "arctan": "\\arctan",
    "asin": "\\arcsin", "acos": "\\arccos", "atan": "\\arctan",
    "sin^{-1}": "\\arcsin", "cos^{-1}": "\\arccos", "tan^{-1}": "\\arctan",
    "\\sin^{-1}": "\\arcsin", "\\cos^{-1}": "\\arccos", "\\tan^{-1}": "\\arctan",
    "\\sin^{(-1)}": "\\arcsin", "\\cos^{(-1)}": "\\arccos", "\\tan^{(-1)}": "\\arctan",
}


def fix_inv_func(s: str) -> str:
    """Replace inverse trig notations with arcsin/arccos/arctan."""
    for k, v in _INV_FN_MAP.items():
        s = s.replace(k, v)
    return s


# ─── Set detection ───────────────────────────────────────────────────────────

def is_set(s: str) -> bool:
    """Return True if s looks like a set literal {a, b, c}."""
    s = s.strip()
    return (s.startswith("{") and s.endswith("}") and
            not s.startswith("\\{"))


# ─── Sqrt fix ────────────────────────────────────────────────────────────────

def fix_sqrt(s: str) -> str:
    """Normalize sqrt notation: sqrt2 -> \\sqrt{2}, \\sqrt2 -> \\sqrt{2}."""
    # \sqrt followed by single char without braces
    s = re.sub(r"\\sqrt\s*([^{\\(\s])", r"\\sqrt{\1}", s)
    # bare sqrt(...) -> \sqrt{...}
    s = re.sub(r"(?<!\\)sqrt\s*\(([^)]*)\)", r"\\sqrt{\1}", s)
    # bare sqrt without parens
    s = re.sub(r"(?<!\\)sqrt\s*([^{(\s])", r"\\sqrt{\1}", s)
    return s


# ─── Fraction fix ────────────────────────────────────────────────────────────

def fix_fracs(s: str) -> str:
    """Normalize fraction shorthands to \\frac{}{} form."""
    # \frac1b -> \frac{1}{b}
    s = re.sub(r"\\frac\s*(\d)\s*(\d)", r"\\frac{\1}{\2}", s)
    # \frac{a}b -> \frac{a}{b}
    s = re.sub(r"(\\frac\{[^}]*\})\s*([^{])", r"\1{\2}", s)
    return s


def fix_a_slash_b(s: str) -> str:
    """Convert simple a/b to \\frac{a}{b} when safe."""
    # Only convert if no existing \frac and just two simple tokens
    if "\\frac" in s or "\\over" in s:
        return s
    # Pattern: plain number/number
    s = re.sub(
        r"^(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)$",
        r"\\frac{\1}{\2}",
        s.strip(),
    )
    return s


# ─── Boxed extraction (from MATH dataset evaluation) ─────────────────────────

def last_boxed_only_string(string: str):
    """Return the last \\boxed{...} substring or None."""
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_open = 0
    while i < len(string):
        if string[i] == "{":
            num_open += 1
        elif string[i] == "}":
            num_open -= 1
            if num_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def remove_boxed(s):
    """Strip \\boxed{ ... } wrapper and return inner content, or None on failure."""
    if s is None:
        return None
    for prefix in ("\\boxed{", "\\fbox{"):
        if s.startswith(prefix) and s.endswith("}"):
            return s[len(prefix) : -1]
    return None
