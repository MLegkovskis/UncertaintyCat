forbidden_patterns = [
    # Logical/conditional words in the equations:
    " otherwise",
    " or ",
    # Code-like inequalities:
    ">=",
    "<=",

    # Textual logical connectors in LaTeX:
    "\\text{and}",

    # Direct Python references:
    "import ",
    "def ",
    "np.",
    "fsolve",
    "python code",
    "the code above",

    # Complex LaTeX environments (align or others):
    "\\begin{align",
    "\\end{align",
    "\\begin{align*}",
    "\\end{align*}",

    # Raw HTML tags:
    "<div>",
    "</div>",
    "<table>",
    "</table>",
    "<html>",
    "</html>",
    "<body>",
    "</body>",

    # Markdown code blocks:
    "```",
    "`"
]