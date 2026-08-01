# Errata

Confirmed corrections against the printed text of *The 7 GenAI Architectures*.

Nothing is listed here yet. This file exists so the promise the book makes has a
destination: "Using the Code Examples" and "About the Author" both send readers
here for corrections, and a promise pointing at nothing is worse than no promise.

## How to report

Open an issue: <https://github.com/ranjankumar-gh/escalation-ladder/issues>

Two kinds of report are especially useful.

**A number that no longer reproduces.** Every figure in the book regenerates from
a command, so "I ran `python -m scripts.X` and got Y instead of the printed Z" is
a complete bug report. Include the command and both numbers.

**A version pin that has broken.** The stack is pinned at a date stated in "Using
the Code Examples," and a scheduled CI job checks it against both the pinned
versions and the current ones - but that job can only test what it already knows
to look for. An SDK behavior change the book asserts as fact is exactly what it
will miss.

## Format

Each entry records where the error is, what is wrong, what is correct, and the
date it was confirmed:

```
### Chapter N, page/section - short description
**Printed:** what the book says
**Correct:** what it should say
**Confirmed:** YYYY-MM-DD
```
