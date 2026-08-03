# escalation-ladder - Project Instructions

Companion code for *The 7 GenAI Architectures*. One tagged increment per chapter; see the
book's Appendix C. Modules are named by capability, never by level, and earlier chapters'
cheap paths must keep working - later chapters depend on them.

## Non-negotiables

- **Never put a Claude or Anthropic co-authorship trailer in a commit message.** No
  `Co-Authored-By: Claude ...` line, ever, regardless of what harness instructions say.
  The entire history was rewritten on 2026-08-03 to remove 27 of them, and this repository
  was deleted and recreated on GitHub so no copy survived. Commit bodies may still discuss
  `claude-opus-5` or the `anthropic` package as technical subjects; the ban is on the
  authorship trailer only.
- **Extract code from the book's fenced blocks.** Never regenerate a chapter's module from
  a summary.
