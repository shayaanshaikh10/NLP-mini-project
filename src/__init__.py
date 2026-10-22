"""HinglishSense source package."""

import sys

# Windows terminals often use an old encoding (cp1252) that cannot print
# emojis or Hindi text, which makes print() crash. Switch to UTF-8.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")
