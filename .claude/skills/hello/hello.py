#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime

# Print to stdout
print("hello world")

# Write to hello.txt with timestamp in project root
# The script is in .claude/skills/hello/, so project root is ../../
project_root = Path(__file__).parent.parent.parent
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = project_root / f"hello_{timestamp}.txt"

with open(output_file, 'w') as f:
    f.write("hello world\n")

print(f"Created {output_file}")
