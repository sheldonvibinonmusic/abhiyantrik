import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('starter_round1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells']):
    ct = cell['cell_type']
    first_line = "".join(cell.get('source', []))[:70].replace('\n', ' ')
    print(f"Cell {i} [{ct}]: {first_line}")
