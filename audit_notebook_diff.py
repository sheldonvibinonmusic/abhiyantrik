import json

with open('starter_round1_original.ipynb', 'r', encoding='utf-8') as f:
    orig = json.load(f)

with open('starter_round1.ipynb', 'r', encoding='utf-8') as f:
    mod = json.load(f)

print(f"Original cells: {len(orig['cells'])}, Modified cells: {len(mod['cells'])}")

for i in range(min(len(orig['cells']), len(mod['cells']))):
    c_orig = orig['cells'][i]
    c_mod = mod['cells'][i]
    src_orig = "".join(c_orig.get('source', []))
    src_mod = "".join(c_mod.get('source', []))
    if src_orig != src_mod:
        print(f"\n--- Difference in Cell {i} [{c_orig['cell_type']}] ---")
        print("Original source snippet:")
        print(src_orig[:150])
        print("...\nModified source snippet:")
        print(src_mod[:150])
        print("...")
