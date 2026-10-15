import pandas as pd
import re

train = pd.read_csv('train.csv')

def parse_msg(m):
    # Match: NINE: (optional prefix). (payload) [(module) (reading) (unit)]
    bracket = re.search(r'\[(.*?)\]', m)
    bracket_content = bracket.group(1) if bracket else ""
    text_before = m[:bracket.start()].strip() if bracket else m.strip()
    
    # remove "NINE:"
    if text_before.startswith('NINE:'):
        text_before = text_before[5:].strip()
        
    parts = text_before.split('. ', 1)
    if len(parts) == 2:
        prefix, payload = parts[0].strip(), parts[1].strip()
    else:
        prefix, payload = "", parts[0].strip()
        
    return prefix, payload, bracket_content

train['prefix'], train['payload'], train['bracket'] = zip(*train['message'].apply(parse_msg))

print("Unique prefixes:", train['prefix'].value_counts())
print("\nTop 25 unique payloads:", train['payload'].value_counts().head(25))
print("\nUnique brackets check:")
print(f"Total rows with bracket: {(train['bracket'] != '').sum()} / {len(train)}")
