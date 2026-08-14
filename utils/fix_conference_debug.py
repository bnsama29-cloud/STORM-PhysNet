with open('ieee_paper/claude/storm_physnet_conference_claude.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

no_blank_envs = {
    'equation', 'align', 'eqnarray', 'multline',
    'figure', 'figure*', 'table', 'table*',
    'tabular', 'tabular*',
    'abstract', 'keywords', 'IEEEkeywords'
}

cleaned = []
math_env_depth = 0
removed = 0

for i, line in enumerate(lines):
    stripped = line.strip()
    
    # Check for environment starts
    if stripped.startswith('\\begin{'):
        env_name = stripped.replace('\\begin{', '').replace('}', '')
        if env_name in no_blank_envs:
            math_env_depth += 1
            print(f'Line {i+1}: ENTER {env_name}, depth={math_env_depth}')
    
    # Check for environment ends
    if stripped.startswith('\\end{'):
        env_name = stripped.replace('\\end{', '').replace('}', '')
        if env_name in no_blank_envs:
            math_env_depth -= 1
            print(f'Line {i+1}: EXIT {env_name}, depth={math_env_depth}')
    
    # If this is a blank line and we're inside a no-blank environment, skip it
    if stripped == '' and math_env_depth > 0:
        removed += 1
        continue
    
    cleaned.append(line)

with open('ieee_paper/claude/storm_physnet_conference_claude.tex', 'w', encoding='utf-8') as f:
    f.writelines(cleaned)

print(f'\nRemoved {removed} blank lines inside math/environment blocks')
print(f'Lines: {len(lines)} -> {len(cleaned)}')
