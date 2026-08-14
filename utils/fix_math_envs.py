with open('ieee_paper/claude/STORM_PhysNet_Access_claude.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Remove blank lines inside math environments and other LaTeX environments
# Keep track of environment state
cleaned = []
in_math_env = False
math_env_depth = 0

# Environments that should not have blank lines inside
no_blank_envs = {
    'equation', 'align', 'eqnarray', 'multline',
    'figure', 'figure*', 'table', 'table*',
    'tabular', 'tabular*',
    'abstract', 'keywords', 'IEEEkeywords'
}

i = 0
while i < len(lines):
    line = lines[i]
    stripped = line.strip()
    
    # Check for environment starts
    if stripped.startswith('\\begin{'):
        env_name = stripped.replace('\\begin{', '').replace('}', '')
        if env_name in no_blank_envs:
            math_env_depth += 1
    
    # Check for environment ends
    if stripped.startswith('\\end{'):
        env_name = stripped.replace('\\end{', '').replace('}', '')
        if env_name in no_blank_envs:
            math_env_depth -= 1
    
    # If this is a blank line and we're inside a no-blank environment, skip it
    if stripped == '' and math_env_depth > 0:
        i += 1
        continue
    
    cleaned.append(line)
    i += 1

with open('ieee_paper/claude/STORM_PhysNet_Access_claude.tex', 'w', encoding='utf-8') as f:
    f.writelines(cleaned)

print(f'Access: removed blank lines inside math/environment blocks')
print(f'Lines: {len(lines)} -> {len(cleaned)}')
