with open('ieee_paper/claude/STORM_PhysNet_Access_claude.tex', 'r', encoding='utf-8') as f:
    access_lines = f.readlines()

with open('ieee_paper/claude/storm_physnet_conference_claude.tex', 'r', encoding='utf-8') as f:
    conf_lines = f.readlines()

def fix_math_blank_lines(lines, filename):
    # Environments that should not have blank lines inside
    no_blank_envs = {
        'equation', 'align', 'eqnarray', 'multline',
        'figure', 'figure*', 'table', 'table*',
        'tabular', 'tabular*',
        'abstract', 'keywords', 'IEEEkeywords'
    }
    
    cleaned = []
    env_stack = []
    removed = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Track environment entry/exit
        if stripped.startswith('\\begin{'):
            env_name = stripped.replace('\\begin{', '').replace('}', '')
            env_stack.append(env_name)
        
        if stripped.startswith('\\end{'):
            env_name = stripped.replace('\\end{', '').replace('}', '')
            if env_stack and env_stack[-1] == env_name:
                env_stack.pop()
        
        # Skip blank lines inside sensitive environments
        if stripped == '' and env_stack and env_stack[-1] in no_blank_envs:
            removed += 1
            continue
        
        cleaned.append(line)
    
    print(f'{filename}: removed {removed} blank lines inside math/environment blocks')
    return cleaned

access_cleaned = fix_math_blank_lines(access_lines, 'Access')
conf_cleaned = fix_math_blank_lines(conf_lines, 'Conference')

with open('ieee_paper/claude/STORM_PhysNet_Access_claude.tex', 'w', encoding='utf-8') as f:
    f.writelines(access_cleaned)

with open('ieee_paper/claude/storm_physnet_conference_claude.tex', 'w', encoding='utf-8') as f:
    f.writelines(conf_cleaned)

print(f'\nAccess: {len(access_lines)} -> {len(access_cleaned)} lines')
print(f'Conference: {len(conf_lines)} -> {len(conf_cleaned)} lines')
