with open('ieee_paper/claude/STORM_PhysNet_Access_claude.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

cleaned = []
for i in range(len(lines)):
    line = lines[i]
    stripped = line.strip()
    
    if stripped == '':
        # Check if this blank line is between two text lines
        prev_line = cleaned[-1].strip() if cleaned else ''
        next_line = lines[i+1].strip() if i+1 < len(lines) else ''
        
        # If both prev and next are text (not commands), remove this blank line
        if (prev_line and not prev_line.startswith('\\') and not prev_line.startswith('%') and 
            not prev_line.startswith('{') and not prev_line.startswith('}') and
            not prev_line.startswith('$') and not prev_line.startswith('\\[') and 
            not prev_line.startswith('\\begin') and not prev_line.startswith('\\end') and
            not prev_line.startswith('\\item') and not prev_line.startswith('#') and
            next_line and not next_line.startswith('\\') and not next_line.startswith('%') and
            not next_line.startswith('{') and not next_line.startswith('}') and
            not next_line.startswith('$') and not next_line.startswith('\\[') and
            not next_line.startswith('\\begin') and not next_line.startswith('\\end') and
            not next_line.startswith('\\item') and not next_line.startswith('#')):
            # Skip this blank line - it's between sentences
            continue
        else:
            cleaned.append(line)
    else:
        cleaned.append(line)

with open('ieee_paper/claude/STORM_PhysNet_Access_claude.tex', 'w', encoding='utf-8') as f:
    f.writelines(cleaned)

print(f'Access: {len(lines)} -> {len(cleaned)} lines')
