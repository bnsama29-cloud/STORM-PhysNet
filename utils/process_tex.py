import re

tex_path = r"F:\Downloads\ieee_final_fixed\ieee_paper\claude\STORM_PhysNet_Access_claude.tex"

with open(tex_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the line index of \section{Discussion}
disc_idx = None
for i, line in enumerate(lines):
    if line.strip().startswith('\\section{Discussion}'):
        disc_idx = i
        break

if disc_idx is None:
    print("Could not find Discussion section")
    sys.exit(1)

# Define the wider-delay paragraph marker
wider_marker = "To test whether the original upper bound of $1.5$\\,h was limiting performance"
# Define the bagged-Transformer sentence marker
bagged_marker = "A bagged Transformer control (sixteen seeds) reached mean PE$_{45\\mathrm{min}}\\approx0.978$ and PE$_{6\\mathrm{h}}\\approx0.895$."

# We'll keep track of whether we have kept the wider-delay paragraph in Discussion
kept_wider = False
kept_bagged = False

new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    # Check if we are entering Discussion section
    if line.strip().startswith('\\section{Discussion}'):
        # reset flags? Actually we want to keep only after this point.
        pass
    # Check for wider-delay paragraph start
    if wider_marker in line:
        # Determine if this occurrence is after disc_idx
        if i > disc_idx and not kept_wider:
            # Keep this paragraph (including the following lines until the paragraph ends)
            # We need to know where the paragraph ends. Assume it ends before a blank line or next \section?
            # For simplicity, we'll keep this line and continue until we see a line that is empty or starts with \section or \subsection?
            # But we can just keep the line and let the rest be added normally; we will skip duplicates later.
            kept_wider = True
            new_lines.append(line)
            i += 1
            # Continue adding lines until we see a line that is just empty (or maybe end of paragraph)
            while i < len(lines) and lines[i].strip() != '' and not lines[i].strip().startswith('\\') and not lines[i].strip().startswith('%'):
                new_lines.append(lines[i])
                i += 1
            # Add the empty line if present
            if i < len(lines) and lines[i].strip() == '':
                new_lines.append(lines[i])
                i += 1
            continue
        else:
            # Skip this paragraph: skip lines until we see an empty line or next section/subsection
            i += 1
            while i < len(lines) and lines[i].strip() != '' and not lines[i].strip().startswith('\\') and not lines[i].strip().startswith('%'):
                i += 1
            if i < len(lines) and lines[i].strip() == '':
                i += 1  # skip the empty line as well
            continue
    # Check for bagged-Transformer sentence
    if bagged_marker in line:
        if i > disc_idx and not kept_bagged:
            # Keep this sentence
            kept_bagged = True
            new_lines.append(line)
            i += 1
            continue
        else:
            # Skip this sentence
            i += 1
            continue
    # Otherwise keep line
    new_lines.append(line)
    i += 1

with open(tex_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Processed tex file.")