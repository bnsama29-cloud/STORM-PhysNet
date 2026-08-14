import os

def revert_file(filepath, replacements):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    for old, new in replacements:
        content = content.replace(old, new)
        
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Reverted {filepath}")
    else:
        print(f"No changes needed for {filepath}")

latex_disclaimer_1 = r" \textit{(Note: Axis and diagram labels retaining the legacy `45-min' or `0.75' terminology correspond to the 1-hour forecast horizon discussed in the text.)}}"
latex_disclaimer_2 = r" \textit{(Note: Axis and diagram labels retaining the legacy '45-min' or '0.75' terminology correspond to the 1-hour forecast horizon discussed in the text.)}}"

latex_replacements = [
    (latex_disclaimer_1, "}"),
    (latex_disclaimer_2, "}"),
    (r"\hat{y}_{t+1\mathrm{h}}", r"\hat{y}_{t+0.75\mathrm{h}}"),
    ("1-hour", "45-min"),
    ("1\,h", "45\,min"),
    ("1.0 h", "0.75 h"),
    ("1.0\,h", "0.75\,h"),
    ("1 h ", "45 min "),
    ("1 h;", "45 min;"),
    ("1 h.", "45 min.")
]

revert_file('ieee_paper/claude/STORM_PhysNet_Access_claude.tex', latex_replacements)
revert_file('ieee_paper/claude/storm_physnet_conference_claude.tex', latex_replacements)

py_replacements = [
    ('["1 h", "6 h", "12 h"]', '["45 min", "6 h", "12 h"]'),
    ('PE (1 h)', 'PE (45 min)'),
]

revert_file('11_paper_extras.py', py_replacements)
revert_file('compute_alpha3.py', py_replacements)
revert_file('fix_figures_local.py', py_replacements)

print("All reverts applied.")
