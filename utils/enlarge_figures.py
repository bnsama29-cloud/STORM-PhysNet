import re

files = [
    r"f:\Downloads\ieee_final_fixed\ieee_paper\claude\STORM_PhysNet_Access_claude.tex",
    r"f:\Downloads\ieee_final_fixed\ieee_paper\claude\storm_physnet_conference_claude.tex"
]

for tex_file in files:
    with open(tex_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Change 0.75\columnwidth back to \columnwidth to make them bigger
    content = re.sub(r"width=0\.75\\columnwidth", r"width=\\columnwidth", content)
    
    with open(tex_file, 'w', encoding='utf-8') as f:
        f.write(content)

print("Enlarged figures to full column width!")
