import re

tex_file = r"f:\Downloads\ieee_final_fixed\ieee_paper\claude\STORM_PhysNet_Access_claude.tex"
with open(tex_file, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Convert most figure* to figure to stop the 2-column float lockup
# We will keep fig_system_architecture as figure* because it's massive.
# But for the others, we change figure* to figure and width to \columnwidth

def replace_fig_star(match):
    env = match.group(1)
    # Don't touch system_architecture
    if "fig_system_architecture" in match.group(0):
        return match.group(0)
    
    # Change figure* to figure
    block = match.group(0).replace(r"\begin{figure*}", r"\begin{figure}").replace(r"\end{figure*}", r"\end{figure}")
    
    # Change width=... to width=\columnwidth
    block = re.sub(r"width=[0-9\.]*\\textwidth", r"width=\\columnwidth", block)
    return block

content = re.sub(r"(\\begin\{figure\*\}.*?\\end\{figure\*\})", replace_fig_star, content, flags=re.DOTALL)

# 2. Scale down existing single-column figures slightly so they fit multiple per page easily
content = re.sub(r"width=\\columnwidth", r"width=0.85\\columnwidth", content)
content = re.sub(r"width=0.9\\columnwidth", r"width=0.75\\columnwidth", content)
content = re.sub(r"width=0.85\\columnwidth", r"width=0.75\\columnwidth", content)

with open(tex_file, 'w', encoding='utf-8') as f:
    f.write(content)

print("Scaled down figures and converted figure* to figure to fix float queue!")
