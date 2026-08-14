import re

tex_file = r"f:\Downloads\ieee_final_fixed\ieee_paper\claude\storm_physnet_conference_claude.tex"
with open(tex_file, 'r', encoding='utf-8') as f:
    content = f.read()

def replace_fig_star(match):
    env = match.group(1)
    if "fig_system_architecture" in match.group(0):
        return match.group(0)
    
    block = match.group(0).replace(r"\begin{figure*}", r"\begin{figure}").replace(r"\end{figure*}", r"\end{figure}")
    block = re.sub(r"width=[0-9\.]*\\textwidth", r"width=\\columnwidth", block)
    return block

content = re.sub(r"(\\begin\{figure\*\}.*?\\end\{figure\*\})", replace_fig_star, content, flags=re.DOTALL)

content = re.sub(r"width=\\columnwidth", r"width=0.85\\columnwidth", content)
content = re.sub(r"width=0.9\\columnwidth", r"width=0.75\\columnwidth", content)
content = re.sub(r"width=0.85\\columnwidth", r"width=0.75\\columnwidth", content)

with open(tex_file, 'w', encoding='utf-8') as f:
    f.write(content)

print("Scaled down figures in conference tex too!")
