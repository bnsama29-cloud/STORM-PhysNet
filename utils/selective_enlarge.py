import re

files = [
    r"f:\Downloads\ieee_final_fixed\ieee_paper\claude\STORM_PhysNet_Access_claude.tex",
    r"f:\Downloads\ieee_final_fixed\ieee_paper\claude\storm_physnet_conference_claude.tex"
]

def upgrade_to_fig_star(content, filename, placement, width):
    # Find the figure block for this filename
    pattern = r"\\begin\{figure\}\[!htbp\]\s*\\centering\s*\\includegraphics\[width=[^\]]+\]\{figures/" + filename + r"\}[\s\S]*?\\end\{figure\}"
    
    match = re.search(pattern, content)
    if match:
        block = match.group(0)
        # Change figure to figure*
        block = block.replace(r"\begin{figure}[!htbp]", f"\\begin{{figure*}}[{placement}]")
        block = block.replace(r"\end{figure}", r"\end{figure*}")
        # Change width
        block = re.sub(r"width=[^\]]+", f"width={width}", block)
        content = content.replace(match.group(0), block)
    return content

for tex_file in files:
    with open(tex_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # The side-by-side plots that MUST be large to be readable:
    content = upgrade_to_fig_star(content, "fig_ensemble_alpha_sweep.png", "!t", "0.85\\textwidth")
    content = upgrade_to_fig_star(content, "noise_robustness.png", "!b", "0.85\\textwidth")
    content = upgrade_to_fig_star(content, "fig_residual_storm_bz.png", "!t", "0.85\\textwidth")
    content = upgrade_to_fig_star(content, "fig_residual_transformer.png", "!b", "0.85\\textwidth")
    content = upgrade_to_fig_star(content, "physics_scatters.png", "!t", "0.85\\textwidth")
    
    with open(tex_file, 'w', encoding='utf-8') as f:
        f.write(content)

print("Selectively enlarged side-by-side figures!")
