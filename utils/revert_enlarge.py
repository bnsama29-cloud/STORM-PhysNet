import re

files = [
    r"f:\Downloads\ieee_final_fixed\ieee_paper\claude\STORM_PhysNet_Access_claude.tex",
    r"f:\Downloads\ieee_final_fixed\ieee_paper\claude\storm_physnet_conference_claude.tex"
]

def revert_fig_star(content, filename):
    # Find the figure* block for this filename
    pattern = r"\\begin\{figure\*\}\[[^\]]+\]\s*\\centering\s*\\includegraphics\[width=[^\]]+\]\{figures/" + filename + r"\}[\s\S]*?\\end\{figure\*\}"
    
    match = re.search(pattern, content)
    if match:
        block = match.group(0)
        # Change figure* to figure
        block = re.sub(r"\\begin\{figure\*\}.*?\]", r"\\begin{figure}[!htbp]", block)
        block = block.replace(r"\end{figure*}", r"\end{figure}")
        # Change width back to \columnwidth
        block = re.sub(r"width=[^\]]+", r"width=\\columnwidth", block)
        content = content.replace(match.group(0), block)
    return content

for tex_file in files:
    with open(tex_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Revert the 5 side-by-side plots back to single column
    content = revert_fig_star(content, "fig_ensemble_alpha_sweep.png")
    content = revert_fig_star(content, "noise_robustness.png")
    content = revert_fig_star(content, "fig_residual_storm_bz.png")
    content = revert_fig_star(content, "fig_residual_transformer.png")
    content = revert_fig_star(content, "physics_scatters.png")
    
    with open(tex_file, 'w', encoding='utf-8') as f:
        f.write(content)

print("Reverted to working single-column layout!")
