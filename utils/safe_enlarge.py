import re

access_tex = r"f:\Downloads\ieee_final_fixed\ieee_paper\claude\STORM_PhysNet_Access_claude.tex"

def upgrade_to_fig_star(content, filename, width):
    pattern = r"\\begin\{figure\}\[!htbp\]\s*\\centering\s*\\includegraphics\[width=[^\]]+\]\{figures/" + filename + r"\}[\s\S]*?\\end\{figure\}"
    match = re.search(pattern, content)
    if match:
        block = match.group(0)
        block = block.replace(r"\begin{figure}[!htbp]", r"\begin{figure*}[!t]")
        block = block.replace(r"\end{figure}", r"\end{figure*}")
        # Use lambda to prevent regex escape processing on \textwidth
        block = re.sub(r"width=[^\]]+", lambda m: f"width={width}", block)
        content = content.replace(match.group(0), block)
    return content

# Read Access paper
with open(access_tex, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Upgrade Figures 3 and 6 to figure* width=0.75\textwidth
content = upgrade_to_fig_star(content, "fig_ensemble_alpha_sweep.png", r"0.75\textwidth")
content = upgrade_to_fig_star(content, "noise_robustness.png", r"0.75\textwidth")

# 2. Combine Figures 10 and 11 into a minipage figure*
residual_pattern = r"\\begin\{figure\}\[!htbp\]\s*\\centering\s*\\includegraphics\[width=\\columnwidth\]\{figures/fig_residual_storm_bz\.png\}[\s\S]*?\\end\{figure\}\s*\\begin\{figure\}\[!htbp\]\s*\\centering\s*\\includegraphics\[width=\\columnwidth\]\{figures/fig_residual_transformer\.png\}[\s\S]*?\\end\{figure\}"

minipage_replacement = r"""\begin{figure*}[!t]
\centering
\begin{minipage}{0.48\textwidth}
\centering
\includegraphics[width=\linewidth]{figures/fig_residual_storm_bz.png}
\caption{Residual distribution (predicted minus true log-flux) for STORM-Bz
on the GOES test set, stratified by storm and quiet windows.}
\label{fig:resid_storm}
\end{minipage}\hfill
\begin{minipage}{0.48\textwidth}
\centering
\includegraphics[width=\linewidth]{figures/fig_residual_transformer.png}
\caption{Same residual view for the Transformer baseline.}
\label{fig:resid_tf}
\end{minipage}
\end{figure*}"""

match = re.search(residual_pattern, content)
if match:
    content = content.replace(match.group(0), minipage_replacement)

# Write back
with open(access_tex, 'w', encoding='utf-8') as f:
    f.write(content)

print("Upgraded 3, 6, 10, 11 to large figures in Access paper!")
