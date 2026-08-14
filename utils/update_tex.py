import os

files = [
    r"f:\Downloads\ieee_final_fixed\ieee_paper\claude\storm_physnet_conference_claude.tex",
    r"f:\Downloads\ieee_final_fixed\ieee_paper\claude\STORM_PhysNet_Access_claude.tex"
]

replacements = {
    "45-minute": "1-hour",
    "45-min": "1-hour",
    "45 min": "1 h",
    "45\,min": "1\,h",
    "0.75h": "1.0h",
    "0.75 h": "1.0 h",
    "0.75\,h": "1.0\,h",
    "45\\mathrm{min}": "1\\mathrm{h}"
}

for path in files:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    for k, v in replacements.items():
        content = content.replace(k, v)
        
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
