import os
import re

files = [
    r'f:\Downloads\ieee_final_fixed\ieee_paper\claude\storm_physnet_conference_claude.tex',
    r'f:\Downloads\ieee_final_fixed\ieee_paper\claude\STORM_PhysNet_Access_claude.tex'
]

replacements = {
    r'45-minute': r'1-hour',
    r'45 min': r'1 h',
    r'45\,min': r'1\,h',
    r'45\\,min': r'1\\,h',
    r'0.75h': r'1.0h',
    r'0.75 h': r'1.0 h',
    r'0.75\,h': r'1.0\,h',
    r'0.75\\,h': r'1.0\\,h',
    r'45\\mathrm{min}': r'1\\mathrm{h}',
    r'PE_{45min}': r'PE_{1h}',
    r'PE_{45\mathrm{min}}': r'PE_{1\mathrm{h}}'
}

for path in files:
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for k, v in replacements.items():
        content = content.replace(k, v)
        
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
print("LaTeX papers successfully aligned to 1-hour horizon.")
