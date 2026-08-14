import re, os, time, subprocess

print('='*60)
print('COMPREHENSIVE CONSISTENCY CHECK')
print('='*60)

# 1. Check both papers for all consistency issues
for name, path in [('Conference', r'F:\Downloads\ieee_final_fixed\ieee_paper\claude\storm_physnet_conference_claude.tex'),
                   ('Access', r'F:\Downloads\ieee_final_fixed\ieee_paper\claude\STORM_PhysNet_Access_claude.tex')]:
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print('\n=== ' + name + ' PAPER ===')
    
    # Horizon terminology
    cnt_45min = len(re.findall(r'45.*min', content))
    cnt_1h = len(re.findall(r'PE.*1h|1-hour|PE_1h', content))
    cnt_075h = len(re.findall(r'0\.75.*h|0\.75h', content))
    cnt_6h = len(re.findall(r'6\\\\h|6h', content))
    cnt_12h = len(re.findall(r'12\\\\h|12h', content))
    print('  45min refs: ' + str(cnt_45min))
    print('  1h refs: ' + str(cnt_1h))
    print('  0.75h refs: ' + str(cnt_075h))
    print('  6h refs: ' + str(cnt_6h))
    print('  12h refs: ' + str(cnt_12h))
    
    # Alpha consistency
    alpha = len(re.findall(r'alpha.*0\.3|alpha.*=.*0\.3', content, re.IGNORECASE))
    print('  alpha=0.3: ' + str(alpha))
    
    # PE definitions
    pe_clim = len(re.findall(r'PE.*clim', content))
    pe_pers = len(re.findall(r'PE.*pers', content))
    print('  PE_clim: ' + str(pe_clim) + ', PE_pers: ' + str(pe_pers))
    
    # Table headers
    idx = content.find('System & PE')
    if idx >= 0:
        print('  Table header: ' + content[idx:idx+100])
    
    # Abstract PE numbers
    abs_start = content.find('\\begin{abstract}')
    abs_end = content.find('\\end{abstract}')
    if abs_start >= 0 and abs_end >= 0:
        abstract = content[abs_start:abs_end]
        pe_nums = re.findall(r'PE.*=.*0\.\d+', abstract)
        print('  Abstract PE numbers: ' + str(pe_nums))
    
    # GRASP table
    grasp_idx = content.find('GRASP')
    if grasp_idx >= 0:
        tab_idx = content.find('\\begin{tabular}', grasp_idx)
        if tab_idx >= 0:
            table_snippet = content[tab_idx:tab_idx+300]
            for l in table_snippet.split('\n'):
                if 'Horizon' in l or '45' in l or '1h' in l or '1h' in l:
                    print('  GRASP table: ' + l.strip())

# 2. Check Git repo
print('\n=== GIT REPO ===')
with open(r'F:\Downloads\STORM-PhysNet-check\configs\config.yaml', 'r') as f:
    cfg = f.read()
for line in cfg.split('\n'):
    if 'forecast_horizons' in line or '0.75' in line or '6.0' in line or '12.0' in line or 'storm_weight' in line:
        print('  ' + line.strip())

with open(r'F:\Downloads\STORM-PhysNet-check\src\data\dataloader.py', 'r') as f:
    dl = f.read()
for line in dl.split('\n'):
    if 'HORIZONS' in line:
        print('  dataloader: ' + line.strip())

# 3. Check figures
print('\n=== FIGURES ===')
fig_dir = r'F:\Downloads\ieee_final_fixed\ieee_paper\claude\figures'
figs = os.listdir(fig_dir)
print('Total figures: ' + str(len(figs)))
today = time.time()
for f in sorted(figs):
    path = os.path.join(fig_dir, f)
    stat = os.stat(path)
    updated = ' ***TODAY***' if stat.st_mtime > today - 86400 else ''
    print('  ' + f + ' (' + str(stat.st_size) + ' bytes)' + updated)

# Check figure references in papers
for name, path in [('Conference', r'F:\Downloads\ieee_final_fixed\ieee_paper\claude\storm_physnet_conference_claude.tex'),
                   ('Access', r'F:\Downloads\ieee_final_fixed\ieee_paper\claude\STORM_PhysNet_Access_claude.tex')]:
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    figs_ref = re.findall(r'figures/fig_([^.]+)\.png', content)
    print('\n' + name + ' paper references ' + str(len(figs_ref)) + ' figures')
    for f in figs_ref:
        print('  - ' + f)
    # Check which figures are NOT referenced
    all_figs_base = [f.replace('fig_', '').replace('.png', '') for f in os.listdir(fig_dir) if f.startswith('fig_')]
    for f in all_figs_base:
        if f not in content:
            print('  WARNING: Figure ' + f + ' NOT referenced in ' + name)

# Check git
print('\n=== GIT STATUS ===')
result = subprocess.run(['git', '-C', r'F:\Downloads\STORM-PhysNet-check', 'log', '--oneline', '-3'], capture_output=True, text=True)
print(result.stdout)
result = subprocess.run(['git', '-C', r'F:\Downloads\STORM-PhysNet-check', 'status'], capture_output=True, text=True)
print(result.stdout)