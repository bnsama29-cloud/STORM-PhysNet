import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

os.makedirs('results', exist_ok=True)

fig, ax = plt.subplots(figsize=(14, 4))
ax.axis('off')
ax.set_xlim(0, 14)
ax.set_ylim(0, 4)

def draw_box(x, y, w, h, text, facecolor='#e6f2ff', edgecolor='#1f77b4'):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                  linewidth=2, edgecolor=edgecolor, facecolor=facecolor)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=11, fontweight='bold', wrap=True)

def draw_arrow(x1, y1, x2, y2, text=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=2, color='black'))
    if text:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.1, text, ha='center', va='bottom', fontsize=10)

# 1. Inputs
draw_box(0.5, 1.5, 1.8, 1.0, "Inputs:\nOMNI & GOES\n(T=72 h)", facecolor='#fff0e6', edgecolor='#ff7f0e')

# 2. Input Projection
draw_box(3.0, 1.5, 1.5, 1.0, "Input\nProjection\n(d_model=128)")
draw_arrow(2.3, 2.0, 3.0, 2.0)

# 3. Learnable Delay
draw_box(5.2, 1.5, 1.8, 1.0, "Learnable Delay\n$\\tau \in [0.5, 1.5]$ h\n(Linear Interp.)", facecolor='#e6ffe6', edgecolor='#2ca02c')
draw_arrow(4.5, 2.0, 5.2, 2.0)

# 4. Transformer Encoder
draw_box(7.7, 1.5, 1.8, 1.0, "Transformer\nEncoder\n(2 L, 4 H)", facecolor='#e6e6ff', edgecolor='#9467bd')
draw_arrow(7.0, 2.0, 7.7, 2.0)

# 5. Bz Physics Gate
draw_box(10.2, 1.5, 1.5, 1.0, "$B_z$ Physics\nGate\n(MLP + Sigmoid)", facecolor='#ffe6e6', edgecolor='#d62728')
draw_arrow(9.5, 2.0, 10.2, 2.0)

# 6. Residual Heads
draw_box(12.4, 2.4, 1.2, 0.6, "1 h Head", facecolor='#f2f2f2', edgecolor='#7f7f7f')
draw_box(12.4, 1.7, 1.2, 0.6, "6 h Head", facecolor='#f2f2f2', edgecolor='#7f7f7f')
draw_box(12.4, 1.0, 1.2, 0.6, "12 h Head", facecolor='#f2f2f2', edgecolor='#7f7f7f')

# Arrows to heads
ax.annotate("", xy=(12.4, 2.7), xytext=(11.7, 2.0), arrowprops=dict(arrowstyle="->", lw=2, color='black', connectionstyle="angle,angleA=0,angleB=180,rad=5"))
ax.annotate("", xy=(12.4, 2.0), xytext=(11.7, 2.0), arrowprops=dict(arrowstyle="->", lw=2, color='black'))
ax.annotate("", xy=(12.4, 1.3), xytext=(11.7, 2.0), arrowprops=dict(arrowstyle="->", lw=2, color='black', connectionstyle="angle,angleA=0,angleB=180,rad=5"))

# Save
plt.tight_layout()
plt.savefig('results/fig_system_architecture.png', dpi=400, bbox_inches='tight')
print("Architecture diagram saved to results/fig_system_architecture.png")
