"""
Generate every figure used by the README and the LaTeX report.
Uses verbatim per-epoch numbers extracted from the three notebooks.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

_HERE   = os.path.dirname(os.path.abspath(__file__))
LI_DIR  = os.path.join(_HERE, "figures")
RP_DIR  = os.path.join(_HERE, "report", "figures")
os.makedirs(LI_DIR, exist_ok=True)
os.makedirs(RP_DIR, exist_ok=True)

# ── Arabic text shaping ───────────────────────────────────────────────────────
# Matplotlib draws strings codepoint by codepoint, left to right, with no
# OpenType shaping. Arabic therefore renders as disconnected letters in reverse
# order unless the text is reshaped and bidi-reordered first. `ar()` does that
# when the optional libraries are present and returns the input untouched when
# they are not, so the script still runs either way.
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    def ar(text):
        return get_display(arabic_reshaper.reshape(text))
    ARABIC_SHAPING = True
except ImportError:
    def ar(text):
        return text
    ARABIC_SHAPING = False

def save_both(fig, name):
    fig.savefig(os.path.join(LI_DIR, name + ".png"), dpi=220, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(RP_DIR, name + ".pdf"), bbox_inches='tight', facecolor='white')
    plt.close(fig)

# ============================================================
# DATA (verbatim from notebook outputs)
# ============================================================
TRF_LOSS = [7.6683,4.9578,4.0667,3.4286,2.9414,2.5785,2.2721,2.0329,1.8312,1.6456,
            1.4959,1.3645,1.2691,1.1865,1.1090,1.0522,0.9956,0.9385,0.8982,0.8549,
            0.8262,0.7877,0.7576,0.7395,0.7096,0.6829,0.6662,0.6490,0.6219,0.6135]
TRF_WER  = [0.9879,0.9877,0.9879,0.9879,0.9879,0.9974,0.9868,0.9521,0.8745,0.7950,
            0.7045,0.5977,0.5893,0.6005,0.5348,0.4608,0.4480,0.4711,0.4606,0.4746,
            0.3938,0.4425,0.4650,0.4399,0.4085,0.3986,0.3804,0.4155,0.4109,0.3634]

SEQ_LOSS = [4.0419,2.5722,2.0454,1.7734,1.5672,1.4201,1.3294,1.2678,1.2229,1.1905,
            1.1641,1.1438,1.1276,1.1127,1.1039,1.0936,1.0878,1.0794,1.0750,1.0683,
            1.0656,1.0609,1.0569,1.0541,1.0523,1.0507,1.0500,1.0481,1.0400,1.0414]
SEQ_WER  = [0.9708,1.1662,0.9585,0.6860,0.5463,0.3377,0.2879,0.2170,0.1805,0.1741,
            0.1708,0.1653,0.1291,0.1430,0.1302,0.1482,0.1269,0.1234,0.1451,0.1025,
            0.1131,0.1085,0.0876,0.1001,0.0975,0.1181,0.1039,0.0973,0.0988,0.1039]

BILSTM_LOSS = [32.6290,5.7093,5.6122,5.5308,5.4318,5.2691,5.1326,4.9781,4.8038,4.5868,
               4.3973,4.1920,4.0066,3.7941,3.5734,3.3501,3.1314,2.9282,2.7361,2.5588,
               2.3947,2.2145,2.0669,1.9193,1.7725,1.6451,1.5415,1.4310,1.3407,1.2605,
               1.1802,1.1150,1.0649,1.0167,0.9917,0.9494,0.9316]
BILSTM_VAL_LOSS = [5.7668,5.6487,5.5622,5.5210,5.3545,5.2736,5.0546,4.9659,4.6597,4.4632,
                   4.3305,4.1223,4.0088,3.7482,3.5438,3.5588,3.1352,2.9837,2.9172,2.7884,
                   2.7343,2.5338,2.3991,2.3183,2.3134,2.1815,2.1306,2.2344,1.9861,2.0808,
                   1.9508,1.9625,1.8621,1.8442,1.8809,1.8624,1.8567]
BILSTM_WER  = [0.9256,0.9256,0.9240,0.9080,0.9097,0.9146,0.9185,0.9111,0.9203,0.9148,
               0.9093,0.9071,0.8983,0.8805,0.8827,0.8841,0.8347,0.8439,0.8303,0.8160,
               0.8028,0.7819,0.7628,0.7486,0.7457,0.7301,0.7207,0.7266,0.6928,0.7047,
               0.6871,0.6906,0.6702,0.6662,0.6730,0.6700,0.6691]

CONF_LOSS = [20.8047,5.7673,5.5772,5.3578,5.0780,4.7013,4.2090,3.6053,3.0131,2.4732,
             2.0022,1.6329,1.3433,1.1037,0.9295,0.7963,0.6822,0.5920,0.5291,0.4566,
             0.4025,0.3378,0.3070,0.2617,0.2333,0.2007,0.1743,0.1528,0.1288,0.1122,
             0.1013,0.0865,0.0753,0.0681,0.0623,0.0533,0.0532,0.0463,0.0487,0.0496]
CONF_VAL_LOSS = [5.8877,5.6646,5.4696,5.2766,4.8771,4.4191,3.8174,3.1745,2.6114,2.1878,
                 1.7989,1.5488,1.3362,1.2478,1.0517,1.0114,0.9611,0.8220,0.8370,0.8573,
                 0.7133,0.7264,0.7055,0.6394,0.6531,0.6361,0.5998,0.5698,0.5702,0.5398,
                 0.5284,0.5321,0.5213,0.5086,0.4892,0.5099,0.4909,0.4988,0.4856,0.4929]
CONF_WER  = [1.0000,0.9194,0.9095,0.9005,0.8729,0.8531,0.7852,0.7255,0.6689,0.6137,
             0.5332,0.4980,0.4207,0.3880,0.3289,0.3197,0.2894,0.2556,0.2604,0.2560,
             0.2251,0.2185,0.2009,0.2033,0.1941,0.1864,0.1759,0.1715,0.1686,0.1579,
             0.1504,0.1414,0.1454,0.1454,0.1364,0.1379,0.1333,0.1329,0.1304,0.1320]

# ============================================================
# FIGURE 1 — HERO: Final Dev WER comparison (LinkedIn cover)
# ============================================================
fig, ax = plt.subplots(figsize=(11, 6.5))
models = ['BiLSTM\n+ CTC', 'Transformer\n+ CTC', 'Conformer\n+ Decoder\n(Seq2Seq)', 'Conformer\n+ CTC ★']
best_wer = [66.62, 36.34, 8.76, 13.04]
colors = ['#9aa0a6', '#5f6368', '#4285f4', '#0b8043']
bars = ax.bar(models, best_wer, color=colors, edgecolor='black', linewidth=0.5, width=0.65)
for bar, v in zip(bars, best_wer):
    ax.text(bar.get_x() + bar.get_width()/2, v + 1.5, f'{v:.2f}%',
            ha='center', va='bottom', fontsize=14, fontweight='bold')
ax.set_ylabel('Best Dev Word Error Rate (%)', fontsize=13)
ax.set_title('Saudi Sign Language Recognition — Model Comparison\n(Isharah 1000 dataset, 9,500 training / 949 dev samples)',
             fontsize=14, fontweight='bold', pad=18)
ax.set_ylim(0, 80)
ax.grid(axis='y', alpha=0.3)
ax.text(0.5, -0.18,
        '★  Conformer + CTC is our final deployed model (9.4M parameters, 40 epochs OneCycleLR)\n'
        'Result: 80% relative WER reduction vs. the BiLSTM baseline',
        ha='center', va='top', transform=ax.transAxes, fontsize=11, style='italic', color='#333')
save_both(fig, '01_model_comparison_hero')

# ============================================================
# FIGURE 2 — Training dynamics for the WINNING model (Conformer+CTC)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
epochs = np.arange(1, 41)
# Left: losses
axes[0].plot(epochs, CONF_LOSS, 'o-', color='#0b8043', label='Train CTC loss', linewidth=2, markersize=4)
axes[0].plot(epochs, CONF_VAL_LOSS, 's-', color='#d93025', label='Dev CTC loss', linewidth=2, markersize=4)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('CTC Loss', fontsize=12)
axes[0].set_title('Conformer + CTC — Loss Curves', fontsize=13, fontweight='bold')
axes[0].legend(loc='upper right', fontsize=11)
axes[0].grid(alpha=0.3)
# Right: WER
axes[1].plot(epochs, np.array(CONF_WER)*100, 'o-', color='#0b8043', linewidth=2, markersize=4)
best_ep = int(np.argmin(CONF_WER)) + 1
best_v  = min(CONF_WER) * 100
axes[1].axvline(best_ep, ls='--', color='#d93025', alpha=0.6)
axes[1].annotate(f'Best: {best_v:.2f}% @ epoch {best_ep}',
                 xy=(best_ep, best_v), xytext=(best_ep-15, best_v+12),
                 fontsize=11, fontweight='bold', color='#d93025',
                 arrowprops=dict(arrowstyle='->', color='#d93025'))
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Dev Word Error Rate (%)', fontsize=12)
axes[1].set_title('Conformer + CTC — Dev WER', fontsize=13, fontweight='bold')
axes[1].grid(alpha=0.3)
plt.suptitle('Best Model Training Dynamics (40 epochs, OneCycleLR, AdamW lr=3e-4)',
             fontsize=14, fontweight='bold', y=1.02)
save_both(fig, '02_conformer_ctc_training')

# ============================================================
# FIGURE 3 — All four models, Dev WER over training
# ============================================================
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(range(1, len(BILSTM_WER)+1), np.array(BILSTM_WER)*100, 'o-', color='#9aa0a6', label='BiLSTM + CTC (37 ep, interrupted)', linewidth=1.6, markersize=4)
ax.plot(range(1, len(TRF_WER)+1),    np.array(TRF_WER)*100,    's-', color='#5f6368', label='Transformer + CTC (30 ep)', linewidth=1.6, markersize=4)
ax.plot(range(1, len(SEQ_WER)+1),    np.array(SEQ_WER)*100,    '^-', color='#4285f4', label='Conformer + Decoder (30 ep)', linewidth=1.6, markersize=4)
ax.plot(range(1, len(CONF_WER)+1),   np.array(CONF_WER)*100,   'D-', color='#0b8043', label='Conformer + CTC (40 ep)', linewidth=2.2, markersize=4)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Dev Word Error Rate (%)', fontsize=12)
ax.set_title('Dev WER Convergence Across All Four Architectures',
             fontsize=14, fontweight='bold', pad=12)
ax.legend(loc='upper right', fontsize=11)
ax.set_ylim(0, 110)
ax.grid(alpha=0.3)
save_both(fig, '03_all_models_dev_wer')

# ============================================================
# FIGURE 4 — Architecture diagram: Conformer + CTC pipeline
# ============================================================
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

def block(x, y, w, h, text, fc='#cfe2f3', tc='black', fontsize=10, fw='normal'):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.1",
                         linewidth=1.2, edgecolor='black', facecolor=fc)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, color=tc, fontweight=fw)

def arrow(x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#333'))

block(2.5, 10.8, 5, 0.8, 'Input video → 86-joint skeleton pose\n(T frames, 86 joints, 2 coords)', '#fbe5cd', fontsize=10, fw='bold')
arrow(5, 10.7, 5, 10.0)
block(2.0, 9.2, 6, 0.7, 'Body-centric normalization + std scaling', '#fbe5cd', fontsize=10)
arrow(5, 9.1, 5, 8.5)
block(2.0, 7.7, 6, 0.7, 'Flatten + velocity features → 344-dim input', '#fbe5cd', fontsize=10)
arrow(5, 7.6, 5, 7.0)
block(2.0, 6.2, 6, 0.7, 'Augmentation: time warp, rotation,\nscale, joint dropout, frame mask', '#fbe5cd', fontsize=9.5)
arrow(5, 6.1, 5, 5.5)
block(2.0, 4.7, 6, 0.7, 'Linear projection → d_model=256 + sinusoidal PE', '#cfe2f3', fontsize=10)
arrow(5, 4.6, 5, 4.0)
block(1.5, 2.6, 7, 1.3, '6 × Conformer Block\n(Macaron FFN → MHSA → Conv → FFN → LN)\nd_model=256, heads=4, conv_kernel=31', '#cfe2f3', fontsize=10, fw='bold')
arrow(5, 2.5, 5, 1.9)
block(2.0, 1.2, 6, 0.7, 'Linear (256 → 678) → log-softmax', '#cfe2f3', fontsize=10)
arrow(5, 1.1, 5, 0.5)
block(2.5, -0.2, 5, 0.6, 'CTC decode → Arabic gloss sequence', '#d9ead3', fontsize=10, fw='bold')

ax.text(0.2, 11.5, '(A) Input pipeline', fontsize=10, fontweight='bold', color='#a64d00')
ax.text(0.2, 5.0,  '(B) Conformer encoder', fontsize=10, fontweight='bold', color='#1c4587')
ax.text(0.2, 0.4,  '(C) CTC head', fontsize=10, fontweight='bold', color='#274e13')

ax.set_title('Conformer + CTC Architecture — 9.4M parameters', fontsize=13, fontweight='bold', pad=14)
save_both(fig, '04_architecture_conformer_ctc')

# ============================================================
# FIGURE 5 — Top-10 missed words, Conformer + Seq2Seq decoder analysis
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
words = [f'{ar(w)} ({en})' for w, en in [
    ('انا', 'I'), ('هو', 'he'), ('سوال', 'Q-mark'), ('رغبه', 'want'),
    ('ذهاب', 'go'), ('اكل', 'eat'), ('الان', 'now'), ('مدرسه', 'school'),
    ('صديق', 'friend'), ('وقت', 'time')]]
counts = [59, 56, 31, 24, 18, 16, 15, 13, 13, 12]
colors_b = plt.cm.viridis(np.linspace(0.15, 0.85, len(words)))
bars = ax.barh(words[::-1], counts[::-1], color=colors_b, edgecolor='black', linewidth=0.4)
for bar, c in zip(bars, counts[::-1]):
    ax.text(c + 0.7, bar.get_y() + bar.get_height()/2, str(c),
            va='center', fontsize=10, fontweight='bold')
ax.set_xlabel('Number of missed predictions', fontsize=12)
ax.set_title('Top-10 Most-Missed Gloss Tokens on Dev Set\n(Seq2Seq decoder analysis — short function words dominate)',
             fontsize=13, fontweight='bold', pad=12)
ax.grid(axis='x', alpha=0.3)
save_both(fig, '05_top_missed_words')

# ============================================================
# FIGURE 6 — Sample predictions table (image of qualitative results)
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5))
ax.axis('off')
header = ['ID', 'Reference (Ground Truth)', 'Conformer + CTC Prediction', 'Status']
rows = [
    ['02_0001', 'سوال هو',                       'سوال',                       'Deletion'],
    ['02_0002', 'هو معلم لغه اشاره',             'هو معرفه لغه اشاره',         'Substitution'],
    ['02_0003', 'استفهام هو معلم هو',            'هو معلم هو',                 'Deletion'],
    ['02_0004', 'هو معلم لا انا مدرسه',          'هو معلم لا انا مدرسه',       'Exact ✓'],
    ['02_0005', 'هو سوال',                       'هو سوال',                    'Exact ✓'],
    ['02_0006', 'هو صديق مدرسه',                 'هو صديق مدرسه',              'Exact ✓'],
    ['02_0007', 'استفهام هو صديق مدرسه',         'استفهام هو صديق مدرسه',      'Exact ✓'],
    ['02_0008', 'انا اسره رقم ثلاث اشخاص',       'انا اسره رقم ثلاث',          'Deletion'],
    ['02_0009', 'رقم عمر ابن بنت',               'رقم عمر ابن بنت',            'Exact ✓'],
    ['02_0010', 'عمر اربع سن عمر',               'عمر اربع سن عمر',            'Exact ✓'],
]
table = ax.table(cellText=rows, colLabels=header,
                 colWidths=[0.10, 0.36, 0.36, 0.18],
                 loc='center', cellLoc='center')
table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.7)
# header
for j in range(len(header)):
    c = table[(0, j)]; c.set_facecolor('#1c4587'); c.set_text_props(color='white', weight='bold')
for i, r in enumerate(rows, 1):
    status_cell = table[(i, 3)]
    if 'Exact' in r[3]:
        status_cell.set_facecolor('#d9ead3')
    elif 'Deletion' in r[3]:
        status_cell.set_facecolor('#fce5cd')
    elif 'Substitution' in r[3]:
        status_cell.set_facecolor('#f4cccc')
ax.set_title('Conformer + CTC — Sample Predictions on Dev Set (first 10 samples)',
             fontsize=13, fontweight='bold', pad=10)
save_both(fig, '06_sample_predictions')

# ============================================================
# FIGURE 7 — Skeleton joint layout (the pose representation)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 9))
ax.set_xlim(-1.5, 1.5); ax.set_ylim(-2.2, 1.5); ax.set_aspect('equal'); ax.axis('off')

def draw_chain(pts, color, lw=2):
    pts = np.asarray(pts)
    ax.plot(pts[:,0], pts[:,1], '-', color=color, lw=lw, alpha=0.7)
    ax.plot(pts[:,0], pts[:,1], 'o', color=color, markersize=5, mec='black', mew=0.4)

# Body skeleton (11 joints)
body_pts = {
    'nose':(0,1.0), 'neck':(0,0.6),
    'r_shoulder':(-0.5,0.5), 'l_shoulder':(0.5,0.5),
    'r_elbow':(-0.8,0.0), 'l_elbow':(0.8,0.0),
    'r_wrist':(-1.0,-0.4), 'l_wrist':(1.0,-0.4),
    'mid_hip':(0,-0.5),
    'r_hip':(-0.25,-0.5), 'l_hip':(0.25,-0.5),
}
draw_chain([body_pts['nose'], body_pts['neck']], '#1c4587', 2.5)
draw_chain([body_pts['r_shoulder'], body_pts['neck'], body_pts['l_shoulder']], '#1c4587', 2.5)
draw_chain([body_pts['r_shoulder'], body_pts['r_elbow'], body_pts['r_wrist']], '#1c4587', 2.5)
draw_chain([body_pts['l_shoulder'], body_pts['l_elbow'], body_pts['l_wrist']], '#1c4587', 2.5)
draw_chain([body_pts['neck'], body_pts['mid_hip']], '#1c4587', 2.5)
draw_chain([body_pts['r_hip'], body_pts['mid_hip'], body_pts['l_hip']], '#1c4587', 2.5)

# Face circle + lip points (face/lips=33; here just illustrative)
face_th = np.linspace(0, 2*np.pi, 60); ax.plot(0.18*np.cos(face_th), 1.0+0.22*np.sin(face_th), color='#9aa0a6', lw=1)
lip_pts = [(0.06*np.cos(t), 0.92+0.04*np.sin(t)) for t in np.linspace(0, 2*np.pi, 20)]
for p in lip_pts: ax.plot(p[0], p[1], 'o', color='#d93025', markersize=3, mec='black', mew=0.3)

# Right hand 21 joints (sketch)
def hand(cx, cy, color):
    palm = (cx, cy)
    ax.plot(*palm, 'o', color=color, markersize=6, mec='black', mew=0.5)
    for offset_x, lengths in zip([-0.18, -0.09, 0.0, 0.09, 0.18], [[0.08,0.06,0.05]]*5):
        prev = palm
        cur = (cx + offset_x, cy - 0.05)
        ax.plot([prev[0], cur[0]], [prev[1], cur[1]], '-', color=color, lw=1.2, alpha=0.7)
        prev = cur
        for L in lengths:
            nx = cur[0] + offset_x*0.4
            ny = cur[1] - L
            ax.plot([cur[0], nx], [cur[1], ny], '-', color=color, lw=1.2, alpha=0.7)
            ax.plot(nx, ny, 'o', color=color, markersize=3.5, mec='black', mew=0.3)
            cur = (nx, ny)
hand(-1.0, -0.55, '#0b8043')
hand( 1.0, -0.55, '#0b8043')

# Labels with counts
ax.text(-1.45, 1.3, 'Face / lips: 33 joints', fontsize=11, color='#d93025', fontweight='bold')
ax.text(-1.45, 1.15, 'Body: 11 joints', fontsize=11, color='#1c4587', fontweight='bold')
ax.text(-1.45, 1.00, 'Hands: 21 × 2 = 42 joints', fontsize=11, color='#0b8043', fontweight='bold')
ax.text(0, -1.8, '86 keypoints × 2 (x,y) = 172 raw features\n+ 172 velocity features = 344-dim input',
        ha='center', fontsize=11.5, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff2cc', edgecolor='black'))
ax.set_title('Skeleton Pose Representation (Isharah 1000)',
             fontsize=13, fontweight='bold', pad=10)
save_both(fig, '07_skeleton_layout')

# ============================================================
# FIGURE 8 — Dataset overview
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left — sample-count bar
labels = ['Train\n(valid)', 'Dev', 'Test\n(blind)']
counts = [9500, 949, 3800]
bars = axes[0].bar(labels, counts, color=['#0b8043', '#4285f4', '#d93025'],
                   edgecolor='black', linewidth=0.6, width=0.55)
for bar, c in zip(bars, counts):
    axes[0].text(bar.get_x()+bar.get_width()/2, c+150, f'{c:,}',
                 ha='center', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Number of samples', fontsize=12)
axes[0].set_title('Dataset Splits', fontsize=13, fontweight='bold')
axes[0].set_ylim(0, 11000); axes[0].grid(axis='y', alpha=0.3)

# Right — vocabulary / sequence length stats
stats = ['Vocab\nsize', 'Avg train\nframes', 'Avg dev\nframes', 'Avg gloss\nlength']
vals  = [675, 214, 327, 4.8]
bars2 = axes[1].bar(stats, vals, color=['#674ea7', '#9900ff', '#a64d79', '#cc0000'],
                    edgecolor='black', linewidth=0.6, width=0.55)
for bar, v in zip(bars2, vals):
    axes[1].text(bar.get_x()+bar.get_width()/2, v+10, f'{v:g}',
                 ha='center', fontsize=12, fontweight='bold')
axes[1].set_yscale('log')
axes[1].set_title('Key Statistics', fontsize=13, fontweight='bold')
axes[1].set_ylim(1, 700); axes[1].grid(axis='y', alpha=0.3, which='both')

plt.suptitle('Isharah 1000 — Saudi Sign Language Dataset',
             fontsize=14, fontweight='bold', y=1.02)
save_both(fig, '08_dataset_overview')

print('All figures saved to:', LI_DIR, 'and', RP_DIR)
print('PNG files generated:')
for f in sorted(os.listdir(LI_DIR)):
    full = os.path.join(LI_DIR, f)
    print(f'  {f}  ({os.path.getsize(full)//1024} KB)')
