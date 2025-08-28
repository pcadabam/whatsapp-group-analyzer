import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
import numpy as np

# Create a preview image for social sharing
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
fig.patch.set_facecolor('#667eea')

# Remove axes
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# Title
ax.text(5, 4.5, '💬 WhatsApp Group Analyzer', 
        fontsize=32, fontweight='bold', color='white', 
        ha='center', va='center')

# Subtitle
ax.text(5, 3.5, 'Discover Your Group\'s Personality & Secrets', 
        fontsize=18, color='white', ha='center', va='center', alpha=0.9)

# Features
features = [
    '🏆 Top Chatters & Activity Patterns',
    '😄 Sentiment Analysis & Mood Tracking', 
    '🔥 Hot Topics & Discussion Themes',
    '🎭 Fun Personality Awards'
]

for i, feature in enumerate(features):
    ax.text(5, 2.5 - (i * 0.4), feature, 
            fontsize=14, color='white', ha='center', va='center', alpha=0.8)

# Call to action
ax.text(5, 0.5, '🚀 Free, Secure & Anonymous Analysis', 
        fontsize=16, fontweight='bold', color='white', 
        ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='rgba(255,255,255,0.2)', 
                  edgecolor='white', alpha=0.8))

# Add gradient effect using patches
from matplotlib.colors import LinearSegmentedColormap
colors = ['#667eea', '#764ba2']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

# Create gradient background
gradient = np.linspace(0, 1, 256).reshape(1, -1)
gradient = np.vstack((gradient, gradient))

ax.imshow(gradient, extent=[0, 10, 0, 6], aspect='auto', cmap=cmap, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/pcadabam/Projects/whatsapp-group-analyzer/preview.png', 
            dpi=150, bbox_inches='tight', facecolor='#667eea', edgecolor='none')
plt.close()

print("Preview image created: preview.png")