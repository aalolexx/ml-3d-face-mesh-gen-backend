def set_sns_style(sns):
    sns.set_theme()
    sns.set_context('paper')
    sns.set(font_scale=1.01)

def save_fig(plt, path):
    plt.savefig(path, bbox_inches='tight', pad_inches=0.025)