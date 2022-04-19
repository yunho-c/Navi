import seaborn as sns
import matplotlib.pyplot as plt

x='Mutual information (iris) - gradient method'
y='Gaze error relative'

#plt.style.use('seaborn-ticks')

style = {
    'text.usetex': True,
    'font.family': ['STIXGeneral'],
    
    # Make legend/label fonts a bit smaller
    
    #'lines.markeredgecolor': 'none',
    #'markers.fillstyle': 'none',

    'lines.linewidth': 1,
    'axes.linewidth': 0.5,
    'lines.markersize': 4,
    
    #'grid.color': '000000',
    'grid.linewidth': 0.5,
    
    'axes.labelsize': 9,
    'axes.grid': True,
    'axes.grid.which': 'both',
    'font.size': 9,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    
    'xtick.major.width': 0.5,
    'xtick.minor.width': 0.2,
    'ytick.major.width': 0.5,
    'ytick.minor.width': 0.2,
    #"xtick.bottom" : True, 
    #"ytick.left" : True,
    #'xtick.minor.visible': True,
    #'ytick.minor.visible': True,
}

#sns.set_style('ticks', rc=style)
#sns.set_context('paper', rc=context)
#sns.set_style()
sns.set_theme(context='paper', style='ticks', palette='colorblind', font='STIXGeneral', rc=style)


styles = {
    'article-col': 241.14749,
    'article-full': 506.295,
    'thesis': 360.0,
}

golden_ratio = (1+5**.5)/2

def gen_grid(width, fraction=1, aspect=golden_ratio, col_wrap=1):
    if isinstance(width, str):
        width = styles[width]
     
    inches_per_pt = 1/ 72.27
    
    fig_width_pt = width * fraction
    fig_width_in = fig_width_pt * inches_per_pt / col_wrap
    fig_height_in = fig_width_in / aspect
    return fig_height_in

def set_size(width, fraction=1, subplots=(1, 1), ratio=(5**.5 - 1)/2):
    if isinstance(width, str):
        width = styles[width]
        
    fig_width_pt = width * fraction
    inches_per_pt = 1/ 72.27
    
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * ratio * (subplots[0] / subplots[1])
    
    return (fig_width_in, fig_height_in)

def plots(width, fraction=1, subplots=(1, 1), **kwargs):
    return plt.subplots(subplots[0], subplots[1], figsize=set_size(width, fraction, subplots), **kwargs)