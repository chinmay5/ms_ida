import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.rcParams['figure.dpi'] = 500
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def show_baseline_dist_plot():
    info_dict = {
        'region': ['Periventricular', 'Subcortical', '(Juxta)cortical', 'Infratentorial'],
        '1y': np.array([2370, 3902, 2126, 554]),
        '2y': np.array([1947, 3214, 1798, 472]),
    }
    colors = ['b', 'g']
    x_pos = ['1 year', '2 year']
    df = pd.DataFrame(info_dict)
    df.plot(x="region", y=["1y", "2y"], kind="bar", color=colors, alpha=0.5, rot=0, fontsize=12, xlabel='')
    plt.ylabel('Lesion Count', fontsize=15)
    # plt.xlabel('Brain region', fontsize=15)
    color_dict = {color_tuple: 1 for color_tuple in colors}
    handles = generate_plot_handles(x_pos, color_dict)
    plt.legend(handles=handles, loc=1, prop={'size': 15})
    plt.tight_layout()
    plt.savefig('value_dist.png', dpi=250)
    plt.show()


def show_dist_change_2yplot():
    # 1y: {1: 1185, 2: 1105, 3: 2011, 4: 286}
    # 2y: {1: 403, 2: 408, 3: 693, 4: 124}
    info_dict = {
        'region': ['Periventricular', 'Subcortical', '(Juxta)cortical', 'Infratentorial'],
        'original': np.array([1947, 3214, 1798, 472]),
        'updated': np.array([99, 136, 88, 26]),
    }
    # Let us normalize the lesion distribution
    info_dict['original'] = info_dict['original'] / np.sum(info_dict['original'])
    info_dict['updated'] = info_dict['updated'] / np.sum(info_dict['updated'])
    colors = ['b', 'g']
    x_pos = ['original', 'post-SPM']
    df = pd.DataFrame(info_dict)
    df.plot(x="region", y=["original", "updated"], kind="bar", color=colors, alpha=0.25, rot=0, fontsize=12, xlabel='')
    # We want to plot the two images side by side so this information might be redundant
    # plt.ylabel('Lesion Proportion', fontsize=15)
    # plt.xlabel('Brain region', fontsize=15)
    color_dict = {color_tuple: 1 for color_tuple in colors}
    handles = generate_plot_handles(x_pos, color_dict)
    plt.legend(handles=handles, loc=1, prop={'size': 15})
    # plt.tight_layout()
    plt.title('Two year Lesion Proportion', fontsize=15)
    plt.savefig('value_dist2y.png', dpi=250)
    plt.show()


def show_dist_change_1yplot():
    # 1y: {1: 1185, 2: 1105, 3: 2011, 4: 286}
    info_dict = {
        'region': ['Periventricular', 'Subcortical', '(Juxta)cortical', 'Infratentorial'],
        'original': np.array([2370, 3902, 2126, 554]),
        'updated': np.array([124, 156, 98, 54]),
    }
    # Let us normalize the lesion distribution
    info_dict['original'] = info_dict['original'] / np.sum(info_dict['original'])
    info_dict['updated'] = info_dict['updated'] / np.sum(info_dict['updated'])
    colors = ['b', 'g']
    x_pos = ['original', 'post-SPM']
    df = pd.DataFrame(info_dict)
    df.plot(x="region", y=["original", "updated"], kind="bar", color=colors, alpha=0.25, rot=0, fontsize=12, xlabel='')
    plt.ylabel('Lesion Proportion', fontsize=15)
    # plt.xlabel('Brain region', fontsize=15)
    color_dict = {color_tuple: 1 for color_tuple in colors}
    handles = generate_plot_handles(x_pos, color_dict)
    plt.legend(handles=handles, loc=1, prop={'size': 15})
    plt.title('One year Lesion Proportion', fontsize=15)
    # plt.tight_layout()
    plt.savefig('value_dist1y.png', dpi=250)
    plt.show()


def generate_plot_handles(brain_reg, color_dict):
    # Sorting using our assigned order
    handles = []
    for name, color in zip(brain_reg, color_dict.keys()):
        handles.append(mpatches.Patch(color=color, label=name, alpha=0.25))
    return handles


def show_knn_plot():
    k = [2, 3, 4, 5, 6, 7, 8]
    auc_gcn_1y = [0.633, 0.650, 0.659, 0.671, 0.649, 0.642, 0.633]
    auc_gcn_2y = [0.631, 0.638, 0.661, 0.664, 0.657, 0.642, 0.638]

    # plt.rcParams["figure.figsize"] = (8, 6)
    fig, ax = plt.subplots()

    ax.plot(k, auc_gcn_1y, color='green', marker='+', alpha=0.55, linewidth=2.0, label='AUC One year')
    ax.plot(k, auc_gcn_2y, color='blue', marker='*', alpha=0.55, linewidth=2.0, label='AUC Two year')
    ax.plot()
    # ax.axis('equal')
    leg = ax.legend(fontsize=16, loc='lower center')
    plt.title('', fontsize=18)
    plt.xlabel(r'retention ratio', fontsize=18)
    plt.ylabel('score', fontsize=18)
    plt.grid(True)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig('knn_effect_plot.png', dpi=1000)
    plt.show()


def show_num_conv_plot():
    k = [1, 2, 3, 4, 5]
    auc_gcn_1y = [0.649, 0.671, 0.645, 0.637, 0.635]
    auc_gcn_2y = [0.656, 0.664, 0.636, 0.628, 0.617]

    # plt.rcParams["figure.figsize"] = (8, 6)
    fig, ax = plt.subplots()

    ax.plot(k, auc_gcn_1y, color='green', marker='+', alpha=0.55, linewidth=2.0, label='AUC One year')
    ax.plot(k, auc_gcn_2y, color='blue', marker='*', alpha=0.55, linewidth=2.0, label='AUC Two year')
    ax.plot()
    # ax.axis('equal')
    leg = ax.legend(fontsize=16, loc='lower center')
    plt.title('', fontsize=18)
    plt.xlabel(r'conv layers', fontsize=18)
    plt.ylabel('score', fontsize=18)
    plt.grid(True)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig('num_conv_plot.png', dpi=1000)
    plt.show()


def show_retention_ratio_plot():
    # lambda_ = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]#, 1.0]
    # auc_sage_2y = [0.678, 0.682, 0.664, 0.656, 0.645, 0.654, 0.649, 0.643, 0.642]#, 0.636]
    # auc_sage_1y = [0.631, 0.664, 0.662, 0.659, 0.654, 0.652, 0.651, 0.645, 0.640]#, 0.651]
    # auc_brats = [0.721, 0.753, 0.767, 0.730, 0.674]

    lambda_ = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    auc_gcn_1y = [0.639, 0.641, 0.648, 0.654, 0.671, 0.661, 0.659, 0.651, 0.643, 0.640]  # , 0.642]
    auc_gcn_2y = [0.634, 0.639, 0.646, 0.650, 0.664, 0.656, 0.654, 0.647, 0.645, 0.643]  # , 0.642]

    # plt.rcParams["figure.figsize"] = (8, 6)
    fig, ax = plt.subplots()

    ax.plot(lambda_, auc_gcn_1y, color='green', marker='+', alpha=0.55, linewidth=2.0, label='AUC One year')
    ax.plot(lambda_, auc_gcn_2y, color='blue', marker='*', alpha=0.55, linewidth=2.0, label='AUC Two year')
    ax.plot()
    # ax.axis('equal')
    leg = ax.legend(fontsize=16, loc='lower center')
    plt.title('', fontsize=18)
    plt.xlabel(r'retention ratio', fontsize=18)
    plt.ylabel('score', fontsize=18)
    plt.grid(True)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig('retention_ratio_plot.png', dpi=1000)
    plt.show()


if __name__ == '__main__':
    show_retention_ratio_plot()
    # show_baseline_dist_plot()
    # show_dist_change_1yplot()
    # show_dist_change_2yplot()
    show_knn_plot()
    show_num_conv_plot()
