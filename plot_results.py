# imports
import json
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('figures/pngs', exist_ok=True)
os.makedirs('figures/pdfs', exist_ok=True)

settings = ['early_fusion', 'late_fusion', 'mixture_of_experts']
setting_labels = ['Early fusion', 'Late fusion', 'MoE']
colors = ['b', 'g', 'r', 'c']
hatches = ['', '///', '\\\\\\']
classes = ['empty', 'midden', 'mound', 'water']
trials = range(1, 51)
class_eval_metrics = ['precision', 'recall', 'f1']
overall_eval_metrics = ['average_precision', 'average_recall', 'average_f1', 'auc']
overall_eval_metric_labels = ['Precision', 'Recall', 'F1', 'AUC']
class_results = {setting: {eval_metric: {image_class: {} for image_class in classes} for eval_metric in class_eval_metrics} for setting in settings}
overall_results = {setting: {eval_metric: {} for eval_metric in overall_eval_metrics} for setting in settings}
width = 0.1
columnspacing = 1
handletextpad = 0.5
handlelength = 1
labelspacing = 0.4
fontsize = 9

for setting in settings:
    for trial in trials:
        with open(f'results/{setting}/{setting}_{trial}.json', 'r') as results_json:
            data = json.load(results_json)

            for eval_metric in class_eval_metrics:
                for image_class in classes:
                    class_results[setting][eval_metric][image_class][trial] = data[eval_metric][classes.index(image_class)]

            for eval_metric in overall_eval_metrics:
                overall_results[setting][eval_metric][trial] = data[eval_metric]

class_results_avg = {setting: {eval_metric: {image_class: np.mean([class_results[setting][eval_metric][image_class][trial] for trial in trials]) for image_class in classes} for eval_metric in class_eval_metrics} for setting in settings}
class_results_se = {setting: {eval_metric: {image_class: np.std([class_results[setting][eval_metric][image_class][trial] for trial in trials])/np.sqrt(len(trials)) for image_class in classes} for eval_metric in class_eval_metrics} for setting in settings}

overall_results_avg = {setting: {eval_metric: np.mean([overall_results[setting][eval_metric][trial] for trial in trials]) for eval_metric in overall_eval_metrics} for setting in settings}
overall_results_se = {setting: {eval_metric: np.std([overall_results[setting][eval_metric][trial] for trial in trials])/np.sqrt(len(trials)) for eval_metric in overall_eval_metrics} for setting in settings}

def check_results_complete():
    for setting in settings:
        for trial in trials:
            if not os.path.exists(f'results/{setting}/{setting}_{trial}.json'):
                print(f'{setting.capitalize()} trial {trial} does not exist')

def plot_class_results(image_class):
    class_avg_results_plot = [[class_results_avg[setting][eval_metric][image_class] for eval_metric in class_eval_metrics] for setting in settings]
    class_se_results_plot = [[class_results_se[setting][eval_metric][image_class] for eval_metric in class_eval_metrics] for setting in settings]

    fig, ax = plt.subplots(dpi=300)
    plt.subplots_adjust(top=0.4, right=0.5)
    x = np.arange(len(class_eval_metrics))
    multiplier = 0

    for i in range(len(class_avg_results_plot)):
        offset = width * multiplier
        ax.bar(x+offset, class_avg_results_plot[i], width, yerr=2*np.array(class_se_results_plot[i]), label=setting_labels[i], color=colors[i], hatch=hatches[i])
        print(class_avg_results_plot[i])
        multiplier += 1

    if image_class != 'empty':
        ax.set_ylim(0, 0.9)

    ax.set_xticks(x + width)
    ax.set_xticklabels([eval_metric.capitalize() for eval_metric in class_eval_metrics])
    ax.set_title(f'{image_class.capitalize()} Class')
    plt.savefig(f'figures/pdfs/{image_class}_results.pdf', bbox_inches='tight', pad_inches=0.1, transparent=False)
    plt.savefig(f'figures/pngs/{image_class}_results.png', bbox_inches='tight', pad_inches=0.1, transparent=False)
    print(f'Plotted {image_class} results')

def plot_overall_results():
    overall_avg_results_plot = [[overall_results_avg[setting][eval_metric] for eval_metric in overall_eval_metrics] for setting in settings]
    overall_se_results_plot = [[overall_results_se[setting][eval_metric] for eval_metric in overall_eval_metrics] for setting in settings]

    fig, ax = plt.subplots(dpi=300)
    plt.subplots_adjust(top=0.4, right=0.5)
    x = np.arange(len(overall_eval_metrics))
    multiplier = 0

    for i in range(len(overall_avg_results_plot)):
        offset = width * multiplier
        ax.bar(x+offset, overall_avg_results_plot[i], width, yerr=2*np.array(overall_se_results_plot[i]), label=setting_labels[i], color=colors[i], hatch=hatches[i])
        print(overall_avg_results_plot[i])
        multiplier += 1

    ax.set_ylim(0, 0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(overall_eval_metric_labels)
    ax.set_title('Macro-Average')
    ax.legend(ncol=1, columnspacing=columnspacing, handletextpad=handletextpad, handlelength=handlelength, labelspacing=labelspacing, fontsize=fontsize)
    plt.savefig(f'figures/pdfs/macro_averaged_results.pdf', bbox_inches='tight', pad_inches=0.1, transparent=False)
    plt.savefig(f'figures/pngs/macro_averaged_results.png', bbox_inches='tight', pad_inches=0.1, transparent=False)
    print(f'Plotted macro-averaged results')

if __name__ == '__main__':
    check_results_complete()

    for image_class in classes:
        plot_class_results(image_class)

    plot_overall_results()
