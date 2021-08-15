import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from skmultiflow.drift_detection import ADWIN, DDM, EDDM, HDDM_A, HDDM_W, KSWIN, PageHinkley


def seq_to_zones(seq):
    start, end = seq[0], seq[0]
    count = start
    for item in seq:
        if not count == item:
            yield start, end
            start, end = item, item
            count = item
        end = item
        count += 1
    yield start, end


def plot_data(df, warning_zones=[], change_points=[], y='x', title=None):
    plt.figure(figsize=(15,5))
    sns.lineplot(
        data=df, 
        y=y,
        x=range(0, df.shape[0]),
        label=y
    )
    # change points
    for change_point in change_points:
        plt.axvline(
            x=change_point, 
            color='red',
            label='change points'
        ) 
    # warning zones
    if warning_zones:
        warning_zones = list(seq_to_zones(sorted(warning_zones)))
    for warning_zone in warning_zones:
        plt.axvspan(
            xmin=warning_zone[0], 
            xmax=warning_zone[1], 
            alpha=0.35, 
            color='orange', 
            label='warning zones'
        )
    if title:
        plt.title(title)
    # Stop matplotlib repeating labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()


def get_cd_methods():
    cd_methods_dict = {
        'ADWIN': ADWIN(), 
        'DDM': DDM(), 
        'EDDM': EDDM(), 
        'HDDM_A': HDDM_A(), 
        'HDDM_W': HDDM_W(), 
        'KSWIN': KSWIN(), 
        'PageHinkley': PageHinkley()
    }
    return cd_methods_dict


def drift_detection(data, cd_method=ADWIN(), verbose=True):
    change_points, warning_zones = list(), list()
    if verbose:
        cd_method_name = cd_method.get_info().split('(')[0]
        print(f'========================= Method: {cd_method_name} =========================')
    for idx, row in data.iterrows():
        elem = row['x']
        cd_method.add_element(elem)
        if cd_method.detected_warning_zone():
            warning_zones.append(idx)
            if verbose:
                print(f'Warning zone has been detected: {elem} - of index: {idx}')
        if cd_method.detected_change():
            change_points.append(idx)
            if verbose:
                print(f'Change has been detected: {elem} - of index: {idx}')
    if verbose:
        print('=='*32)
        print()
    return warning_zones, change_points