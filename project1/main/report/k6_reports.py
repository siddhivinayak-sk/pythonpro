import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np

"""
This report collects data from K6 results and generate charts for the provided test scenarios.
cmd: python reports.py <parent_directory_which_contains_list_of_test_cycle>

Hierarchy of the directory & test:
parent_directory:
  test_cycle_1:
    K6_result_load-ABC_SCENARIO-1.json
    K6_result_load-ABC_SCENARIO-2.json
    K6_result_load-ABC_SCENARIO-3.json
    K6_result_load-DEF_SCENARIO-1.json
    K6_result_load-DEF_SCENARIO-2.json
    K6_result_load-DEF_SCENARIO-3.json
    resource.csv
  test_cycle_2:
    K6_result_load-ABC_SCENARIO-1.json
    K6_result_load-ABC_SCENARIO-2.json
    K6_result_load-ABC_SCENARIO-3.json
    K6_result_load-DEF_SCENARIO-1.json
    K6_result_load-DEF_SCENARIO-2.json
    K6_result_load-DEF_SCENARIO-3.json
    resource.csv

Here, resource.csv contains the CPU and Memory usage for each k6 test cycle in CSV format e.g.:
test,cpu,memory
K6_output-ABC_SCENARIO-1.json,1.150,918.55
K6_output-ABC_SCENARIO-2.json,0.531,933.92
K6_output-ABC_SCENARIO-3.json,0.521,934.67
K6_output-DEF_SCENARIO-1.json,1.150,918.55
K6_output-DEF_SCENARIO-2.json,0.531,933.92
K6_output-DEF_SCENARIO-3.json,0.521,934.67

Additionally, it is possible to have multiple metrics for same scenario like:
DEF_SCENARIO has been tested with GET and POST http calls. In this case, there will be two metrics for same test
scenario. The metrics_label_mapping will be used to map the metric name to the label for the chart.   

Libs Used:
python -m pip install -U matplotlib
python -m pip install -U pandas
"""


test_and_scenario = {
    'ABC_SCENARIO': ['http_metrics_get_external'],
    'DEF_SCENARIO': ['http_metrics_get_external', 'http_metrics_post_external'],
}
metrics_label_mapping = {
    'http_metrics_get_external': 'GET External API',
    'http_metrics_post_external': 'POST External API',
}
cycle_label_mapping = {
    'SERVICENAME-0.0.1-ENVNAME-CYCLENAME1': 'Cycle 1 Label',
    'SERVICENAME-0.0.1-ENVNAME-CYCLENAME2': 'Cycle 2 Label',
}
base_cycle_name = 'SERVICENAME-0.0.1-ENVNAME-CYCLENAME1'
test_cpu_memory_resource_file_name = '/resource.csv'
test_json_file_name_prefix = 'K6_result_load-'
chart_dir_name = '/charts'
table_css = '<style>table {background: #caedfb; border: 1px solid black;border-spacing: 0px; font-family: sans-serif;} th, td {border: 1px solid black;padding: 4px;} th {background: #0f9ed5; color: #ffffff; font-weight: bold;} .base { background: #d904ab; padding: 3px; border-radius: 25px; } .target { background: #fcfc3f; padding: 3px; border-radius: 25px; color: #000;} tr:nth-child(even) {background: #caedfb} tr:nth-child(odd) {background: #FFF}</style>'


def iterate_over_dirs(path):
    """
    This function iterates over the directories and process each directory.
    """
    test_cycles = []
    base_cycle = None
    for test_dir_name in os.listdir(path):
        if os.path.isdir(os.path.join(path, test_dir_name)):
            print("Processing started for directory: " + test_dir_name)
            test_cycle = print_json_metrics(path, test_dir_name)
            transform_cycle_to_table(test_cycle, path, test_dir_name)
            test_cycles.append(test_cycle)
            if base_cycle_name == test_dir_name:
                base_cycle = test_cycle
            print("Processing completed for directory: " + test_dir_name)
    if base_cycle is not None:
        transform_to_table_comparative(base_cycle, test_cycles, path)


def transform_to_table_comparative(base_cycle, target_test_cycle, path):
    """
    Iterate over the test cycles and transform to table.
    """
    for test_cycle in target_test_cycle:
        if base_cycle.name != test_cycle.name:
            transform_to_table_comparative_cycle(base_cycle, test_cycle, path)


def transform_to_table_comparative_cycle(base_cycle, target_test_cycle, path):
    """
    Generate comparative results
    """
    target_cycle_name = target_test_cycle.name
    if base_cycle is not None:
        table = table_css
        table += '<table border="1">'
        table += '<tr>'
        table += f'<th colspan="20">{cycle_label(base_cycle.name)} vs {cycle_label(target_test_cycle.name)}</th>'
        table += '</tr>'
        table += '<tr>'
        table += '<th></th>'
        table += f'<th><span class="base">{base_cycle_name}</span></th>'
        table += f'<th><span class="target">{target_cycle_name}</span></th>'
        table += f'<th><span class="base">{base_cycle_name}</span></th>'
        table += f'<th><span class="target">{target_cycle_name}</span></th>'
        table += f'<th><span class="base">{base_cycle_name}</span></th>'
        table += f'<th><span class="target">{target_cycle_name}</span></th>'
        table += f'<th><span class="base">{base_cycle_name}</span></th>'
        table += f'<th><span class="target">{target_cycle_name}</span></th>'
        table += f'<th><span class="base">{base_cycle_name}</span></th>'
        table += f'<th><span class="target">{target_cycle_name}</span></th>'
        table += f'<th><span class="base">{base_cycle_name}</span></th>'
        table += f'<th><span class="target">{target_cycle_name}</span></th>'
        table += '<th></th>'
        table += f'<th><span class="base">{base_cycle_name}</span></th>'
        table += f'<th><span class="target">{target_cycle_name}</span></th>'
        table += f'<th><span class="base">{base_cycle_name}</span></th>'
        table += f'<th><span class="target">{target_cycle_name}</span></th>'
        table += f'<th><span class="base">{base_cycle_name}</span></th>'
        table += f'<th><span class="target">{target_cycle_name}</span></th>'
        table += '</tr>'
        table += '<tr>'
        table += '<th>Test Scenario</th>'
        table += '<th>Requests</th>'
        table += '<th>Requests</th>'
        table += '<th>VUs</th>'
        table += '<th>VUs</th>'
        table += '<th>RPS</th>'
        table += '<th>RPS</th>'
        table += '<th>Errors</th>'
        table += '<th>Errors</th>'
        table += '<th>CPU  (mills)</th>'
        table += '<th>CPU  (mills)</th>'
        table += '<th>Memory  (MB)</th>'
        table += '<th>Memory  (MB)</th>'
        table += '<th>Metric Name</th>'
        table += '<th>AVG (mills)</th>'
        table += '<th>AVG (mills)</th>'
        table += '<th>P90 (mills)</th>'
        table += '<th>P90 (mills)</th>'
        table += '<th>P95 (mills)</th>'
        table += '<th>P95 (mills)</th>'
        table += '</tr>'
        for k6_test in base_cycle.metrics:
            name_span = len(k6_test.metrics) * len(k6_test.metrics[0].test_metrics)
            metric_span = len(k6_test.metrics[0].test_metrics)
            for metric_index, metric in enumerate(k6_test.metrics):
                for test_metric_index, test_metric in enumerate(metric.test_metrics):
                    target_metric, target_test_metric = metric_and_test_metric_from_k6_test_cycle(target_test_cycle, k6_test.name, metric_index, test_metric.name)
                    if metric_index == 0 and test_metric_index == 0:
                        table += '<tr>'
                        table += f'<td rowspan="{name_span}">{k6_test.name}</td>'
                        table += f'<td rowspan="{metric_span}"><span class="base">{metric.reqs}</span></td>'
                        table += f'<td rowspan="{metric_span}"><span class="target">{target_metric.reqs}</span></td>'
                        table += f'<td rowspan="{metric_span}"><span class="base">{metric.vus}</span></td>'
                        table += f'<td rowspan="{metric_span}"><span class="target">{target_metric.vus}</span></td>'
                        table += f'<td rowspan="{metric_span}"><span class="base">{f2(metric.rates)}</span></td>'
                        table += f'<td rowspan="{metric_span}"><span class="target">{f2(target_metric.rates)}</span></td>'
                        table += f'<td rowspan="{metric_span}"><span class="base">{metric.errors}</span></td>'
                        table += f'<td rowspan="{metric_span}"><span class="target">{target_metric.errors}</span></td>'
                        table += f'<td rowspan="{metric_span}"><span class="base">{f3(metric.cpu)}</span></td>'
                        table += f'<td rowspan="{metric_span}"><span class="target">{f3(target_metric.cpu)}</span></td>'
                        table += f'<td rowspan="{metric_span}"><span class="base">{f2(metric.memory)}</span></td>'
                        table += f'<td rowspan="{metric_span}"><span class="target">{f2(target_metric.memory)}</span></td>'
                        table += f'<td>{metric_label(test_metric.name)}</td>'
                        table += f'<td><span class="base">{f2(test_metric.avg)}</span></td>'
                        table += f'<td><span class="target">{f2(target_test_metric.avg)}</span></td>'
                        table += f'<td><span class="base">{f2(test_metric.p90)}</span></td>'
                        table += f'<td><span class="target">{f2(target_test_metric.p90)}</span></td>'
                        table += f'<td><span class="base">{f2(test_metric.p95)}</span></td>'
                        table += f'<td><span class="target">{f2(target_test_metric.p95)}</span></td>'
                        table += '</tr>'
                    elif test_metric_index == 0:
                        table += '<tr>'
                        table += f'<td rowspan="{metric_span}"><span class="base">{metric.reqs}</span></td>'
                        table += f'<td rowspan="{metric_span}"><span class="target">{target_metric.reqs}</span></td>'
                        table += f'<td rowspan="{metric_span}"><span class="base">{metric.vus}</span></td>'
                        table += f'<td rowspan="{metric_span}"><span class="target">{target_metric.vus}</span></td>'
                        table += f'<td rowspan="{metric_span}"><span class="base">{f2(metric.rates)}</span></td>'
                        table += f'<td rowspan="{metric_span}"><span class="target">{f2(target_metric.rates)}</span></td>'
                        table += f'<td rowspan="{metric_span}"><span class="base">{metric.errors}</span></td>'
                        table += f'<td rowspan="{metric_span}"><span class="target">{target_metric.errors}</span></td>'
                        table += f'<td rowspan="{metric_span}"><span class="base">{f3(metric.cpu)}</span></td>'
                        table += f'<td rowspan="{metric_span}"><span class="target">{f3(target_metric.cpu)}</span></td>'
                        table += f'<td rowspan="{metric_span}"><span class="base">{f2(metric.memory)}</span></td>'
                        table += f'<td rowspan="{metric_span}"><span class="target">{f2(target_metric.memory)}</span></td>'
                        table += f'<td>{metric_label(test_metric.name)}</td>'
                        table += f'<td><span class="base">{f2(test_metric.avg)}</span></td>'
                        table += f'<td><span class="target">{f2(target_test_metric.avg)}</span></td>'
                        table += f'<td><span class="base">{f2(test_metric.p90)}</span></td>'
                        table += f'<td><span class="target">{f2(target_test_metric.p90)}</span></td>'
                        table += f'<td><span class="base">{f2(test_metric.p95)}</span></td>'
                        table += f'<td><span class="target">{f2(target_test_metric.p95)}</span></td>'
                        table += '</tr>'
                    else:
                        table += '<tr>'
                        table += f'<td>{metric_label(test_metric.name)}</td>'
                        table += f'<td><span class="base">{f2(test_metric.avg)}</span></td>'
                        table += f'<td><span class="target">{f2(target_test_metric.avg)}</span></td>'
                        table += f'<td><span class="base">{f2(test_metric.p90)}</span></td>'
                        table += f'<td><span class="target">{f2(target_test_metric.p90)}</span></td>'
                        table += f'<td><span class="base">{f2(test_metric.p95)}</span></td>'
                        table += f'<td><span class="target">{f2(target_test_metric.p95)}</span></td>'
                        table += '</tr>'
        table += '</table>'
        write_file_on_disk(path + '/' + target_test_cycle.name + '/' + chart_dir_name + '/' + base_cycle_name + '_vs_' + target_cycle_name + '.html', table)
        generate_comparison_charts(base_cycle, target_test_cycle, path)


def generate_comparison_charts(base_cycle, target_test_cycle, path):
    """
    Generate comparison charts for difference
    """
    for base_k6_test in base_cycle.metrics:
        target_k6_test = get_test_scenario_from_k6_test_cycle(target_test_cycle, base_k6_test.name)
        if target_k6_test is not None:
            last_instance_base_metrics = base_k6_test.metrics[-1]
            last_instance_target_metrics = target_k6_test.metrics[-1]
            last_base_test_metrics = last_instance_base_metrics.test_metrics[-1]
            last_target_test_metrics = get_last_test_metric_from_metric(last_instance_target_metrics, last_base_test_metrics.name)
            if last_target_test_metrics is not None:
                bar_chart_for_compare_avg_p90_p95(last_base_test_metrics, last_target_test_metrics, base_k6_test, base_cycle, target_test_cycle, path)
                bar_chart_for_compare_vus_rps_error(last_instance_base_metrics, last_instance_target_metrics, base_k6_test, base_cycle, target_test_cycle, path)
                bar_chart_for_compare_rps_cpu_memory(last_instance_base_metrics, last_instance_target_metrics, base_k6_test, base_cycle, target_test_cycle, path)
                bar_chart_for_compare_vus_cpu_memory(last_instance_base_metrics, last_instance_target_metrics, base_k6_test, base_cycle, target_test_cycle, path)


def bar_chart_for_compare_rps_cpu_memory(last_instance_base_metrics, last_instance_target_metrics, base_k6_test, base_cycle, target_test_cycle, path):
    """
    This function generates the bar chart for the comparison of RPS, CPU and Memory metrics.
    """
    perf_metrics = [cycle_label(base_cycle.name), cycle_label(target_test_cycle.name)]
    rates_metrics = [float(f2(last_instance_base_metrics.rates)), float(f2(last_instance_target_metrics.rates))]
    cpu_metrics = [float(f3(last_instance_base_metrics.cpu)), float(f3(last_instance_target_metrics.cpu))]
    memory_metrics = [float(f2(last_instance_base_metrics.memory)), float(f2(last_instance_target_metrics.memory))]
    bar_positions = np.arange(len(perf_metrics))
    bar_width = 0.2
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    bars_one = axs[0].bar(bar_positions - bar_width, rates_metrics, bar_width, label='RPS', color='#00ff00')
    bars_two = axs[0].bar(bar_positions + bar_width, cpu_metrics, bar_width, label='CPU (mills)', color='#0080ff')
    axs[0].set_xlabel('RPS and CPU Metrics')
    axs[0].set_ylabel('Values')
    axs[0].set_xticks(bar_positions)
    axs[0].set_xticklabels(perf_metrics)
    axs[0].set_title(base_k6_test.name + '\n' + cycle_label(base_cycle.name) + ' vs ' + cycle_label(target_test_cycle.name))
    axs[0].legend()
    calculate_and_set_bar_label_on_axs(axs[0], bars_one)
    calculate_and_set_bar_label_on_axs(axs[0], bars_two)
    bars_three = axs[1].bar(bar_positions - bar_width, rates_metrics, bar_width, label='RPS', color='#00ff00')
    bars_four = axs[1].bar(bar_positions + bar_width, memory_metrics, bar_width, label='Memory (MB)', color='#00bfff')
    axs[1].set_xlabel('RPS and Memory Metrics')
    axs[1].set_ylabel('Values')
    axs[1].set_xticks(bar_positions)
    axs[1].set_xticklabels(perf_metrics)
    axs[1].set_title(base_k6_test.name + '\n' + cycle_label(base_cycle.name) + ' vs ' + cycle_label(target_test_cycle.name))
    axs[1].legend()
    calculate_and_set_bar_label_on_axs(axs[1], bars_three)
    calculate_and_set_bar_label_on_axs(axs[1], bars_four)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, left=0.10, right=0.90)
    plt.savefig(path + '/' + target_test_cycle.name + '/' + chart_dir_name + '/' + base_k6_test.name + '/rps_cpu_memory_compare_' + base_cycle.name + '_vs_' + target_test_cycle.name + '.png')
    plt.clf()


def bar_chart_for_compare_vus_cpu_memory(last_instance_base_metrics, last_instance_target_metrics, base_k6_test, base_cycle, target_test_cycle, path):
    """
    This function generates the bar chart for the comparison of VUs, CPU and Memory metrics.
    """
    perf_metrics = [cycle_label(base_cycle.name), cycle_label(target_test_cycle.name)]
    vus_metrics = [last_instance_base_metrics.vus, last_instance_target_metrics.vus]
    cpu_metrics = [float(f3(last_instance_base_metrics.cpu)), float(f3(last_instance_target_metrics.cpu))]
    memory_metrics = [float(f2(last_instance_base_metrics.memory)), float(f2(last_instance_target_metrics.memory))]
    bar_positions = np.arange(len(perf_metrics))
    bar_width = 0.2
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    bars_one = axs[0].bar(bar_positions - bar_width, vus_metrics, bar_width, label='VUs', color='#00ff00')
    bars_two = axs[0].bar(bar_positions + bar_width, cpu_metrics, bar_width, label='CPU (mills)', color='#0080ff')
    axs[0].set_xlabel('RPS and CPU Metrics')
    axs[0].set_ylabel('Values')
    axs[0].set_xticks(bar_positions)
    axs[0].set_xticklabels(perf_metrics)
    axs[0].set_title(base_k6_test.name + '\n' + cycle_label(base_cycle.name) + ' vs ' + cycle_label(target_test_cycle.name))
    axs[0].legend()
    calculate_and_set_bar_label_on_axs(axs[0], bars_one)
    calculate_and_set_bar_label_on_axs(axs[0], bars_two)
    bars_three = axs[1].bar(bar_positions - bar_width, vus_metrics, bar_width, label='VUs', color='#00ff00')
    bars_four = axs[1].bar(bar_positions + bar_width, memory_metrics, bar_width, label='Memory (MB)', color='#00bfff')
    axs[1].set_xlabel('RPS and Memory Metrics')
    axs[1].set_ylabel('Values')
    axs[1].set_xticks(bar_positions)
    axs[1].set_xticklabels(perf_metrics)
    axs[1].set_title(base_k6_test.name + '\n' + cycle_label(base_cycle.name) + ' vs ' + cycle_label(target_test_cycle.name))
    axs[1].legend()
    calculate_and_set_bar_label_on_axs(axs[1], bars_three)
    calculate_and_set_bar_label_on_axs(axs[1], bars_four)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, left=0.10, right=0.90)
    plt.savefig(path + '/' + target_test_cycle.name + '/' + chart_dir_name + '/' + base_k6_test.name + '/vus_cpu_memory_compare_' + base_cycle.name + '_vs_' + target_test_cycle.name + '.png')
    plt.clf()


def bar_chart_for_compare_avg_p90_p95(last_base_test_metrics, last_target_test_metrics, base_k6_test, base_cycle, target_test_cycle, path):
    """
    This function generates the bar chart for the comparison of AVG, P90 and P95 metrics.
    """
    perf_metrics = [cycle_label(base_cycle.name), cycle_label(target_test_cycle.name)]
    avg_metrics = [float(f2(last_base_test_metrics.avg)), float(f2(last_target_test_metrics.avg))]
    p90_metrics = [float(f2(last_base_test_metrics.p90)), float(f2(last_target_test_metrics.p90))]
    p95_metrics = [float(f2(last_base_test_metrics.p95)), float(f2(last_target_test_metrics.p95))]
    bar_positions = np.arange(len(perf_metrics))
    bar_width = 0.1
    bars_avg = plt.bar(bar_positions - bar_width, avg_metrics, bar_width, label='AVG', color='#00ff00')
    bars_p90 = plt.bar(bar_positions, p90_metrics, bar_width, label='P90', color='#0080ff')
    bars_p95 = plt.bar(bar_positions + bar_width, p95_metrics, bar_width, label='P95', color='#00bfff')
    calculate_and_set_bar_label(bars_avg)
    calculate_and_set_bar_label(bars_p90)
    calculate_and_set_bar_label(bars_p95)
    plt.xlabel('AVG, P90 and P95 Metrics')
    plt.ylabel('Values in Mills')
    plt.title(base_k6_test.name + '\n' + cycle_label(base_cycle.name) + ' vs ' + cycle_label(target_test_cycle.name))
    plt.xticks(bar_positions, perf_metrics)
    plt.grid(color = 'green', linestyle = '--', linewidth = 0.1)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path + '/' + target_test_cycle.name + '/' + chart_dir_name + '/' + base_k6_test.name + '/avg_p90_p95_compare_' + base_cycle.name + '_vs_' + target_test_cycle.name + '.png')
    plt.clf()


def bar_chart_for_compare_vus_rps_error(last_instance_base_metrics, last_instance_target_metrics, base_k6_test, base_cycle, target_test_cycle, path):
    """
    This function generates the bar chart for the comparison of VUs, RPS and Errors metrics.
    """
    perf_metrics = [cycle_label(base_cycle.name), cycle_label(target_test_cycle.name)]
    vus_metrics = [last_instance_base_metrics.vus, last_instance_target_metrics.vus]
    rates_metrics = [float(f2(last_instance_base_metrics.rates)), float(f2(last_instance_target_metrics.rates))]
    error_metrics = [last_instance_base_metrics.errors, last_instance_target_metrics.errors]
    bar_positions = np.arange(len(perf_metrics))
    bar_width = 0.1
    bars_vus = plt.bar(bar_positions - bar_width, vus_metrics, bar_width, label='VUs', color='#00ff00')
    bars_rps = plt.bar(bar_positions, rates_metrics, bar_width, label='RPS', color='#0080ff')
    bars_error_count = plt.bar(bar_positions + bar_width, error_metrics, bar_width, label='Error Count', color='#00bfff')
    calculate_and_set_bar_label(bars_vus)
    calculate_and_set_bar_label(bars_rps)
    calculate_and_set_bar_label(bars_error_count)
    plt.xlabel('VUs, RPS and Error Count')
    plt.ylabel('Values')
    plt.title(base_k6_test.name + '\n' + cycle_label(base_cycle.name) + ' vs ' + cycle_label(target_test_cycle.name))
    plt.xticks(bar_positions, perf_metrics)
    plt.grid(color = 'green', linestyle = '--', linewidth = 0.1)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path + '/' + target_test_cycle.name + '/' + chart_dir_name + '/' + base_k6_test.name + '/vus_rps_error_count_compare_' + base_cycle.name + '_vs_' + target_test_cycle.name + '.png')
    plt.clf()


def calculate_and_set_bar_label(bars):
    for bar in bars:
        y_val = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, y_val, y_val, va='bottom')


def calculate_and_set_bar_label_on_axs(asx, bars):
    for bar in bars:
        y_val = bar.get_height()
        asx.text(bar.get_x() + bar.get_width()/2.0, y_val, y_val, va='bottom')


def get_test_scenario_from_k6_test_cycle(test_cycle, test_scenario_name):
    """
    This function returns the test scenario from the provided test cycle.
    """
    test_data = None
    for k6_test in test_cycle.metrics:
        if k6_test.name == test_scenario_name:
            test_data = k6_test
    return test_data


def get_last_test_metric_from_metric(metric, metric_name):
    """
    This function returns the test scenario from the provided test cycle.
    """
    test_data = None
    for test_metric in metric.test_metrics:
        if test_metric.name == metric_name:
            test_data = test_metric
    return test_data


def metric_and_test_metric_from_k6_test_cycle(test_cycle, target_metric_name, target_metric_index, target_test_metric_name):
    """
    This function returns the metric and test metric from the provided test cycle.
    """
    for k6_test in test_cycle.metrics:
        if k6_test.name == target_metric_name:
            for metric_index, metric in enumerate(k6_test.metrics):
                if metric_index == target_metric_index:
                    for test_metric in metric.test_metrics:
                        if test_metric.name == target_test_metric_name:
                            return metric, test_metric


def transform_cycle_to_table(test_cycle, path, test_dir_name):
    """
    This function transforms the test cycle to table.
    """
    table = table_css
    table += '<table border="1">'
    table += '<tr>'
    table += f'<th colspan="11">{cycle_label(test_cycle.name)}</th>'
    table += '</tr>'
    table += '<tr>'
    table += '<th>Test Scenario</th>'
    table += '<th>Requests</th>'
    table += '<th>VUs</th>'
    table += '<th>RPS</th>'
    table += '<th>Errors</th>'
    table += '<th>CPU  (mills)</th>'
    table += '<th>Memory  (MB)</th>'
    table += '<th>Metric Name</th>'
    table += '<th>AVG (mills)</th>'
    table += '<th>P90 (mills)</th>'
    table += '<th>P95 (mills)</th>'
    table += '</tr>'
    for k6_test in test_cycle.metrics:
        name_span = len(k6_test.metrics) * len(k6_test.metrics[0].test_metrics)
        metric_span = len(k6_test.metrics[0].test_metrics)
        for metric_index, metric in enumerate(k6_test.metrics):
            for test_metric_index, test_metric in enumerate(metric.test_metrics):
                if metric_index == 0 and test_metric_index == 0:
                    table += '<tr>'
                    table += f'<td rowspan="{name_span}">{k6_test.name}</td>'
                    table += f'<td rowspan="{metric_span}">{metric.reqs}</td>'
                    table += f'<td rowspan="{metric_span}">{metric.vus}</td>'
                    table += f'<td rowspan="{metric_span}">{f2(metric.rates)}</td>'
                    table += f'<td rowspan="{metric_span}">{metric.errors}</td>'
                    table += f'<td rowspan="{metric_span}">{f3(metric.cpu)}</td>'
                    table += f'<td rowspan="{metric_span}">{f2(metric.memory)}</td>'
                    table += f'<td>{metric_label(test_metric.name)}</td>'
                    table += f'<td>{f2(test_metric.avg)}</td>'
                    table += f'<td>{f2(test_metric.p90)}</td>'
                    table += f'<td>{f2(test_metric.p95)}</td>'
                    table += '</tr>'
                elif test_metric_index == 0:
                    table += '<tr>'
                    table += f'<td rowspan="{metric_span}">{metric.reqs}</td>'
                    table += f'<td rowspan="{metric_span}">{metric.vus}</td>'
                    table += f'<td rowspan="{metric_span}">{f2(metric.rates)}</td>'
                    table += f'<td rowspan="{metric_span}">{metric.errors}</td>'
                    table += f'<td rowspan="{metric_span}">{f3(metric.cpu)}</td>'
                    table += f'<td rowspan="{metric_span}">{f2(metric.memory)}</td>'
                    table += f'<td>{metric_label(test_metric.name)}</td>'
                    table += f'<td>{f2(test_metric.avg)}</td>'
                    table += f'<td>{f2(test_metric.p90)}</td>'
                    table += f'<td>{f2(test_metric.p95)}</td>'
                    table += '</tr>'
                else:
                    table += '<tr>'
                    table += f'<td>{metric_label(test_metric.name)}</td>'
                    table += f'<td>{f2(test_metric.avg)}</td>'
                    table += f'<td>{f2(test_metric.p90)}</td>'
                    table += f'<td>{f2(test_metric.p95)}</td>'
                    table += '</tr>'
    table += '</table>'
    write_file_on_disk(path + '/' + test_dir_name + chart_dir_name + '/summary-table.html', table)


def write_file_on_disk(target_file, content):
    """
    This function writes the content to the file on the provided path.
    """
    with open(target_file, 'w') as f:
        f.write(content)


def print_json_metrics(parent_directory, test_cycle_name):
    """
    This function iterates through defined test scenarios and process for each test scenario.
    """
    path = '/' + test_cycle_name
    k6_tests = []
    for test_name in test_and_scenario:
      cycle = process_test(test_name, test_and_scenario[test_name], parent_directory, path)
      k6_tests.append(cycle)
    return K6Cycle(test_cycle_name, k6_tests)


def process_test(test_name, metric_name_list, parent_directory, path):
    """
    This function processes the test for the provided test_name and metric_name_list.
    """
    print("Processing started for test scenario: " + test_name)
    list_of_json_files = list_files_for_the_test_from_dir(test_name, parent_directory, path)
    k6_test_obj = parse_json_and_get_k6_metrics(test_name, metric_name_list, list_of_json_files, parent_directory, path)
    generate_charts_for_test(k6_test_obj, parent_directory, path)
    print("Processing completed for test scenario: " + test_name)
    return k6_test_obj


def generate_charts_for_test(k6_test_obj, parent_directory, path):
    """
    This function generates the charts for the provided k6_test_obj.
    """
    path_to_save = parent_directory + path + chart_dir_name + '/' + k6_test_obj.name
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    else:
        print("Directory already exists: " + path_to_save)
    errors = []
    rates = []
    vus = []
    avg_http_metrics = {}
    p90_http_metrics = {}
    p95_http_metrics = {}
    for k6_metric in k6_test_obj.metrics:
        errors.append(k6_metric.errors)
        rates.append(k6_metric.rates)
        vus.append(k6_metric.vus)
        for metric in k6_metric.test_metrics:
            if metric.name in avg_http_metrics:
                avg_http_metrics[metric.name].append(metric.avg)
            else:
                avg_http_metrics[metric.name] = [metric.avg]
            if metric.name in p90_http_metrics:
                p90_http_metrics[metric.name].append(metric.avg)
            else:
                p90_http_metrics[metric.name] = [metric.avg]
            if metric.name in p95_http_metrics:
                p95_http_metrics[metric.name].append(metric.avg)
            else:
                p95_http_metrics[metric.name] = [metric.avg]
    plot_rate_vs_vus(rates, vus, path_to_save)
    plot_vus_vs_rate(vus, rates, path_to_save)
    plot_errors_vs_vus(vus, errors, path_to_save)
    plot_errors_vs_rps(rates, errors, path_to_save)
    plot_avg_response_time_vs_vus(vus, avg_http_metrics, path_to_save)
    plot_p90_response_time_vs_vus(vus, p90_http_metrics, path_to_save)
    plot_p95_response_time_vs_vus(vus, p95_http_metrics, path_to_save)
    plot_avg_response_time_vs_rps(rates, avg_http_metrics, path_to_save)
    plot_p90_response_time_vs_rps(rates, p90_http_metrics, path_to_save)
    plot_p95_response_time_vs_rps(rates, p95_http_metrics, path_to_save)


def plot_avg_response_time_vs_vus(vus, avg_http_metrics, path):
    """
    This function plots a scatter plot of response time vs vus and saves it to rate_vs_vu.png.
    """
    for key, value in avg_http_metrics.items():
        plt.scatter(vus, value, label=metrics_label_mapping[key])
    plt.xlabel('VUs')
    plt.ylabel('Response Time [ms]')
    plt.title('Avg Response Time vs VUs')
    plt.legend()
    plt.savefig(path + '/avg_response_time_vs_vu.png')
    plt.clf()


def plot_p90_response_time_vs_vus(vus, p90_http_metrics, path):
    """
    This function plots a scatter plot of response time vs vus and saves it to rate_vs_vu.png.
    """
    for key, value in p90_http_metrics.items():
        plt.scatter(vus, value, label=metrics_label_mapping[key])
    plt.xlabel('VUs')
    plt.ylabel('Response Time [ms]')
    plt.title('Response Time (P(90)) vs VUs')
    plt.legend()
    plt.savefig(path + '/p90_response_time_vs_vu.png')
    plt.clf()


def plot_p95_response_time_vs_vus(vus, p95_http_metrics, path):
    """
    This function plots a scatter plot of response time vs vus and saves it to rate_vs_vu.png.
    """
    for key, value in p95_http_metrics.items():
        plt.scatter(vus, value, label=metrics_label_mapping[key])
    plt.xlabel('VUs')
    plt.ylabel('Response Time [ms]')
    plt.title('Response Time (P(95)) vs VUs')
    plt.legend()
    plt.savefig(path + '/p95_response_time_vs_vu.png')
    plt.clf()


def plot_avg_response_time_vs_rps(rates, avg_http_metrics, path):
    """
    This function plots a scatter plot of response time vs vus and saves it to avg_response_time_vs_rps.png.
    """
    for key, value in avg_http_metrics.items():
        plt.scatter(rates, value, label=metrics_label_mapping[key])
    plt.xlabel('Requests per second')
    plt.ylabel('Response Time [ms]')
    plt.title('Avg Response Time vs Requests per second')
    plt.legend()
    plt.savefig(path + '/avg_response_time_vs_rps.png')
    plt.clf()


def plot_p90_response_time_vs_rps(rates, p90_http_metrics, path):
    """
    This function plots a scatter plot of response time vs vus and saves it to p90_response_time_vs_rps.png.
    """
    for key, value in p90_http_metrics.items():
        plt.scatter(rates, value, label=metrics_label_mapping[key])
    plt.xlabel('Requests per second')
    plt.ylabel('Response Time (P(90)) [ms]')
    plt.title('Response Time (P(90)) vs Requests per second')
    plt.legend()
    plt.savefig(path + '/p90_response_time_vs_rps.png')
    plt.clf()


def plot_p95_response_time_vs_rps(rates, p95_http_metrics, path):
    """
    This function plots a scatter plot of response time vs vus and saves it to p95_response_time_vs_rps.png.
    """
    for key, value in p95_http_metrics.items():
        plt.scatter(rates, value, label=metrics_label_mapping[key])
    plt.xlabel('Requests per second')
    plt.ylabel('Response Time (P(95)) [ms]')
    plt.title('Response Time (P(95)) vs Requests per second')
    plt.legend()
    plt.savefig(path + '/p95_response_time_vs_rps.png')
    plt.clf()


def plot_rate_vs_vus(rates, vus, path):
    """
    This function plots a scatter plot of rate vs vus and saves it to rate_vs_vu.png.
    """
    plt.scatter(rates, vus, label='VUs')
    plt.ylabel('VUs')
    plt.xlabel('Requests per second')
    plt.title('Request per second vs VUs')
    plt.legend()
    plt.savefig(path + '/rps_vs_vu.png')
    plt.clf()


def plot_vus_vs_rate(vus, rates, path):
    """
    This function plots a scatter plot of rate vs vus and saves it to vu_vs_rate.png.
    """
    plt.scatter(vus, rates, label='RPS')
    plt.ylabel('RPS')
    plt.xlabel('VUs')
    plt.title('VUs vs Request per second')
    plt.legend()
    plt.savefig(path + '/vu_vs_rate.png')
    plt.clf()


def plot_errors_vs_vus(vus, errors, path):
    """
    This function plots a scatter plot of errors vs vus and saves it to errors_vs_vu.png.
    """
    plt.scatter(vus, errors, label='Errors')
    plt.xlabel('VUs')
    plt.ylabel('Number of errors')
    plt.title('Number of errors vs VUs')
    plt.legend()
    plt.savefig(path + '/errors_vs_vu.png')
    plt.clf()


def plot_errors_vs_rps(rates, errors, path):
    """
    This function plots a scatter plot of errors vs vus and saves it to errors_vs_rps.png.
    """
    plt.scatter(rates, errors, label='Errors')
    plt.xlabel('Requests per second')
    plt.ylabel('Number of errors')
    plt.title('Number of errors vs Requests per second')
    plt.legend()
    plt.savefig(path + '/errors_vs_rps.png')
    plt.clf()


def list_files_for_the_test_from_dir(test_name, parent_directory, path):
    """
    This function returns the list of files from the directory for the provided test_name.
    """
    test_file_prefix = test_json_file_name_prefix + test_name + '-'
    directory = parent_directory + path
    return [f for f in os.listdir(directory) if f.endswith(".json") and f.startswith(test_file_prefix)]


def parse_json_and_get_k6_metrics(test_name, metric_name_list, list_of_json_files, parent_directory, path):
    """
    This function reads the json file and return the metrics object.
    """
    k6_metrics = []
    for json_file_name in list_of_json_files:
        cpu, memory = obtain_cpu_memory_usage(json_file_name, parent_directory, path)
        with open(os.path.join(parent_directory + path, json_file_name), "r") as f:
            content = f.read()
            data = json.loads(content)
            k6_metric_obj = K6Metric()
            k6_metric_obj.cpu = cpu
            k6_metric_obj.memory = memory
            k6_metric_obj.reqs = data['metrics']['http_reqs']['count']
            k6_metric_obj.rates = data['metrics']['http_reqs']['rate']
            k6_metric_obj.vus = data['metrics']['vus']['max']
            k6_metric_obj.errors = data['metrics']['error_rate']['passes']
            k6_metric_obj_test_metrics = []
            for metric_name in metric_name_list:
                test_metric_obj = TestMetric()
                test_metric_obj.avg = data['metrics'][metric_name]['avg']
                test_metric_obj.p90 = data['metrics'][metric_name]['p(90)']
                test_metric_obj.p95 = data['metrics'][metric_name]['p(95)']
                test_metric_obj.name = metric_name
                test_metric_obj.test_metrics = []
                k6_metric_obj_test_metrics.append(test_metric_obj)
            k6_metrics.append(k6_metric_obj)
            k6_metric_obj.test_metrics = k6_metric_obj_test_metrics
    return K6Test(test_name, k6_metrics)


def f2(num):
    return "{:.2f}".format(num)


def f3(num):
    return "{:.3f}".format(num)


def cycle_label(cycle_name):
    value = cycle_label_mapping.get(cycle_name)
    if value is not None:
        return value
    else:
        return cycle_name


def metric_label(metric_name):
    value = metrics_label_mapping.get(metric_name)
    if value is not None:
        return value
    else:
        return metric_name


def obtain_cpu_memory_usage(test_file_name, parent_directory, path):
    """
    This function returns the cpu and memory usage for provided test file.
    """
    try:
        df = pd.read_csv(parent_directory + path + test_cpu_memory_resource_file_name)
        cpu, memory = 0.0, 0.0
        test_name_entry = test_file_name.removeprefix(test_json_file_name_prefix).removesuffix('.json')
        for index, row in df.iterrows():
            if ('-' + test_name_entry) in row.iloc[0]:
                cpu = row.iloc[1]
                memory = row.iloc[2]
                break
        return cpu, memory
    except Exception as e:
        print("Error while resource reading file: " + test_cpu_memory_resource_file_name)
        print(e)
        return 0.0, 0.0


class K6Metric:
    def __init__(self, reqs=0, rates=0.0, errors=0.0, vus=0, cpu=0.0, memory=0.0, test_metrics=None):
        self.reqs = reqs
        self.rates = rates
        self.errors = errors
        self.vus = vus
        self.cpu = cpu
        self.memory = memory
        self.test_metrics = test_metrics if test_metrics is not None else []


class TestMetric:
    def __init__(self, avg=0.0, p90=0.0, p95=0.0, name=""):
        self.avg = avg
        self.p90 = p90
        self.p95 = p95
        self.name = name


class K6Test:
    def __init__(self, name="", metrics=None):
        self.name = name
        self.metrics = metrics if metrics is not None else []


class K6Cycle:
    def __init__(self, name="", cycles=None):
        self.name = name
        self.metrics = cycles if cycles is not None else []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('directory', type=str, help='The directory to process')
    args = parser.parse_args()
    iterate_over_dirs(args.directory)
