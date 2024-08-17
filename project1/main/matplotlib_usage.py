import matplotlib.pyplot as plt
import numpy as np

"""
Matplotlib is a low level graph plotting library in python that serves as a visualization utility.
"""


def metplotlib_usage():
    # plot_line_by_start_and_end_points()
    # plot_dot_by_start_and_end_points()
    # plot_line_by_multi_points()
    # plot_dot_by_multi_points()
    # plot_line_by_y_axis_only()
    # plot_line_by_y_axis_only_with_marker()
    # plot_line_by_y_axis_only_with_marker('*')
    # plot_line_by_y_axis_only_with_marker('.')
    # plot_line_by_y_axis_only_with_marker(',')
    # plot_line_by_y_axis_only_with_marker('x')
    # plot_line_by_y_axis_only_with_marker('X')
    # plot_line_by_y_axis_only_with_marker('+')
    # plot_line_by_y_axis_only_with_marker('P')
    # plot_line_by_y_axis_only_with_marker('s')
    # plot_line_by_y_axis_only_with_marker('D')
    # plot_line_by_y_axis_only_with_marker('d')
    # plot_line_by_y_axis_only_with_marker('p')
    # plot_line_by_y_axis_only_with_marker('H')
    # plot_line_by_y_axis_only_with_marker('h')
    # plot_line_by_y_axis_only_with_marker('v')
    # plot_line_by_y_axis_only_with_marker('^')
    # plot_line_by_y_axis_only_with_marker('>')
    # plot_line_by_y_axis_only_with_marker('<')
    # plot_line_by_y_axis_only_with_marker('1')
    # plot_line_by_y_axis_only_with_marker('2')
    # plot_line_by_y_axis_only_with_marker('3')
    # plot_line_by_y_axis_only_with_marker('4')
    # plot_line_by_y_axis_only_with_marker('|')
    # plot_line_by_y_axis_only_with_marker('_')
    # plot_line_by_y_axis_only_with_format()
    # plot_line_by_y_axis_only_with_format('o:r')  # fromat_str = marker|line|color, bullet, dotted line with red color
    # plot_line_by_y_axis_only_with_format('o-r')  # bullet, solid line with red color
    # plot_line_by_y_axis_only_with_format('o--r')  # bullet, dashed line with red color
    # plot_line_by_y_axis_only_with_format('o-.r')  # bullet, dashed-dot line with red color
    # plot_2_lines()
    # sub_plot()
    # scatter_plot()
    # scatter_plot_with_color()
    # scatter_plot_with_color_and_size()
    # bar_chart_1()
    # histogram_plot()
    # pie_chart_simple()
    pie_chart_explode()
    pass


def pie_chart_explode():
    y = np.array([35, 25, 25, 15])
    my_colors = ["black", "hotpink", "b", "#4CAF50"]
    my_labels = ["Apples", "Bananas", "Cherries", "Dates"]
    my_explode = [0.3, 0, 0, 0.2]
    plt.pie(y, labels=my_labels, explode=my_explode, colors=my_colors, shadow=True)
    plt.legend(title="Four Fruits", loc="upper left")
    plt.show()


def pie_chart_simple():
    y = np.array([35, 25, 25, 15])
    my_labels = ["Apples", "Bananas", "Cherries", "Dates"]
    plt.pie(y, labels=my_labels, startangle=90)
    plt.show()


def histogram_plot():
    x = np.random.normal(170, 10, 250)
    plt.hist(x)
    plt.show()


def bar_chart_1():
    x = np.array(["A", "B", "C", "D"])
    y = np.array([3, 8, 1, 10])
    plt.barh(x, y, color='red', height=0.1)  # horizontal bar chart
    plt.bar(x, y, color='blue', width=0.1)  # vertical bar chart
    plt.show()


def scatter_plot_with_color_and_size():
    x = np.random.randint(100, size=(100))
    y = np.random.randint(100, size=(100))
    colors = np.random.randint(100, size=(100))
    sizes = 10 * np.random.randint(100, size=(100))
    plt.scatter(x, y, c=colors, s=sizes, alpha=0.5, cmap='nipy_spectral')
    plt.colorbar()
    plt.savefig("c:/sandeep/plot.png")  # save plot to file
    plt.show()


def scatter_plot_with_color():
    x = np.array([5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6])
    y = np.array([99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86])
    colors = np.array([0, 10, 20, 30, 40, 45, 50, 55, 60, 70, 80, 90, 100])
    plt.scatter(x, y, c=colors, cmap='viridis')
    plt.colorbar()
    plt.show()


def scatter_plot():
    x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
    y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
    plt.scatter(x, y, color = 'hotpink')
    x = np.array([2,2,8,1,15,8,12,9,7,3,11,4,7,14,12])
    y = np.array([100,105,84,105,90,99,90,95,94,100,79,112,91,80,85])
    plt.scatter(x, y, color = '#88c999')
    plt.show()


def sub_plot():
    x = np.array([0, 1, 2, 3])
    y = np.array([3, 8, 1, 10])
    plt.subplot(2, 3, 1)
    plt.plot(x, y)
    plt.title("Chart 1")
    x = np.array([0, 1, 2, 3])
    y = np.array([10, 20, 30, 40])
    plt.subplot(2, 3, 2)
    plt.plot(x, y)
    plt.title("Chart 2")
    x = np.array([0, 1, 2, 3])
    y = np.array([3, 8, 1, 10])
    plt.subplot(2, 3, 3)
    plt.plot(x, y)
    plt.title("Chart 3")
    x = np.array([0, 1, 2, 3])
    y = np.array([10, 20, 30, 40])
    plt.subplot(2, 3, 4)
    plt.plot(x, y)
    plt.title("Chart 4")
    x = np.array([0, 1, 2, 3])
    y = np.array([3, 8, 1, 10])
    plt.subplot(2, 3, 5)
    plt.plot(x, y)
    plt.title("Chart 5")
    x = np.array([0, 1, 2, 3])
    y = np.array([10, 20, 30, 40])
    plt.subplot(2, 3, 6)
    plt.plot(x, y)
    plt.title("Chart 6")
    plt.suptitle("Charts")
    plt.show()


def plot_2_lines():
    font1 = {'family':'serif','color':'blue','size':20}
    font2 = {'family':'serif','color':'darkred','size':15}
    x1 = np.array([0, 1, 2, 3])
    y1 = np.array([3, 8, 1, 10])
    x2 = np.array([0, 1, 2, 3])
    y2 = np.array([6, 2, 7, 11])
    plt.plot(x1, y1, x2, y2)
    plt.xlabel("Average Pulse", fontdict=font2)  # x-axis label
    plt.ylabel("Calorie Burnage", fontdict=font2)  # y-axis label
    plt.title("Sports Watch Data", fontdict=font1, loc='left')  # title
    plt.grid(color='green', linestyle='--', linewidth=0.5)  # grid
    # plt.grid(axis='x')  # grid
    plt.show()


def plot_line_by_y_axis_only_with_format(format_str='o'):
    y_points = np.array([3, 8, 1, 10, 5, 7])
    plt.plot(y_points, format_str)
    plt.show()


def plot_line_by_y_axis_only_with_marker(marker='o'):
    y_points = np.array([3, 8, 1, 10, 5, 7])
    plt.plot(y_points, marker=marker)  # apart from marker, we can use 'linestyle', 'color', 'label', 'marker size'
    plt.show()


def plot_line_by_y_axis_only():
    y_points = np.array([3, 8, 1, 10, 5, 7])
    plt.plot(y_points)
    plt.show()


def plot_dot_by_multi_points():
    x_points = np.array([1, 2, 6, 8])
    y_points = np.array([3, 8, 1, 10])
    plt.plot(x_points, y_points, 'o')
    plt.show()


def plot_line_by_multi_points():
    x_points = np.array([1, 2, 6, 8])
    y_points = np.array([3, 8, 1, 10])
    plt.plot(x_points, y_points)
    plt.show()


def plot_dot_by_start_and_end_points():
    x_points = np.array([1, 8])
    y_points = np.array([3, 10])
    plt.plot(x_points, y_points, 'o')
    plt.show()


def plot_line_by_start_and_end_points():
    x_points = np.array([1, 8])
    y_points = np.array([3, 10])
    plt.plot(x_points, y_points)
    plt.show()


if __name__ == '__main__':
    metplotlib_usage()
