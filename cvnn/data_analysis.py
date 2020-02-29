import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import plotly
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
import seaborn as sns
import pandas as pd
import numpy as np
import glob
import re
import os
from pathlib import Path
from pdb import set_trace
import scipy.stats as stats
from cvnn.utils import create_folder

DEFAULT_PLOTLY_COLORS = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
                         'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
                         'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
                         'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
                         'rgb(188, 189, 34)', 'rgb(23, 190, 207)']


def add_transparency(color='rgb(31, 119, 180)', alpha=0.5):
    pattern = re.compile("^rgb\([0-9]+, [0-9]+, [0-9]+\)$")
    assert re.match(pattern, color)    # Recognized color format!
    color = re.sub("^rgb", "rgba", color)
    color = re.sub("\)$", ", {})".format(alpha), color)
    return color


def extract_values(color='rgb(31, 119, 180)'):
    pattern = re.compile("^rgb\([0-9]+, [0-9]+, [0-9]+\)$")
    assert re.match(pattern, color)  # Recognized color format!
    return [float(s) for s in re.findall(r'\b\d+\b', color)]


def find_intersection_of_gaussians(m1, m2, std1, std2):
    a = 1 / (2 * std1 ** 2) - 1 / (2 * std2 ** 2)
    b = m2 / (std2 ** 2) - m1 / (std1 ** 2)
    c = m1 ** 2 / (2 * std1 ** 2) - m2 ** 2 / (2 * std2 ** 2) - np.log(std2 / std1)
    return np.roots([a, b, c])


def add_params(fig, ax, y_label=None, x_label=None, loc=None, title=None,
               filename="./results/plot_2_gaussian_output.png", showfig=False, savefig=True):
    """
    :param fig:
    :param ax:
    :param y_label: The y axis label.
    :param x_label: The x axis label.
    :param loc: can be a string or an integer specifying the legend location. default: None.
                    https://matplotlib.org/api/legend_api.html#matplotlib.legend.Legend
    :param title: str or None. The legend’s title. Default is no title (None).
    :param filename: Only used when savefig=True. The name of the figure to be saved
    :param showfig: Boolean. If true it will show the figure using matplotlib show method
    :param savefig: Boolean. If true it will save the figure with the name of filename parameter
    :return None:
    """
    # Figure parameters
    if loc is not None:
        fig.legend(loc=loc)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if title is not None:
        ax.set_title(title)
    # save/show results
    if showfig:
        fig.show()
    if savefig:
        fig.savefig(filename)


def get_trailing_number(s):
    """
    Search for a termination of a file name that has the ".csv" extension and that has a number at the end.
    It gives the number at the end of the file. This number can have any amount of digits.
    Example:
    x = get_trailing_number("my/path/to/file/any_43_start_name9872.csv")    # x = 9872
    y = get_trailing_number("my/path/to/file/any_43_start_name.csv")        # y = None
    z = get_trailing_number("my/path/to/file/any_43_start_name85498.txt")   # y = None
    :param s: The string to search for the specific term
    :return: The number located before the extension. None if there is no number.
    """
    m = re.search(r'\d+.csv$', s)  # I get only the end of the string (last number of any size and .csv extension)
    # splitext gets root [0] and extension [1] of the name.
    return int(os.path.splitext(m.group())[0]) if m else None


"""
Confusion Matrix
----------------
Given results and labels directly
"""


def plot_confusion_matrix(data, filename=None, library='plotly', axis_legends=None, showfig=False):
    if library == 'seaborn':
        fig, ax = plt.subplots()
        sns.heatmap(data,
                    annot=True,
                    linewidths=.5,
                    cbar=True,
                    )
        if filename is not None:
            fig.savefig(filename)
    elif library == 'plotly':
        z = data.values.tolist()
        if axis_legends is None:
            y = [str(j) for j in data.axes[0].tolist()]
            x = [str(i) for i in data.axes[1].tolist()]
        else:
            y = []
            x = []
            for j in data.axes[0].tolist():
                if isinstance(j, int):
                    y.append(axis_legends[j])
                elif isinstance(j, str):
                    y.append(j)
                else:
                    print("WTF?! should never have arrived here")
            for i in data.axes[1].tolist():
                if isinstance(i, int):
                    x.append(axis_legends[i])
                elif isinstance(i, str):
                    x.append(i)
                else:
                    print("WTF?! should never have arrived here")
        # fig = go.Figure(data=go.Heatmap(z=z, x=x, y=y))
        fig = ff.create_annotated_heatmap(z, x=x, y=y)
    if showfig:
        fig.show()


def sparse_confusion_matrix(y_pred_np, y_label_np, filename=None, axis_legends=None):
    y_pred_pd = pd.Series(y_pred_np, name='Predicted')
    y_label_pd = pd.Series(y_label_np, name='Actual')
    df = pd.crosstab(y_label_pd, y_pred_pd, rownames=['Actual'], colnames=['Predicted'], margins=True)
    plot_confusion_matrix(df, filename, library='plotly', axis_legends=axis_legends)
    return df


def categorical_confusion_matrix(y_pred_np, y_label_np, filename=None, axis_legends=None):
    return sparse_confusion_matrix(np.argmax(y_pred_np, axis=1), np.argmax(y_label_np, axis=1), filename, axis_legends)


class Plotter:

    def __init__(self, path, file_suffix=".csv"):
        assert os.path.exists(path)
        self.path = Path(path)
        self.pandas_list = []
        self.labels = []
        self.file_suffix = file_suffix
        self._csv_to_pandas()

    def _csv_to_pandas(self):
        self.pandas_list = []
        self.labels = []
        for file in os.listdir(self.path):
            if file.endswith(self.file_suffix):
                self.pandas_list.append(pd.read_csv(self.path / file))
                self.labels.append(re.sub(self.file_suffix + '$', '', file))

    def reload_data(self):
        self._csv_to_pandas()

    def plot_everything(self, reload=False, library='plotly', showfig=False, savefig=True, index_loc=None):
        if reload:
            self._csv_to_pandas()
        assert len(self.pandas_list) != 0
        for key in self.pandas_list[0]:
            self.plot_key(key, reload=False, library=library, showfig=showfig, savefig=savefig, index_loc=index_loc)

    def plot_key(self, key='loss', reload=False, library='plotly', showfig=False, savefig=True, index_loc=None):
        if reload:
            self._csv_to_pandas()
        if library == 'matplotlib':
            self._plot_matplotlib(key=key, showfig=showfig, savefig=savefig, index_loc=index_loc)
        elif library == 'plotly':
            self._plot_plotly(key=key, showfig=showfig, savefig=savefig, index_loc=index_loc)
        else:
            print("Warning: Unrecognized library to plot " + library)

    def _plot_matplotlib(self, key='loss', showfig=False, savefig=True, index_loc=None):
        fig, ax = plt.subplots()
        title = None
        for i, data in enumerate(self.pandas_list):
            if key in data:
                if title is not None:
                    title += " vs. " + self.labels[i]
                else:
                    title = self.labels[i]
                if index_loc is not None:
                    if 'stats' in data.keys():
                        data = data[data['stats'] == 'mean']
                    else:
                        print("Warning: Trying to index an array without index")
                ax.plot(data[key], 'o-', label=self.labels[i])
        title += " " + key
        fig.legend(loc="upper right")
        ax.set_ylabel(key)
        ax.set_xlabel("step")
        ax.set_title(title)
        if showfig:
            fig.show()
        if savefig:
            fig.savefig(str(self.path / key) + ".png")

    def _plot_plotly(self, key='loss', showfig=False, savefig=True, func=min, index_loc=None):
        fig = go.Figure()
        annotations = []
        title = ''
        for i, data in enumerate(self.pandas_list):
            if key in data:
                if title is not None:
                    title += " vs. " + self.labels[i]
                else:
                    title = self.labels[i]
                if index_loc is not None:
                    if 'stats' in data.keys():
                        data = data[data['stats'] == 'mean']
                    else:
                        print("Warning: Trying to index an array without index")
                x = list(range(len(data[key])))
                fig.add_trace(go.Scatter(x=x, y=data[key], mode='lines', name=self.labels[i],
                                         line_color=DEFAULT_PLOTLY_COLORS[i]))
                # Add points
                fig.add_trace(go.Scatter(x=[x[-1]],
                                         y=[data[key].to_list()[-1]],
                                         mode='markers',
                                         name='last value',
                                         marker_color=DEFAULT_PLOTLY_COLORS[i]))
                # Max/min points
                func_value = func(data[key])
                # ATTENTION! this will only give you first occurrence
                func_index = data[key].to_list().index(func_value)
                if func_index != len(data[key]) - 1:
                    fig.add_trace(go.Scatter(x=[func_index],
                                             y=[func_value],
                                             mode='markers',
                                             name=func.__name__,
                                             text=['{0:.2f}%'.format(func_value)],
                                             textposition="top center",
                                             marker_color=DEFAULT_PLOTLY_COLORS[i]))
                    # Min annotations
                    annotations.append(dict(xref="x", yref="y", x=func_index, y=func_value,
                                            xanchor='left', yanchor='middle',
                                            text='{0:.2f}'.format(func_value),
                                            font=dict(family='Arial',
                                                      size=14),
                                            showarrow=False, ay=-40))
                # Right annotations
                annotations.append(dict(xref='paper', x=0.95, y=data[key].to_list()[-1],
                                        xanchor='left', yanchor='middle',
                                        text='{0:.2f}'.format(data[key].to_list()[-1]),
                                        font=dict(family='Arial',
                                                  size=16),
                                        showarrow=False))
        title += " " + key
        fig.update_layout(annotations=annotations,
                          title=title,
                          xaxis_title='steps',
                          yaxis_title=key)
        if savefig:
            plotly.offline.plot(fig, filename=str(self.path / key) + ".html")
        elif showfig:
            fig.show()

    def get_full_pandas_dataframe(self):
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html
        self._csv_to_pandas()
        length = len(self.pandas_list[0])
        for data_frame in self.pandas_list:  # TODO: Check if.
            assert length == len(data_frame)  # What happens if NaN? Can I cope not having same len?

        result = pd.DataFrame({
            'network': [self.get_net_name()] * length,
            'step': list(range(length)),
            'path': [self.path] * length
        })

        for data_frame, data_label in zip(self.pandas_list, self.labels):
            data_frame.columns = [data_label + " " + str(col) for col in data_frame.columns]
            # concatenated = pd.concat(self.pandas_list, keys=self.labels)
            result = pd.concat([result, data_frame], axis=1, sort=False)
        return result

    def get_net_name(self):
        str_to_match = "_metadata.txt"
        for file in os.listdir(self.path):
            if file.endswith(str_to_match):
                return re.sub(str_to_match + "$", '', file)  # See that there is no need to open the file
        return "Name not found"


class MonteCarloPlotter(Plotter):

    def __init__(self, path):
        file_suffix = "_statistical_result.csv"
        self.filter_keys = ['step', 'stats']
        super().__init__(path, file_suffix=file_suffix)

    def plot_everything(self, reload=False, library='plotly', showfig=False, savefig=True, index_loc='mean'):
        if reload:
            self._csv_to_pandas()
        assert len(self.pandas_list) != 0
        for key in self.pandas_list[0]:
            if key not in self.filter_keys:
                self.plot_key(key, reload=False, library=library, showfig=showfig, savefig=savefig, index_loc=index_loc)

    def plot_key(self, key='test accuracy', reload=False, library='plotly', showfig=False, savefig=True,
                 index_loc='mean'):
        super().plot_key(key, reload, library, showfig, savefig, index_loc)

    def plot_distribution(self, key='test accuracy', showfig=False, savefig=True):
        fig = go.Figure()
        for i, data in enumerate(self.pandas_list):
            x = data['step'].unique().tolist()
            x_rev = x[::-1]
            data_mean = data[data['stats'] == 'mean'][key].tolist()
            data_max = data[data['stats'] == 'max'][key].tolist()
            data_min = data[data['stats'] == 'min'][key][::-1].tolist()
            data_25 = data[data['stats'] == '25%'][key][::-1].tolist()
            data_75 = data[data['stats'] == '75%'][key].tolist()
            # set_trace()
            fig.add_trace(go.Scatter(
                x=x + x_rev,
                y=data_max + data_min,
                fill='toself',
                fillcolor=add_transparency(DEFAULT_PLOTLY_COLORS[i], 0.1),
                line_color=add_transparency(DEFAULT_PLOTLY_COLORS[i], 0),
                showlegend=True,
                name=self.labels[i] + " borders",
            ))
            fig.add_trace(go.Scatter(
                x=x + x_rev,
                y=data_75 + data_25,
                fill='toself',
                fillcolor=add_transparency(DEFAULT_PLOTLY_COLORS[i], 0.2),
                line_color=add_transparency(DEFAULT_PLOTLY_COLORS[i], 0),
                showlegend=True,
                name=self.labels[i] + " 75%",
            ))
            fig.add_trace(go.Scatter(
                x=x, y=data_mean,
                line_color=DEFAULT_PLOTLY_COLORS[i],
                name=self.labels[i],
            ))

        title = ''
        for label in self.labels:
            title += label.replace('_', ' ') + ' vs '
        title = title[:-3] + key

        fig.update_traces(mode='lines')
        fig.update_layout(title=title, xaxis_title='steps', yaxis_title=key)

        if savefig:
            plotly.offline.plot(fig, filename=str(self.path / key) + ".html")
        elif showfig:
            fig.show()


class MonteCarloAnalyzer:

    def __init__(self, df=None, path=None):
        if path is not None and df is not None:  # I have data and the place where I want to save it
            self.df = df  # DataFrame with all the data
            self.path = Path(path)
            self.df.to_csv(self.path / "run_data.csv")  # Save the results for latter use
        elif path is not None and df is None:  # Load df from Path
            if not path.endswith('.csv'):
                path += '.csv'
            self.df = pd.read_csv(path)
            self.path = Path(os.path.split(path)[0])  # Keep only the path and not the filename
        elif path is None and df is not None:  # Save df into default path
            self.path = create_folder("./montecarlo/")
            self.df = df  # DataFrame with all the data
            self.df.to_csv(self.path / "run_data.csv")  # Save the results for latter use
        else:  # I have nothing
            self.path = create_folder("./montecarlo/")
            self.df = pd.DataFrame()
        self.plotable_info = ['train loss', 'test loss', 'train accuracy', 'test accuracy']
        self.monte_carlo_plotter = MonteCarloPlotter(self.path)

    def set_df(self, df):
        self.df = df  # DataFrame with all the data
        self.df.to_csv(self.path / "run_data.csv")  # Save the results for latter use
        self.save_stat_results()
        self.monte_carlo_plotter.reload_data()

    def plot_everything_histogram(self, library='plotly', step=-1, showfig=False, savefig=True):
        for key in self.plotable_info[0]:
            self.plot_histogram(key, library=library, step=step, showfig=showfig, savefig=savefig)

    def plot_histogram(self, key='test accuracy', step=-1, library='plotly', showfig=False, savefig=True):
        if library == 'matplotlib':
            self._plot_histogram_matplotlib(key=key, step=step, showfig=showfig, savefig=savefig)
        elif library == 'plotly':
            self._plot_histogram_plotly(key=key, step=step, showfig=showfig, savefig=savefig)
        elif library == 'seaborn':
            self._plot_histogram_seaborn(key=key, step=step, showfig=showfig, savefig=savefig)
        else:
            print("Warning: Unrecognized library to plot " + library)
            return None

    def _plot_histogram_matplotlib(self, key='test accuracy', step=-1, showfig=False, savefig=True):
        fig, ax = plt.subplots()
        bins = np.linspace(0, 1, 501)
        min_ax = 1.0
        max_ax = 0.0
        networks_availables = self.df.network.unique()
        title = ''
        if step == -1:
            step = max(self.df.step)
        for net in networks_availables:
            filter = [a == net and b == step for a, b in zip(self.df.network, self.df.step)]
            data = self.df[filter]  # Get only the data to plot
            ax.hist(data[key], bins, alpha=0.5, label=net)
            min_ax = min(min_ax, min(data[key]))
            max_ax = max(max_ax, max(data[key]))
        title += key + " comparison"
        ax.axis(xmin=min_ax - 0.01, xmax=max_ax + 0.01)
        add_params(fig, ax, x_label=key, title=title, loc='upper right',
                   filename=self.path / (key + "_matplotlib.png"), showfig=showfig, savefig=savefig)
        return fig, ax

    def _plot_histogram_plotly(self, key='test accuracy', step=-1, showfig=False, savefig=True):
        networks_availables = self.df.network.unique()
        title = ''
        if step == -1:
            step = max(self.df.step)
        hist_data = []
        group_labels = []
        for net in networks_availables:
            title += net + ' '
            filter = [a == net and b == step for a, b in zip(self.df.network, self.df.step)]
            data = self.df[filter]  # Get only the data to plot
            hist_data.append(data[key].to_list())
            group_labels.append(net)
            # fig.add_trace(px.histogram(np.array(data[key]), marginal="box"))
            # fig.add_trace(go.Histogram(x=np.array(data[key]), name=net))
        fig = ff.create_distplot(hist_data, group_labels, bin_size=0.01)  # https://plot.ly/python/distplot/
        title += key + " comparison"

        # Overlay both histograms
        fig.update_layout(barmode='overlay')
        # Reduce opacity to see both histograms
        fig.update_traces(opacity=0.75)
        fig.update_layout(title=title.replace('_', ' '),
                          xaxis_title=key)
        if savefig:
            plotly.offline.plot(fig, filename=str(self.path / (key + "_histogram.html")))
        elif showfig:
            fig.show()
        return fig

    def _plot_histogram_seaborn(self, key='test accuracy', step=-1, showfig=False, savefig=True):
        fig = plt.figure()
        bins = np.linspace(0, 1, 501)
        min_ax = 1.0
        max_ax = 0.0
        ax = None
        networks_availables = self.df.network.unique()
        title = ''
        if step == -1:
            step = max(self.df.step)
        for net in networks_availables:
            filter = [a == net and b == step for a, b in zip(self.df.network, self.df.step)]
            data = self.df[filter]  # Get only the data to plot
            ax = sns.distplot(data[key], bins, label=net)
            min_ax = min(min_ax, min(data[key]))
            max_ax = max(max_ax, max(data[key]))
        title += " " + key
        ax.axis(xmin=min_ax - 0.01, xmax=max_ax + 0.01)
        add_params(fig, ax, x_label=key, title=title, loc='upper right',
                   filename=self.path / (key + "_seaborn.png"), showfig=showfig, savefig=savefig)
        return fig, ax

    def save_stat_results(self):
        # save csv file for each network with 4 columns
        networks_availables = self.df.network.unique()
        for net in networks_availables:
            data = self.df[self.df.network == net]
            cols = ['train loss', 'test loss', 'train accuracy', 'test accuracy']
            frames = []
            keys = []
            for step in data.step.unique():
                frames.append(data[data.step == step][cols].describe())
                keys.append(step)
            data_to_save = pd.concat(frames, keys=keys, names=['step', 'stats'])
            data_to_save.to_csv(self.path / (net + "_statistical_result.csv"))

    def show_plotly_table(self):
        values = [key for key in self.df.keys()]
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(self.df.columns),
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[self.df.values.tolist()],
                       fill_color='lavender',
                       align='left'))
        ])
        fig.show()

    def plot_3d_hist(self, steps=None, key='test accuracy'):
        # https://stackoverflow.com/questions/60398154/plotly-how-to-make-a-3d-stacked-histogram/60403270#60403270
        # https://plot.ly/python/v3/3d-filled-line-plots/
        # https://community.plot.ly/t/will-there-be-3d-bar-charts-in-the-future/1045/3
        # https://matplotlib.org/examples/mplot3d/bars3d_demo.html
        if steps is None:
            # steps = [int(x) for x in np.linspace(min(self.df.step), max(self.df.step), 6)]
            steps = [int(x) for x in np.logspace(min(self.df.step), np.log2(max(self.df.step)), 8, base=2)]
            # steps = [int(x) for x in np.logspace(min(self.df.step), np.log10(max(self.df.step)), 8)]
            steps[0] = 0
        networks_availables = self.df.network.unique()
        cols = ['step', key]
        fig = go.Figure()
        for step in steps:  # TODO: verify steps are in df
            for i, net in enumerate(networks_availables):
                filter = [a == net and b == step for a, b in zip(self.df.network, self.df.step)]
                data_to_plot = self.df[filter][cols]
                # https://stackoverflow.com/a/60403270/5931672
                counts, bins = np.histogram(data_to_plot[key], bins=10, density=False)
                counts = list(np.repeat(counts, 2).tolist())    # I do this to stop pycharm warning
                counts.insert(0, 0)
                counts.append(0)
                bins = np.repeat(bins, 2)

                fig.add_traces(go.Scatter3d(x=[step] * len(counts), y=bins, z=counts,
                                            mode='lines', name=net.replace("_", " ") + "; step: " + str(step),
                                            surfacecolor=add_transparency(DEFAULT_PLOTLY_COLORS[i], 0),
                                            surfaceaxis=0,
                                            line=dict(color=DEFAULT_PLOTLY_COLORS[i], width=4)
                                            )
                               )
                """
                # https://plot.ly/python/3d-surface-coloring/
                fig.add_traces(go.Surface(
                    x=[step] * len(counts), y=bins, z=counts,
                    surfacecolor=extract_values(DEFAULT_PLOTLY_COLORS[i]).append(0.3))
                )"""

        fig.update_layout(title='Multiple',
                          scene=dict(
                              xaxis=dict(title='step'),
                              yaxis=dict(title=key),
                              zaxis=dict(title='counts'),
                              xaxis_type="log"))
        plotly.offline.plot(fig, filename=str(self.path / (key + "_3d_histogram.html")))


if __name__ == "__main__":
    # plotter = Plotter("./log/2020/02February/25Tuesday/run-14h16m23")
    # plotter.plot_everything(library="plotly", reload=True, showfig=True, savefig=True)
    # plotter.get_full_pandas_dataframe()
    monte_carlo_analyzer = MonteCarloAnalyzer(df=None,
                                              path="./montecarlo/2020/02February/26Wednesday/run-19h16m20/run_data.csv")
    # monte_carlo_analyzer.plot_histogram(library='plotly')
    # monte_carlo_analyzer.monte_carlo_plotter.plot_key(library='plotly')
    # monte_carlo_analyzer.monte_carlo_plotter.plot_distribution()
    monte_carlo_analyzer.plot_3d_hist()
    # monte_carlo_analyzer.monte_carlo_plotter.plot_distribution('test loss')

__author__ = 'J. Agustin BARRACHINA'
__version__ = '0.1.4'
__maintainer__ = 'J. Agustin BARRACHINA'
__email__ = 'joseagustin.barra@gmail.com; jose-agustin.barrachina@centralesupelec.fr'
