from dataclasses import dataclass
import pandas as pd
import numpy as np
import re
import os
import sys
from pathlib import Path
from pdb import set_trace
from cvnn.utils import create_folder
import logging
import cvnn

logger = logging.getLogger(cvnn.__name__)
AVAILABLE_LIBRARIES = set()

try:
    import plotly
    import plotly.graph_objects as go
    import plotly.figure_factory as ff
    AVAILABLE_LIBRARIES.add('plotly')
except ImportError as e:
    logger.info("Plotly not installed, consider installing it to get more plotting capabilities")
try:
    import matplotlib.pyplot as plt
    AVAILABLE_LIBRARIES.add('matplotlib')
    DEFAULT_MATPLOTLIB_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']  # [1:] Uncomment to remove blue color
except ImportError as e:
    logger.info("Matplotlib not installed, consider installing it to get more plotting capabilities")
if 'matplotlib' in AVAILABLE_LIBRARIES:
    try:
        import seaborn as sns
        AVAILABLE_LIBRARIES.add('seaborn')
    except ImportError as e:
        logger.info("Seaborn not installed, consider installing it to get more plotting capabilities")
    try:
        import tikzplotlib
        AVAILABLE_LIBRARIES.add('tikzplotlib')
    except ImportError as e:
        logger.info("Tikzplotlib not installed, consider installing it to get more plotting capabilities")

DEFAULT_PLOTLY_COLORS = ['rgb(31, 119, 180)',  # Blue
                         'rgb(255, 127, 14)',  # Orange
                         'rgb(44, 160, 44)',  # Green
                         'rgb(214, 39, 40)',
                         'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
                         'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
                         'rgb(188, 189, 34)', 'rgb(23, 190, 207)']


@dataclass
class Resolution:
    width: int
    height: int


RESOLUTIONS_16_9 = {
    'lowest': Resolution(1024, 576),
    'low': Resolution(1152, 648),
    'HD': Resolution(1280, 720),  # 720p
    'FHD': Resolution(1920, 1080),  # 1080p
    'QHD': Resolution(2560, 1440),  # 1440p
    'UHD': Resolution(2560, 1440)  # 4K or 2160p
}
RESOLUTIONS_4_3 = {
    '640×480': Resolution(640, 480),
    '800×600': Resolution(800, 600),
    '960×720': Resolution(960, 720),
    '1024×768': Resolution(1024, 768),
    '1280×960': Resolution(1280, 960),
    # https://www.comtech-networking.com/blog/item/4-what-is-the-screen-resolution-or-the-aspect-ratio-what-do-720p-1080i-1080p-mean/
}

PLOTLY_CONFIG = {
    'scrollZoom': True,
    'editable': True,
    'toImageButtonOptions': {
        'format': 'svg',  # one of png, svg, jpeg, webp
        # 'filename': 'custom_image',
        'height': RESOLUTIONS_4_3['800×600'].height,
        'width': RESOLUTIONS_4_3['800×600'].width,
        'scale': 1  # Multiply title/legend/axis/canvas sizes by this factor
    }
}


def triangulate_histogram(x, y, z):
    # https://community.plot.ly/t/adding-a-shape-to-a-3d-plot/1441/8?u=negu93
    if len(x) != len(y) != len(z):
        raise ValueError("The  lists x, y, z, must have the same length")
    n = len(x)
    if n % 2:
        raise ValueError("The length of lists x, y, z must be an even number")
    pts3d = np.vstack((x, y, z)).T
    pts3dp = np.array([[x[2 * k + 1], y[2 * k + 1], 0] for k in range(1, n // 2 - 1)])
    pts3d = np.vstack((pts3d, pts3dp))
    # triangulate the histogram bars:
    tri = [[0, 1, 2], [0, 2, n]]
    for k, i in zip(list(range(n, n - 3 + n // 2)), list(range(3, n - 4, 2))):
        tri.extend([[k, i, i + 1], [k, i + 1, k + 1]])
    tri.extend([[n - 3 + n // 2, n - 3, n - 2], [n - 3 + n // 2, n - 2, n - 1]])
    return pts3d, np.array(tri)


def add_transparency(color='rgb(31, 119, 180)', alpha=0.5):
    pattern = re.compile("^rgb\([0-9]+, [0-9]+, [0-9]+\)$")
    if not re.match(pattern, color):
        logger.error("Unrecognized color format")
        sys.exit(-1)
    color = re.sub("^rgb", "rgba", color)
    color = re.sub("\)$", ", {})".format(alpha), color)
    return color


def extract_values(color='rgb(31, 119, 180)'):
    pattern = re.compile("^rgb\([0-9]+, [0-9]+, [0-9]+\)$")
    if not re.match(pattern, color):
        logger.error("Unrecognized color format")
        sys.exit(-1)
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
    if 'matplotlib' not in AVAILABLE_LIBRARIES:
        logger.warning("No Matplotlib installed, function " + add_params.__name__ + " was called but will be omitted")
        return None
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
        os.makedirs(os.path.split(filename)[0], exist_ok=True)
        fig.savefig(filename, transparent=True)


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


# ----------------
# Confusion Matrix
# ----------------


def plot_confusion_matrix(data, filename=None, library='plotly', axis_legends=None, showfig=False):
    if library == 'seaborn':
        if 'seaborn' not in AVAILABLE_LIBRARIES:
            logger.warning("No Seaborn installed, function " + plot_confusion_matrix.__name__ + " was called but will be omitted")
            return None
        else:
            fig, ax = plt.subplots()
            sns.heatmap(data,
                        annot=True,
                        linewidths=.5,
                        cbar=True,
                        )
            if filename is not None:
                fig.savefig(filename)
            if showfig:
                fig.show()
    elif library == 'plotly':
        if 'plotly' not in AVAILABLE_LIBRARIES:
            logger.warning(
                "No Plotly installed, function " + plot_confusion_matrix.__name__ + " was called but will be omitted")
            return None
        else:
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
                        logger.critical("WTF?! should never have arrived here")
                for i in data.axes[1].tolist():
                    if isinstance(i, int):
                        x.append(axis_legends[i])
                    elif isinstance(i, str):
                        x.append(i)
                    else:
                        logger.critical("WTF?! should never have arrived here")
            # fig = go.Figure(data=go.Heatmap(z=z, x=x, y=y))
            fig = ff.create_annotated_heatmap(z, x=x, y=y)
            if showfig:
                fig.show()


def confusion_matrix(y_pred_np, y_label_np, filename=None, axis_legends=None):
    # TODO: Assert matrices are not empty
    categorical = (len(np.shape(y_label_np)) > 1)
    if categorical:
        y_pred_np = np.argmax(y_pred_np, axis=1)
        y_label_np = np.argmax(y_label_np, axis=1)
    y_pred_pd = pd.Series(y_pred_np, name='Predicted')
    y_label_pd = pd.Series(y_label_np, name='Actual')
    df = pd.crosstab(y_label_pd, y_pred_pd, rownames=['Actual'], colnames=['Predicted'], margins=True)
    if filename is not None:
        df.to_csv(filename)
    # plot_confusion_matrix(df, filename, library='plotly', axis_legends=axis_legends)
    return df


# ----------------
# Comparison
# ----------------


class SeveralMonteCarloComparison:

    def __init__(self, label, x, paths, round=2):
        """
        This class is used to compare several monte carlo runs done with cvnn.montecarlo.MonteCarlo class.
        MonteCarlo let's you compare different models between them but let's you not change other values like epochs.
        You can run as several MonteCarlo runs and then use SeveralMonteCarloComparison class to compare the results.

        Example of usage:

        ```
        # Run several Monte Carlo's
        for learning_rate in learning_rates:
            monte_carlo = RealVsComplex(complex_network)
            monte_carlo.run(x, y, iterations=iterations, learning_rate=learning_rate,
                            epochs=epochs, batch_size=batch_size, display_freq=display_freq,
                            shuffle=True, debug=debug, data_summary=dataset.summary())
        # Run self
        several = SeveralMonteCarloComparison('learning rate', x = learning_rates,
                                              paths = ["path/to/1st/run/run_data",
                                                       "path/to/2nd/run/run_data",
                                                       "path/to/3rd/run/run_data",
                                                       "path/to/4th/run/run_data"]
        several.box_plot(showfig=True)
        ```

        :label: string that describes what changed between each montecarlo run
        :x: List of the value for each monte carlo run wrt :label:.
        :paths: Full path to each monte carlo run_data saved file (Must end with run_data)
            NOTE: x and paths must be the same size
        """
        self.x_label = label
        if all([item.isdigit() for item in x]):
            self.x = list(map(int, x))
        elif all([item.replace(".", "", 1).isdigit() for item in x]):
            self.x = np.round(list(map(float, x)), round)
        else:
            self.x = x
        self.monte_carlo_runs = []
        for path in paths:
            self.monte_carlo_runs.append(MonteCarloAnalyzer(path=path))
        if not len(self.x) == len(self.monte_carlo_runs):
            logger.error("x ({0}) and paths ({1}) must be the same size".format(len(self.x),
                                                                                len(self.monte_carlo_runs)))
        frames = [self.monte_carlo_runs[0].df]
        # frames[0]['network'] = frames[0]['network'] + " " + x[0]
        for i, monte_carlo_run in enumerate(self.monte_carlo_runs[1:]):
            frames.append(self.monte_carlo_runs[i + 1].df)
            # frames[i + 1]['network'] = frames[i + 1]['network'] + " " + x[i + 1]
        self.df = pd.concat(frames)
        self.monte_carlo_analyzer = MonteCarloAnalyzer(df=self.df)

    def box_plot(self, key='accuracy', library='plotly', epoch=-1, showfig=False, savefile=None):
        if library == 'plotly':
            self._box_plot_plotly(key=key, epoch=epoch, showfig=showfig, savefile=savefile)
        # TODO: https://seaborn.pydata.org/examples/grouped_boxplot.html
        elif library == 'seaborn':
            self._box_plot_seaborn(key=key, epoch=epoch, showfig=showfig, savefile=savefile)
        else:
            logger.warning("Unrecognized library to plot " + library)
        return None

    def _box_plot_plotly(self, key='accuracy', epoch=-1, showfig=False, savefile=None):
        # https://en.wikipedia.org/wiki/Box_plot
        # https://plot.ly/python/box-plots/
        # https://towardsdatascience.com/understanding-boxplots-5e2df7bcbd51
        # Median (Q2 / 50th Percentile): Middle value of the dataset. ex. median([1, 3, 3, 6, 7, 8, 9]) = 6
        # First quartile (Q1 / 25th Percentile): Middle value between the median and the min(dataset) = 1
        # Third quartile (Q3 / 75th Percentile): Middle value between the median and the max(dataset) = 9
        # Interquartile Range (IQR) = Q3 - Q1
        # Whishker: [Q1 - 1.5*IQR, Q3 + 1.5*IQR], whatever is out of this is an outlier.
        # suspected outlier: [Q1 - 3*IQR, Q3 + 3*IQR]
        if 'plotly' not in AVAILABLE_LIBRARIES:
            logger.warning(
                "No Plotly installed, function " + self._box_plot_plotly.__name__ + " was called but will be omitted")
            return None
        savefig = False
        if savefile is not None:
            savefig = True

        epochs = []
        for i in range(len(self.monte_carlo_runs)):
            if epoch == -1:
                epochs.append(max(self.monte_carlo_runs[i].df.epoch))
            else:
                epochs.append(epoch)

        fig = go.Figure()

        for i, mc_run in enumerate(self.monte_carlo_runs):
            df = mc_run.df
            networks_availables = df.network.unique()
            for color_index, net in enumerate(networks_availables):
                filter = [a == net and b == epochs[i] for a, b in zip(df.network, df.epoch)]
                data = df[filter]
                fig.add_trace(go.Box(
                    y=data[key],
                    # x=[self.x[i]] * len(data[key]),
                    name=net.replace('_', ' ') + " " + str(self.x[i]),
                    whiskerwidth=0.2,
                    notched=True,  # confidence intervals for the median
                    fillcolor=add_transparency(DEFAULT_PLOTLY_COLORS[color_index], 0.5),
                    boxpoints='suspectedoutliers',  # to mark the suspected outliers
                    line=dict(color=DEFAULT_PLOTLY_COLORS[color_index]),
                    boxmean=True  # Interesting how sometimes it falls outside the box
                ))

        fig.update_layout(
            title=self.x_label + ' Box Plot',
            xaxis=dict(title=self.x_label),
            yaxis=dict(
                title=key,
                autorange=True,
                showgrid=True,
                dtick=0.05,
            ),
            # boxmode='group',
            # boxgroupgap=0,
            # boxgap=0,
            showlegend=True
        )
        if savefig:
            if not savefile.endswith('.html'):
                savefile += '.html'
            os.makedirs(os.path.split(savefile)[0], exist_ok=True)
            plotly.offline.plot(fig,
                                filename=savefile, config=PLOTLY_CONFIG, auto_open=showfig)
            # fig.write_image(savefile.replace('.html', extension))
        elif showfig:
            fig.show(config=PLOTLY_CONFIG)

    def _box_plot_seaborn(self, key='accuracy', epoch=-1, showfig=False, savefile=None, extension=".svg"):
        if 'seaborn' not in AVAILABLE_LIBRARIES:
            logger.warning(
                "No Seaborn installed, function " + self._box_plot_seaborn.__name__ + " was called but will be omitted")
            return None
        epochs = []
        for i in range(len(self.monte_carlo_runs)):
            if epoch == -1:
                epochs.append(max(self.monte_carlo_runs[i].df.epoch))
            else:
                epochs.append(epoch)
        # Prepare data
        frames = []
        # color_pal = []
        for i, mc_run in enumerate(self.monte_carlo_runs):
            df = mc_run.df
            # color_pal += sns.color_palette()[:len(df.network.unique())]
            filter = df['epoch'] == epochs[i]
            data = df[filter]
            data[self.x_label] = self.x[i]
            frames.append(data)
        result = pd.concat(frames)

        # Run figure
        fig = plt.figure()
        # set_trace()
        ax = sns.boxplot(x=self.x_label, y=key, hue="network", data=result, boxprops=dict(alpha=.3))
        # palette=color_pal)
        # sns.despine(offset=1, trim=True)
        # Make black lines the color of the box
        for i, artist in enumerate(ax.artists):
            col = artist.get_facecolor()[:-1]  # the -1 removes the transparency
            artist.set_edgecolor(col)
            for j in range(i * 6, i * 6 + 6):
                line = ax.lines[j]
                line.set_color(col)
                line.set_mfc(col)
                line.set_mec(col)

        if savefile is not None:
            if not savefile.endswith(extension):
                savefile += extension
            os.makedirs(os.path.split(savefile)[0], exist_ok=True)
            fig.savefig(savefile, transparent=True)
            if 'tikzplotlib' not in AVAILABLE_LIBRARIES:
                logger.warning(
                    "No Tikzplotlib installed, function " + self._box_plot_seaborn.__name__ + " will not save tex file")
            else:
                tikzplotlib.save(Path(os.path.split(savefile)[0]) / ("tikz_box_plot_" + self.x_label + ".tex"))
        if showfig:
            fig.show()
        return fig, ax

    def save_pandas_csv_result(self, path, epoch=-1):
        # TODO: Check path
        if epoch == -1:
            epoch = max(self.monte_carlo_runs[0].df.epoch)  # TODO: Assert it's the same for all cases
        cols = ['val_loss', 'loss', 'val_accuracy', 'accuracy']
        for i, run in enumerate(self.monte_carlo_runs):
            df = run.df
            networks_availables = df.network.unique()
            for col, net in enumerate(networks_availables):
                filter = [a == net and b == epoch for a, b in zip(df.network, df.epoch)]
                data = df[filter].describe()
                data = data[cols]
                data.to_csv(path + net + "_" + self.x[i] + "_stats.csv")

    def plot_histogram(self, key='accuracy', epoch=-1, library='seaborn', showfig=False, savefig=True, title='',
                       extension=".svg"):
        self.monte_carlo_analyzer.plot_histogram(key=key, epoch=epoch, library=library, showfig=showfig, savefig=savefig,
                                                 title=title, extension=extension)


class Plotter:

    def __init__(self, path, file_suffix:str = "_results_fit.csv", data_results_dict: dict = None, model_name: str = None):
        """
        This class manages the plot of results for a model train.
        It opens the csv files (test and train) saved during training and plots results as wrt each epoch saved.
        This class is generally used to plot accuracy and loss evolution during training.

        :path: Full path where the csv results are stored
        :file_suffix: (optional) let's you filter csv files to open only files that ends with the suffix.
            By default it opens every csv file it finds.
        """
        if not os.path.exists(path):
            logger.error("Path {} does not exist".format(path))
            sys.exit(-1)
        self.path = Path(path)
        if not file_suffix.endswith(".csv"):
            file_suffix += ".csv"
        self.file_suffix = file_suffix
        self.pandas_list = []
        self.labels = []
        self.model_name = model_name
        
        if data_results_dict:
            result_pandas = pd.DataFrame.from_dict(data_results_dict)
            result_pandas.to_csv(self.path / f"{model_name}{file_suffix}", index=False)
        
        self._csv_to_pandas()

    def _csv_to_pandas(self):
        """
        Opens the csv files as pandas dataframe and stores them in a list (self.pandas_list).
        Also saves the name of the file where it got the pandas frame as a label.
        This function is called by the constructor.
        """
        self.pandas_list = []
        self.labels = []
        files = os.listdir(self.path)
        files.sort()  # Respect the colors for the plot of Monte Carlo.
        # For ComplexVsReal Monte Carlo it has first the Complex model and SECOND the real one.
        # So ordering the files makes sure I open the Complex model first and so it plots with the same colours.
        # TODO: Think a better way without loosing generality (This sort is all done because of the ComplexVsReal case)
        for file in files:
            if file.endswith(self.file_suffix):
                self.pandas_list.append(pd.read_csv(self.path / file))
                self.labels.append(re.sub(self.file_suffix + '$', '', file).replace('_', ' '))

    def reload_data(self):
        """
        If data inside the working path has changed (new csv files or modified csv files),
        this function reloads the data to be plotted with that new information.
        """
        self._csv_to_pandas()

    def get_full_pandas_dataframe(self):
        """
        Merges every dataframe obtained from each csv file into a single dataframe.
        It adds the columns:
            - network: name of the train model
            - epoch: information of the epoch index
            - path: path where the information of the train model was saved (used as parameter with the constructor)
        :retun: pd.Dataframe
        """
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html
        self._csv_to_pandas()
        if len(self.pandas_list) == 0:
            logger.error("Error: There was no csv logs to open")
            sys.exit(-1)
        length = len(self.pandas_list[0])
        for data_frame in self.pandas_list:  # TODO: Check if.
            if not length == len(data_frame):  # What happens if NaN? Can I cope not having same len?
                logger.error("Data frame length should have been {0} and was {1}".format(length, len(data_frame)))

        result = pd.DataFrame({
            'network': [self.get_net_name()] * length,
            'epoch': list(range(1, length+1)),
            'path': [self.path] * length
        })

        for data_frame, data_label in zip(self.pandas_list, self.labels):
            # data_frame.columns = [data_label + " " + str(col) for col in data_frame.columns]
            # concatenated = pd.concat(self.pandas_list, keys=self.labels)
            result = pd.concat([result, data_frame], axis=1, sort=False)
        return result

    def get_net_name(self) -> str:
        if self.model_name:
            return self.model_name
        str_to_match = "_metadata.txt"
        for file in os.listdir(self.path):
            if file.endswith(str_to_match):
                # See that there is no need to open the file
                return re.sub(str_to_match + "$", '', file).replace('_', ' ')
        return "Name not found"

    # ====================
    #        Plot
    # ====================

    def plot_everything(self, reload=False, library='plotly', showfig=False, savefig=True, index_loc=None,
                        extension=".svg"):
        if reload:
            self._csv_to_pandas()
        if not len(self.pandas_list) != 0:
            logger.error("Empty pandas list to plot")
            return None
        for key in ["loss", "accuracy"]:
            self.plot_key(key, reload=False, library=library, showfig=showfig, savefig=savefig, index_loc=index_loc,
                          extension=extension)

    def plot_key(self, key='loss', reload=False, library='plotly', showfig=False, savefig=True, index_loc=None,
                 extension=".svg"):
        if reload:
            self._csv_to_pandas()
        if library == 'matplotlib':
            self._plot_matplotlib(key=key, showfig=showfig, savefig=savefig, index_loc=index_loc, extension=extension)
        elif library == 'plotly':
            self._plot_plotly(key=key, showfig=showfig, savefig=savefig, index_loc=index_loc)
        else:
            logger.warning("Unrecognized library to plot " + library)

    def _plot_matplotlib(self, key='loss', showfig=False, savefig=True, index_loc=None, extension=".svg"):
        if 'matplotlib' not in AVAILABLE_LIBRARIES:
            logger.warning("No Matplotlib installed, function " + self._plot_matplotlib.__name__ +
                           " was called but will be omitted")
            return None
        fig, ax = plt.subplots()
        ax.set_prop_cycle('color', DEFAULT_MATPLOTLIB_COLORS)
        title = None
        for i, data in enumerate(self.pandas_list):
            if title is not None:
                title += " vs. " + self.labels[i]
            else:
                title = self.labels[i]
            for k in data:
                if key in k:
                    if index_loc is not None:
                        if 'stats' in data.keys():
                            data = data[data['stats'] == 'mean']
                        else:
                            logger.warning("Warning: Trying to index an array without index")
                    ax.plot(data[k], 'o-', label=(k.replace(key, '') + self.labels[i]).replace('_', ' '))
        title += " " + key
        fig.legend(loc="upper right")
        ax.set_ylabel(key)
        ax.set_xlabel("epoch")
        ax.set_title(title)
        if showfig:
            fig.show()
        if savefig:
            fig.savefig(str(self.path / key) + "_matplotlib" + extension, transparent=True)

    def _plot_plotly(self, key='loss', showfig=False, savefig=True, func=min, index_loc=None):
        if 'plotly' not in AVAILABLE_LIBRARIES:
            logger.warning("No Plotly installed, function " + self._plot_plotly.__name__ +
                           " was called but will be omitted")
            return None
        fig = go.Figure()
        annotations = []
        title = None
        for i, data in enumerate(self.pandas_list):
            if title is not None:
                title += " vs. " + self.labels[i]
            else:
                title = self.labels[i]
            j = 0
            for k in data:
                if key in k:
                    color = DEFAULT_PLOTLY_COLORS[j * len(self.pandas_list) + i]
                    j += 1
                    if index_loc is not None:
                        if 'stats' in data.keys():
                            data = data[data['stats'] == 'mean']
                        else:
                            logger.warning("Trying to index an array without index")
                    x = list(range(len(data[k])))
                    fig.add_trace(go.Scatter(x=x, y=data[k], mode='lines',
                                             name=(k.replace(key, '') + self.labels[i]).replace('_', ' '),
                                             line_color=color))
                    # Add points
                    fig.add_trace(go.Scatter(x=[x[-1]],
                                             y=[data[k].to_list()[-1]],
                                             mode='markers',
                                             name='last value',
                                             marker_color=color))
                    # Max/min points
                    func_value = func(data[k])
                    # ATTENTION! this will only give you first occurrence
                    func_index = data[k].to_list().index(func_value)
                    if func_index != len(data[k]) - 1:
                        fig.add_trace(go.Scatter(x=[func_index],
                                                 y=[func_value],
                                                 mode='markers',
                                                 name=func.__name__,
                                                 text=['{0:.2f}%'.format(func_value)],
                                                 textposition="top center",
                                                 marker_color=color))
                        # Min annotations
                        annotations.append(dict(xref="x", yref="y", x=func_index, y=func_value,
                                                xanchor='left', yanchor='middle',
                                                text='{0:.2f}'.format(func_value),
                                                font=dict(family='Arial',
                                                          size=14),
                                                showarrow=False, ay=-40))
                    # Right annotations
                    annotations.append(dict(xref='paper', x=0.95, y=data[k].to_list()[-1],
                                            xanchor='left', yanchor='middle',
                                            text='{0:.2f}'.format(data[k].to_list()[-1]),
                                            font=dict(family='Arial',
                                                      size=16),
                                            showarrow=False))
        title += " " + key
        fig.update_layout(annotations=annotations,
                          title=title,
                          xaxis_title='epochs',
                          yaxis_title=key)
        if savefig:
            plotly.offline.plot(fig, filename=str(self.path / key) + ".html",
                                config=PLOTLY_CONFIG, auto_open=showfig)
            # fig.write_image(str(self.path / key) + "_plotly" + extension)
        elif showfig:
            fig.show(config=PLOTLY_CONFIG)


class MonteCarloPlotter(Plotter):

    def __init__(self, path):
        file_suffix = "_statistical_result.csv"
        super().__init__(path, file_suffix=file_suffix)

    def plot_everything(self, reload=False, library='plotly', showfig=False,
                        savefig=True, index_loc='mean', extension=".svg"):
        # Rename this function to change index_loc default value
        super().plot_everything(reload=False, library=library, showfig=showfig, savefig=savefig, index_loc=index_loc)

    def plot_key(self, key='accuracy', reload=False, library='plotly', showfig=False, savefig=True,
                 index_loc='mean', extension=".svg"):
        # Rename this function to change index_loc default value
        super().plot_key(key, reload, library, showfig, savefig, index_loc, extension=extension)

    def plot_line_confidence_interval(self, key='accuracy', showfig=False, savefig=True, library='matplotlib',
                                      title='', full_border=True, x_axis='epoch', extension=".svg"):
        if library == 'plotly':
            self._plot_line_confidance_interval_plotly(key=key, showfig=showfig, savefig=savefig,
                                                       title=title, full_border=full_border, x_axis=x_axis)
        elif library == 'matplotlib' or library == 'seaborn':
            self._plot_line_confidence_interval_matplotlib(key=key, showfig=showfig, savefig=savefig,
                                                           title=title, x_axis=x_axis, extension=extension)
        else:
            logger.warning("Warning: Unrecognized library to plot " + library)
            return None

    def _plot_line_confidence_interval_matplotlib(self, key='accuracy', showfig=False, savefig=True,
                                                  title='', x_axis='epoch', extension=".svg"):
        if 'matplotlib' not in AVAILABLE_LIBRARIES:
            logger.warning("No Matplotlib installed, function " +
                           self._plot_line_confidence_interval_matplotlib.__name__ + " was called but will be omitted")
            return None
        fig, ax = plt.subplots()
        for i, data in enumerate(self.pandas_list):
            x = data[x_axis].unique().tolist()
            data_mean = data[data['stats'] == 'mean'][key].tolist()
            data_max = data[data['stats'] == 'max'][key].tolist()
            data_min = data[data['stats'] == 'min'][key].tolist()
            data_50 = data[data['stats'] == '50%'][key].tolist()
            data_25 = data[data['stats'] == '25%'][key].tolist()
            data_75 = data[data['stats'] == '75%'][key].tolist()
            ax.plot(x, data_mean, color=DEFAULT_MATPLOTLIB_COLORS[i],
                    label=self.labels[i].replace('_', ' ') + ' mean')
            ax.plot(x, data_50, '--', color=DEFAULT_MATPLOTLIB_COLORS[i],
                    label=self.labels[i].replace('_', ' ') + ' median')
            ax.fill_between(x, data_25, data_75, color=DEFAULT_MATPLOTLIB_COLORS[i], alpha=.4,
                            label=self.labels[i].replace('_', ' ') + ' interquartile')
            ax.fill_between(x, data_min, data_max, color=DEFAULT_MATPLOTLIB_COLORS[i], alpha=.15,
                            label=self.labels[i].replace('_', ' ') + ' border')
        for label in self.labels:
            title += label.replace('_', ' ') + ' vs '
        title = title[:-3] + key

        ax.set_title(title)
        ax.set_xlabel(x_axis)
        ax.set_ylabel(key)
        ax.grid()
        ax.legend()
        if showfig:
            fig.show()
        if savefig:
            os.makedirs(str(self.path / "plots/lines_confidence/"), exist_ok=True)
            fig.savefig(self.path / ("plots/lines_confidence/montecarlo_" +
                                     key.replace(" ", "_") + "_matplotlib" + extension), transparent=True)
            if 'tikzplotlib' not in AVAILABLE_LIBRARIES:
                logger.warning(
                    "No Tikzplotlib installed, function " + self._plot_line_confidence_interval_matplotlib.__name__ +
                    " will not save tex file")
            else:
                tikzplotlib.save(self.path / ("plots/lines_confidence/montecarlo_" +
                                              key.replace(" ", "_") + "_matplotlib" + ".tex"))
        # set_trace()

    def _plot_line_confidance_interval_plotly(self, key='accuracy', showfig=False, savefig=True,
                                              title='', full_border=True, x_axis='epoch'):
        if 'plotly' not in AVAILABLE_LIBRARIES:
            logger.warning("No Plotly installed, function " + self._plot_line_confidance_interval_plotly.__name__ +
                           " was called but will be omitted")
            return None
        fig = go.Figure()
        for i, data in enumerate(self.pandas_list):
            # set_trace()
            x = data[x_axis].unique().tolist()
            x_rev = x[::-1]
            data_mean = data[data['stats'] == 'mean'][key].tolist()
            data_max = data[data['stats'] == 'max'][key].tolist()
            data_min = data[data['stats'] == 'min'][key][::-1].tolist()
            data_50 = data[data['stats'] == '50%'][key].tolist()
            data_25 = data[data['stats'] == '25%'][key][::-1].tolist()
            data_75 = data[data['stats'] == '75%'][key].tolist()
            # set_trace()
            if full_border:
                fig.add_trace(go.Scatter(
                    x=x + x_rev,
                    y=data_max + data_min,
                    fill='toself',
                    fillcolor=add_transparency(DEFAULT_PLOTLY_COLORS[i], 0.1),
                    line_color=add_transparency(DEFAULT_PLOTLY_COLORS[i], 0),
                    showlegend=True,
                    name=self.labels[i].replace('_', ' ') + " borders",
                ))
            fig.add_trace(go.Scatter(
                x=x + x_rev,
                y=data_75 + data_25,
                fill='toself',
                fillcolor=add_transparency(DEFAULT_PLOTLY_COLORS[i], 0.2),
                line_color=add_transparency(DEFAULT_PLOTLY_COLORS[i], 0),
                showlegend=True,
                name=self.labels[i].replace('_', ' ') + " interquartile",
            ))
            fig.add_trace(go.Scatter(
                x=x, y=data_mean,
                line_color=DEFAULT_PLOTLY_COLORS[i],
                name=self.labels[i].replace('_', ' ') + " mean",
            ))
            fig.add_trace(go.Scatter(
                x=x, y=data_50,
                line=dict(color=DEFAULT_PLOTLY_COLORS[i], dash='dash'),
                name=self.labels[i].replace('_', ' ') + " median",
            ))
        for label in self.labels:
            title += label.replace('_', ' ') + ' vs '
        title = title[:-3] + key

        fig.update_traces(mode='lines')
        fig.update_layout(title=title, xaxis_title=x_axis, yaxis_title=key)

        if savefig:
            os.makedirs(str(self.path / "plots/lines_confidence/"), exist_ok=True)
            plotly.offline.plot(fig,
                                filename=str(self.path / ("plots/lines_confidence/montecarlo_" +
                                                          key.replace(" ", "_"))) + ".html",
                                config=PLOTLY_CONFIG, auto_open=showfig)
            # fig.write_image(str(self.path / ("plots/lines/montecarlo_" + key.replace(" ", "_"))) + extension)
        elif showfig:
            fig.show(config=PLOTLY_CONFIG)

    def plot_train_vs_test(self, key='loss', showfig=False, savefig=True, median=False, x_axis='epoch'):
        if 'plotly' not in AVAILABLE_LIBRARIES:
            logger.warning("No Plotly installed, function " + self.plot_train_vs_test.__name__ +
                           " was called but will be omitted")
            return None
        fig = go.Figure()
        # test plots
        label = 'mean'
        if median:
            label = '50%'
        for i, data in enumerate(self.pandas_list):
            x = data[x_axis].unique().tolist()
            data_mean_test = data[data['stats'] == label]["test " + key].tolist()
            fig.add_trace(go.Scatter(
                x=x, y=data_mean_test,
                line_color=DEFAULT_PLOTLY_COLORS[i],
                name=self.labels[i] + " test",
            ))
            data_mean_train = data[data['stats'] == label]["train " + key].tolist()
            fig.add_trace(go.Scatter(
                x=x, y=data_mean_train,
                line_color=DEFAULT_PLOTLY_COLORS[i + len(self.pandas_list)],
                name=self.labels[i].replace("_", " ") + " train ",
            ))
        title = "train and test " + key + " " + label.replace("50%", "median")
        fig.update_traces(mode='lines')
        fig.update_layout(title=title, xaxis_title=x_axis, yaxis_title=key)

        if savefig:
            os.makedirs(self.path / "plots/lines/", exist_ok=True)
            plotly.offline.plot(fig,
                                filename=str(self.path / ("plots/lines/montecarlo_" + key.replace(" ", "_")))
                                         + "_" + label.replace("50%", "median") + ".html",
                                config=PLOTLY_CONFIG, auto_open=showfig)
        elif showfig:
            fig.show(config=PLOTLY_CONFIG)


class MonteCarloAnalyzer:

    def __init__(self, df=None, path=None):
        self.confusion_matrix = []
        if path is not None and df is not None:  # I have data and the place where I want to save it
            self.df = df  # DataFrame with all the data
            self.path = Path(path)
            self.df.to_csv(self.path / "run_data.csv")  # Save the results for latter use
        elif path is not None and df is None:  # Load df from Path
            if not path.endswith('.csv'):
                path += '.csv'
            self.df = pd.read_csv(Path(path))  # Path(__file__).parents[1].absolute() /
            self.path = Path(os.path.split(path)[0])  # Keep only the path and not the filename
        elif path is None and df is not None:  # Save df into default path
            self.path = create_folder("./log/montecarlo/")
            self.df = df  # DataFrame with all the data
            self.df.to_csv(self.path / "run_data.csv")  # Save the results for latter use
        else:  # I have nothing
            self.path = create_folder("./log/montecarlo/")
            self.df = pd.DataFrame()
        self.plotable_info = ['loss', 'val_loss', 'accuracy', 'val_accuracy']  # TODO: Consider delete
        self.monte_carlo_plotter = MonteCarloPlotter(self.path)
        self.summary = []

    def set_df(self, df, conf_mat=None):
        self.df = df  # DataFrame with all the data
        self.df.to_csv(self.path / "run_data.csv")  # Save the results for latter use
        if conf_mat is not None:
            for i in range(len(conf_mat)):
                mat = conf_mat[i]["matrix"]
                group = mat.groupby(mat.index)
                self.confusion_matrix.append({"name": conf_mat[i]["name"], "matrix": group.mean()})
        self.save_stat_results()
        self.monte_carlo_plotter.reload_data()

    def save_stat_results(self):
        # save csv file for each network with 4 columns
        self.summary = []
        networks_availables = self.df.network.unique()
        for net in networks_availables:
            data = self.df[self.df.network == net]
            cols = ['loss', 'val_loss', 'accuracy', 'val_accuracy']
            frames = []
            keys = []
            for epoch in data.epoch.unique():
                desc_frame = data[data.epoch == epoch][cols].describe()
                frames.append(desc_frame)
                keys.append(epoch)
            data_to_save = pd.concat(frames, keys=keys, names=['epoch', 'stats'])
            data_to_save.to_csv(self.path / (net + "_statistical_result.csv"))
            self.summary.append(data_to_save)
        # Save confusion matrix
        for i in range(len(self.confusion_matrix)):
            self.confusion_matrix[i]["matrix"].to_csv(self.path / (self.confusion_matrix[i]["name"]
                                                                   + "_confusion_matrix.csv"))

    # ------------
    # Plot methods
    # ------------

    def do_all(self, extension=".svg", showfig=False, savefig=True):
        """self.monte_carlo_plotter.plot_train_vs_test(key='loss')
        self.monte_carlo_plotter.plot_train_vs_test(key='accuracy')
        self.monte_carlo_plotter.plot_train_vs_test(key='loss', median=True)
        self.monte_carlo_plotter.plot_train_vs_test(key='accuracy', median=True)"""

        key_list = ['accuracy', 'loss', 'val_accuracy', 'val_loss']
        for key in key_list:
            # self.plot_3d_hist(key=key)
            for lib in ['seaborn', 'plotly']:
                try:
                    self.box_plot(key=key, extension=extension, library=lib, showfig=showfig, savefig=savefig)
                except:
                    logger.warning("Could not plot " + key + " Histogram with " + str(lib), exc_info=True)
                try:
                    self.plot_histogram(key=key, library=lib, showfig=showfig, savefig=savefig, extension=extension)
                except np.linalg.LinAlgError:
                    logger.warning("Could not plot Histogram with " + str(lib) + " because matrix was singular",
                                   exc_info=True)
                except:
                    logger.warning("Could not plot " + key + " Histogram with " + str(lib), exc_info=True)
                try:
                    self.monte_carlo_plotter.plot_line_confidence_interval(key=key, x_axis='epoch', library=lib,
                                                                           showfig=showfig, savefig=savefig)
                except:
                    logger.warning("Could not plot " + key + " line_confidence_interval with " + str(lib),
                                   exc_info=True)

    def box_plot(self, epoch=-1, library='plotly', key='val_accuracy', showfig=False, savefig=True, extension='.svg'):
        if library == 'plotly':
            self._box_plot_plotly(key=key, epoch=epoch, showfig=showfig, savefig=savefig)
        elif library == 'seaborn':
            self._box_plot_seaborn(key=key, epoch=epoch, showfig=showfig, savefig=savefig, extension=extension)
        else:
            logger.warning("Warning: Unrecognized library to plot " + library)
            return None

    def _box_plot_plotly(self, epoch=-1, key='val_accuracy', showfig=False, savefig=True):
        if 'plotly' not in AVAILABLE_LIBRARIES:
            logger.warning("No Plotly installed, function " + self._box_plot_plotly.__name__ +
                           " was called but will be omitted")
            return None
        fig = go.Figure()
        if epoch == -1:
            epoch = max(self.df.epoch)
        networks_availables = self.df.network.unique()
        # set_trace()
        for col, net in enumerate(networks_availables):
            filter = [a == net and b == epoch for a, b in zip(self.df.network, self.df.epoch)]
            data = self.df[filter]
            fig.add_trace(go.Box(
                y=data[key],
                # x=[self.x[i]] * len(data[key]),
                name=net.replace('_', ' '),
                whiskerwidth=0.2,
                notched=True,  # confidence intervals for the median
                fillcolor=add_transparency(DEFAULT_PLOTLY_COLORS[col], 0.5),
                boxpoints='suspectedoutliers',  # to mark the suspected outliers
                line=dict(color=DEFAULT_PLOTLY_COLORS[col]),
                boxmean=True  # Interesting how sometimes it falls outside the box
            ))
        fig.update_layout(
            # title='Montecarlo Box Plot ' + key,
            xaxis=dict(
                title="network",
            ),
            yaxis=dict(
                title=key,
                autorange=True,
                showgrid=True,
                dtick=0.05,
            ),
            # boxmode='group',
            # boxgroupgap=0,
            # boxgap=0,
            showlegend=False
        )
        if savefig:
            os.makedirs(self.path / "plots/box_plot/", exist_ok=True)
            plotly.offline.plot(fig,
                                filename=str(self.path / (
                                        "plots/box_plot/montecarlo_" + key.replace(" ", "_") + "_box_plot.html")),
                                config=PLOTLY_CONFIG, auto_open=showfig)
            # fig.write_image(str(self.path / ("plots/box_plot/montecarlo_" + key.replace(" ", "_")
            #                                 + "_box_plot" + extension)))
        elif showfig:
            fig.show(config=PLOTLY_CONFIG)

    def _box_plot_seaborn(self, epoch=-1, key='val_accuracy', showfig=False, savefig=True, extension='.svg'):
        if 'seaborn' not in AVAILABLE_LIBRARIES:
            logger.warning("No Seaborn installed, function " + self._box_plot_seaborn.__name__ +
                           " was called but will be omitted")
            return None
        if epoch == -1:
            epoch = max(self.df.epoch)
        # Prepare data
        filter = self.df['epoch'] == epoch
        data = self.df[filter]

        # Run figure
        fig = plt.figure()
        ax = sns.boxplot(x="network", y=key, data=data, boxprops=dict(alpha=.3))
        # Make black lines the color of the box
        for i, artist in enumerate(ax.artists):
            col = artist.get_facecolor()[:-1]  # the -1 removes the transparency
            artist.set_edgecolor(col)
            for j in range(i * 6, i * 6 + 6):
                line = ax.lines[j]
                line.set_color(col)
                line.set_mfc(col)
                line.set_mec(col)

        if savefig is not None:
            os.makedirs(self.path / "plots/box_plot/", exist_ok=True)
            filename = str(self.path / ("plots/box_plot/montecarlo_" + key.replace(" ", "_") + "_box_plot"))
            fig.savefig(filename + extension)
            if 'tikzplotlib' not in AVAILABLE_LIBRARIES:
                logger.warning(
                    "No Tikzplotlib installed, function " + self._box_plot_seaborn.__name__ + " will not save tex file")
            else:
                tikzplotlib.save(filename + ".tex")
        if showfig:
            fig.show()
        return fig, ax

    def show_plotly_table(self):
        if 'plotly' not in AVAILABLE_LIBRARIES:
            logger.warning("No Plotly installed, function " + self.show_plotly_table.__name__ +
                           " was called but will be omitted")
            return None
        # TODO: Not yet debugged
        values = [key for key in self.df.keys()]
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(self.df.columns),
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[self.df.values.tolist()],
                       fill_color='lavender',
                       align='left'))
        ])
        fig.show(config=PLOTLY_CONFIG)

    def plot_3d_hist(self, epochs=None, key='val_accuracy', title=''):
        # https://stackoverflow.com/questions/60398154/plotly-how-to-make-a-3d-stacked-histogram/60403270#60403270
        # https://plot.ly/python/v3/3d-filled-line-plots/
        # https://community.plot.ly/t/will-there-be-3d-bar-charts-in-the-future/1045/3
        # https://matplotlib.org/examples/mplot3d/bars3d_demo.html
        if 'plotly' not in AVAILABLE_LIBRARIES:
            logger.warning("No Plotly installed, function " + self.plot_3d_hist.__name__ +
                           " was called but will be omitted")
            return None
        if epochs is None:
            # epochs = [int(x) for x in np.linspace(min(self.df.epoch), max(self.df.epoch), 6)]
            epochs = [int(x) for x in np.logspace(min(self.df.epoch), np.log2(max(self.df.epoch)), 8, base=2)]
            # epochs = [int(x) for x in np.logspace(min(self.df.epoch), np.log10(max(self.df.epoch)), 8)]
            epochs[0] = 0
        networks_availables = self.df.network.unique()
        cols = ['epoch', key]
        fig = go.Figure()
        for epoch in epochs:  # TODO: verify epochs are in df
            for i, net in enumerate(networks_availables):
                filter = [a == net and b == epoch for a, b in zip(self.df.network, self.df.epoch)]
                data_to_plot = self.df[filter][cols]
                # https://stackoverflow.com/a/60403270/5931672
                counts, bins = np.histogram(data_to_plot[key], bins=10, density=False)
                counts = list(np.repeat(counts, 2).tolist())  # I do this to stop pycharm warning
                counts.insert(0, 0)
                counts.append(0)
                bins = np.repeat(bins, 2)

                fig.add_traces(go.Scatter3d(x=[epoch] * len(counts), y=bins, z=counts,
                                            mode='lines', name=net.replace("_", " ") + "; epoch: " + str(epoch),
                                            surfacecolor=add_transparency(DEFAULT_PLOTLY_COLORS[i], 0),
                                            # surfaceaxis=0,
                                            line=dict(color=DEFAULT_PLOTLY_COLORS[i], width=4)
                                            )
                               )
                verts, tri = triangulate_histogram([epoch] * len(counts), bins, counts)
                x, y, z = verts.T
                I, J, K = tri.T
                fig.add_traces(go.Mesh3d(x=x, y=y, z=z, i=I, j=J, k=K, color=DEFAULT_PLOTLY_COLORS[i], opacity=0.4))
        for net in networks_availables:
            title += net + ' '
        title += key + " comparison"
        fig.update_layout(title=title,
                          scene=dict(
                              xaxis=dict(title='epoch'),
                              yaxis=dict(title=key),
                              zaxis=dict(title='counts'),
                              xaxis_type="log"))
        os.makedirs(self.path / "plots/histogram/", exist_ok=True)
        plotly.offline.plot(fig,
                            filename=str(self.path / (
                                    "plots/histogram/montecarlo_" + key.replace(" ", "_") + "_3d_histogram.html")),
                            config=PLOTLY_CONFIG, auto_open=False)

    def plot_histogram(self, key='val_accuracy', epoch=-1, library='seaborn', showfig=False, savefig=True, title='',
                       extension=".svg"):
        if library == 'matplotlib':
            self._plot_histogram_matplotlib(key=key, epoch=epoch, showfig=showfig, savefig=savefig, title=title,
                                            extension=extension)
        elif library == 'plotly':
            self._plot_histogram_plotly(key=key, epoch=epoch, showfig=showfig, savefig=savefig, title=title)
        elif library == 'seaborn':
            self._plot_histogram_seaborn(key=key, epoch=epoch, showfig=showfig, savefig=savefig, title=title,
                                         extension=extension)
        else:
            logger.warning("Warning: Unrecognized library to plot " + library)
            return None

    def _plot_histogram_matplotlib(self, key='val_accuracy', epoch=-1,
                                   showfig=False, savefig=True, title='', extension=".svg"):
        if 'matplotlib' not in AVAILABLE_LIBRARIES:
            logger.warning("No Matplotlib installed, function " + self._plot_histogram_matplotlib.__name__ +
                           " was called but will be omitted")
            return None
        fig, ax = plt.subplots()
        ax.set_prop_cycle('color', DEFAULT_MATPLOTLIB_COLORS)
        bins = np.linspace(0, 1, 501)
        min_ax = 1.0
        max_ax = 0.0
        networks_availables = self.df.network.unique()
        if epoch == -1:
            epoch = max(self.df.epoch)
        for net in networks_availables:
            filter = [a == net and b == epoch for a, b in zip(self.df.network, self.df.epoch)]
            data = self.df[filter]  # Get only the data to plot
            ax.hist(data[key], bins, alpha=0.5, label=net.replace("_", " "))
            min_ax = min(min_ax, min(data[key]))
            max_ax = max(max_ax, max(data[key]))
        title += key + " histogram"
        ax.axis(xmin=min_ax - 0.01, xmax=max_ax + 0.01)
        add_params(fig, ax, x_label=key, y_label="occurances", title=title, loc='upper right',
                   filename=self.path / (
                           "plots/histogram/montecarlo_" + key.replace(" ", "_") + "_matplotlib" + extension),
                   showfig=showfig, savefig=savefig)
        return fig, ax

    def _plot_histogram_plotly(self, key='val_accuracy', epoch=-1, showfig=False, savefig=True, title=''):
        if 'plotly' not in AVAILABLE_LIBRARIES:
            logger.warning("No Plotly installed, function " + self._plot_histogram_plotly.__name__ +
                           " was called but will be omitted")
            return None
        networks_availables = self.df.network.unique()
        if epoch == -1:
            epoch = max(self.df.epoch)
        hist_data = []
        group_labels = []
        for net in networks_availables:
            title += net + ' '
            filter = [a == net and b == epoch for a, b in zip(self.df.network, self.df.epoch)]
            data = self.df[filter]  # Get only the data to plot
            hist_data.append(data[key].to_list())
            group_labels.append(net.replace("_", " "))
            # fig.add_trace(px.histogram(np.array(data[key]), marginal="box"))
            # fig.add_trace(go.Histogram(x=np.array(data[key]), name=net))
        fig = ff.create_distplot(hist_data, group_labels, bin_size=0.0005)  # https://plot.ly/python/distplot/
        title += key + " comparison"

        # Overlay both histograms
        fig.update_layout(barmode='overlay')
        # Reduce opacity to see both histograms
        fig.update_traces(opacity=0.75)
        fig.update_layout(title=title.replace('_', ' '),
                          xaxis_title=key)
        if savefig:
            os.makedirs(self.path / "plots/histogram/", exist_ok=True)
            plotly.offline.plot(fig,
                                filename=str(self.path / ("plots/histogram/montecarlo_" + key.replace(" ", "_")
                                                          + "_histogram.html")),
                                config=PLOTLY_CONFIG, auto_open=showfig)
            # fig.write_image(str(self.path / ("plots/histogram/montecarlo_" + key.replace(" ", "_")
            #                                  + "plotly_histogram" + extension)))
        elif showfig:
            fig.show(config=PLOTLY_CONFIG)
        return fig

    def _plot_histogram_seaborn(self, key='val_accuracy', epoch=-1,
                                showfig=True, savefig=True, title='', extension=".svg"):
        if 'seaborn' not in AVAILABLE_LIBRARIES:
            logger.warning("No Seaborn installed, function " + self._plot_histogram_seaborn.__name__ +
                           " was called but will be omitted")
            return None
        fig = plt.figure()
        bins = np.linspace(0, 1, 501)
        min_ax = 1.0
        max_ax = 0.0
        ax = None
        networks_availables = self.df.network.unique()
        # networks_availables = ['complex network', 'real network', 'polar real network']
        if epoch == -1:
            epoch = max(self.df.epoch)
        for net in networks_availables:
            filter = [a == net and b == epoch for a, b in zip(self.df.network, self.df.epoch)]
            data = self.df[filter]  # Get only the data to plot
            ax = sns.histplot(data[key], bins=bins, label=net.replace("_", " "))
            min_ax = min(min_ax, min(data[key]))
            max_ax = max(max_ax, max(data[key]))
        title += " " + key + " histogram"
        # set_trace()
        ax.set_xlim((min_ax - 0.01, max_ax + 0.01))
        fig.legend(loc='upper left')  # , bbox_to_anchor=(0., 0.3, 0.5, 0.5))
        add_params(fig, ax, x_label=key.capitalize(), y_label="Occurrences",  # loc='upper left',
                   filename=self.path / (
                           "plots/histogram/montecarlo_" + key.replace(" ", "_") + "_seaborn" + extension),
                   showfig=showfig, savefig=savefig)
        if 'tikzplotlib' not in AVAILABLE_LIBRARIES:
            logger.warning(
                "No Tikzplotlib installed, function " + self._plot_histogram_seaborn.__name__ +
                " will not save tex file")
        else:
            tikzplotlib.save(self.path / ("plots/histogram/montecarlo_" + key.replace(" ", "_") + "_seaborn" + ".tex"))
        return fig, ax


if __name__ == "__main__":
    path = "/home/barrachina/Documents/onera/src/PolSar/Oberpfaffenhofen/log/montecarlo/2020/10October/06Tuesday/run-16h58m53/run_data.csv"
    monte = MonteCarloAnalyzer(path=path)
    monte.do_all(showfig=False, savefig=False)
    monte.plot_histogram(library='matplotlib', showfig=False, savefig=False)
    monte.monte_carlo_plotter.plot_train_vs_test(showfig=False, savefig=False)
    monte.monte_carlo_plotter.plot_everything(showfig=False, savefig=False)
    """
    paths = [
        '/home/barrachina/Documents/onera/src/PolSar/Oberpfaffenhofen/log/montecarlo/2020/10October/06Tuesday/run-16h58m53/run_data',
        '/home/barrachina/Documents/onera/src/PolSar/Oberpfaffenhofen/log/montecarlo/2020/10October/07Wednesday/run-06h10m00/run_data',
        '/home/barrachina/Documents/onera/src/PolSar/Oberpfaffenhofen/log/montecarlo/2020/10October/09Friday/run-19h32m25/run_data'
    ]
    several = SeveralMonteCarloComparison('Activation Functions', x=['ReLU', 'tanh', 'sigmoid'], paths=paths)
    several.box_plot(library='seaborn', showfig=True,
                     savefile='/home/barrachina/Documents/onera/src/PolSar/Oberpfaffenhofen/log/act_fun_comparison_box_plot')
    several.box_plot(library='plotly', showfig=True,
                     savefile='/home/barrachina/Documents/onera/src/PolSar/Oberpfaffenhofen/log/act_fun_comparison_box_plot')
    """

__author__ = 'J. Agustin BARRACHINA'
__version__ = '0.1.34'
__maintainer__ = 'J. Agustin BARRACHINA'
__email__ = 'joseagustin.barra@gmail.com; jose-agustin.barrachina@centralesupelec.fr'
