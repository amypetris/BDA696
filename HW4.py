import os
import sys

import numpy as np
import pandas as pd
import scipy.stats as ss
import sklearn.datasets
import statsmodels.api
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from scipy.stats import binned_statistic
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix


def classify_type(data):
    if len(set(data)) != 2:
        return "con"
    else:
        return "cat"


def plot_cat_response_cat_predictor(
    pred_data,
    pred_name,
    response,
):
    file_name = f"plots/{pred_name}-plot.html"
    conf_matrix = confusion_matrix(pred_data, response)
    fig = go.Figure(data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max()))
    fig.update_layout(
        title="Categorical Predictor by Categorical Response",
        xaxis_title="Response",
        yaxis_title=pred_name,
    )
    fig.write_html(
        file=file_name,
        include_plotlyjs="cdn",
        include_mathjax="cdn",
    )
    return file_name
    # fig.show()


def plot_con_resp_cat_predictor(pred_data, pred_name, response):
    file_name = f"plots/{pred_name}-plot.html"
    fig = go.Figure()
    for curr_hist, curr_group in zip(pred_data, response):
        fig.add_trace(
            go.Violin(
                x=np.repeat(curr_group, 200),
                y=[curr_hist],
                name=curr_group,
                box_visible=True,
                meanline_visible=True,
            )
        )
    fig.update_layout(
        title="Continuous Response by Categorical Predictor",
        xaxis_title=pred_name,
        yaxis_title="Response",
    )
    fig.write_html(
        file=file_name,
        include_plotlyjs="cdn",
        include_mathjax="cdn",
    )

    return file_name


def plot_cat_resp_con_predictor(pred_data, pred_name, response):
    file_name = f"plots/{pred_name}-plot.html"
    group_labels = list(set([int(x) for x in response]))
    hist_data0 = pred_data[response == group_labels[0]]
    hist_data1 = pred_data[response == group_labels[1]]
    hist_data = [hist_data0, hist_data1]
    fig = ff.create_distplot(hist_data, group_labels, bin_size=0.2)
    fig.update_layout(
        title="Continuous Predictor by Categorical Response",
        xaxis_title=pred_name,
        yaxis_title="Distribution",
    )

    fig.write_html(
        file=file_name,
        include_plotlyjs="cdn",
        include_mathjax="cdn",
    )

    return file_name


def plot_con_response_con_predictor(pred_data, pred_name, response):
    file_name = f"plots/{pred_name}-plot.html"
    fig = px.scatter(x=pred_data, y=response, trendline="ols")
    fig.update_layout(
        title="Continuous Response by Continuous Predictor",
        xaxis_title=pred_name,
        yaxis_title="Response",
    )

    fig.write_html(
        file=file_name,
        include_plotlyjs="cdn",
        include_mathjax="cdn",
    )

    return file_name


def lin_regression(column, pred_name, response):
    predictor = statsmodels.api.add_constant(column)
    linear_regression_model = statsmodels.api.OLS(response, predictor)
    linear_regression_model_fitted = linear_regression_model.fit()

    # Get the stats
    t_value = round(linear_regression_model_fitted.tvalues[1], 6)
    p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])

    # Plot the figure
    fig = px.scatter(x=column, y=response, trendline="ols")
    fig.update_layout(
        title=f"Variable: {pred_name}: (t-value={t_value}) (p-value={p_value})",
        xaxis_title=f"Variable: {pred_name}",
        yaxis_title="y",
    )
    return t_value, p_value


def log_regression(column, pred_name, response):
    predictor = statsmodels.api.add_constant(column)
    logistic_regression_model = statsmodels.api.Logit(response, predictor)
    logistic_regression_model_fitted = logistic_regression_model.fit()

    # Get the stats
    t_value = round(logistic_regression_model_fitted.tvalues[1], 6)
    p_value = "{:.6e}".format(logistic_regression_model_fitted.pvalues[1])
    return t_value, p_value


def plot_difference_with_mean(x, y, num_bins, pred_name):
    file_name = f"diffmeanplots/Diffmeanplot - {pred_name}.html"
    hist, bins = np.histogram(x, num_bins)
    bin_means, bin_edges, binnumber = binned_statistic(
        x, y, statistic="mean", bins=10, range=None
    )
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[1:] - bin_width / 2
    population_mean = np.average(y)
    data = [
        # The population Bar plot
        go.Bar(x=bin_centers, y=hist, yaxis="y2", name="Population", opacity=0.5),
        # The bin mean plot
        go.Scatter(
            x=bin_centers, y=bin_means, name="Binned Difference with Mean"  # noqa W605
        ),
        # The population average plot
        go.Scatter(
            x=[np.min(x), np.max(x)],
            y=[population_mean, population_mean],
            name="Population Mean",  # noqa W605
            mode="lines",
        ),
    ]

    layout = go.Layout(
        title=f"Difference with mean of response - {pred_name}",
        xaxis_title="Predictor Bin",
        yaxis_title="Response",
        yaxis2=dict(title="Population", overlaying="y", anchor="y3", side="right"),
    )

    fig = go.Figure(data=data, layout=layout)
    fig.write_html(
        file=file_name,
        include_plotlyjs="cdn",
        include_mathjax="cdn",
    )
    return file_name


def table_difference_with_mean(x, y, num_bins, pred_name):
    file_name = f"diffmeantables/Difference with mean - {pred_name}.html"
    hist, bins = np.histogram(x, num_bins)
    # binned_statistic() function shown to me by Jue Li
    bin_means, bin_edges, binnumber = binned_statistic(
        x, y, statistic="mean", bins=10, range=None
    )
    bin_centers = [(bin_edges[i] - bin_edges[i + 1]) / 2 for i in range(num_bins)]
    population_mean = np.average(y)
    lower_bound = [bins[i] for i in range(1, num_bins + 1)]
    upper_bound = [bins[i] for i in range(1, num_bins + 1)]
    pop_proportion = hist / len(y)
    mean_squared_diff = np.power((bin_means - population_mean), 2)
    mean_squared_diff_weighted = mean_squared_diff * pop_proportion

    # Outputs to help the slides
    df = pd.DataFrame(
        list(
            zip(
                lower_bound,
                upper_bound,
                bin_centers,
                hist,
                bin_means,
                np.repeat(population_mean, num_bins),
                mean_squared_diff,
                pop_proportion,
                mean_squared_diff_weighted,
            )
        ),
        columns=[
            "LowerBin",
            "UpperBin",
            "BinCenters",
            "BinCount",
            "BinMeans",
            "PopulationMean",
            "MeanSquaredDiff",
            "PopulationProportion",
            "MeanSquaredDiffWeighted",
        ],
    )
    tab = df.to_html()
    text_file = open(file_name, "w")
    text_file.write(tab)
    return file_name


def random_forest_importance_ranking(X, y):
    random_forest = RandomForestClassifier(random_state=1234)
    random_forest.fit(X, y)
    result = permutation_importance(
        random_forest, X, y, n_repeats=10, random_state=1234
    )
    importances = result["importances_mean"]
    importance_rank = ss.rankdata(importances)
    return 10 - importance_rank


def fun(path):
    f_url = os.path.basename(path)
    return '<a href="{}">{}</a>'.format(path, f_url)


def main():
    # create folders used later to store plots and tables
    if not os.path.isdir("plots"):
        os.mkdir("plots")
    if not os.path.isdir("diffmeanplots"):
        os.mkdir("diffmeanplots")
    if not os.path.isdir("diffmeantables"):
        os.mkdir("diffmeantables")

    df = sklearn.datasets.load_breast_cancer()
    X = df.data
    y = df.target
    response_type = classify_type(y)

    predictor_names = []
    predictor_types = []
    plots = []
    t_values = []
    p_values = []
    diff_mean_plots = []
    diff_mean_tables = []

    # classify predictors as categorical or continuous
    for idx, column in enumerate(X.T):
        pred_name = df.feature_names[idx]
        pred_type = classify_type(column)
        predictor_names.append(pred_name)
        predictor_types.append(pred_type)

    # loop through predictors are do appropriate analysis for each type
    for idx, pred in enumerate(predictor_names):
        x_type = predictor_types[idx]
        y_type = response_type
        x_data = X.T[idx]
        response = y
        pred_name = pred
        if x_type == "cat" and y_type == "cat":
            plot_link = plot_cat_response_cat_predictor(x_data, pred_name, response)
            t_value = 0
            p_value = 0
        elif x_type == "cat" and y_type == "con":
            plot_link = plot_con_resp_cat_predictor(x_data, pred_name, response)
            t_value = 0
            p_value = 0
        elif x_type == "con" and y_type == "cat":
            plot_link = plot_cat_resp_con_predictor(x_data, pred_name, response)
            t_value, p_value = log_regression(x_data, pred_name, response)
        elif x_type == "con" and y_type == "con":
            plot_link = plot_con_response_con_predictor(x_data, pred_name, response)
            t_value, p_value = lin_regression(x_data, pred_name, response)
        plot2_link = plot_difference_with_mean(x_data, y, 10, pred_name)
        table_link = table_difference_with_mean(x_data, y, 10, pred_name)
        plots.append(plot_link)
        t_values.append(t_value)
        p_values.append(p_value)
        diff_mean_plots.append(plot2_link)
        diff_mean_tables.append(table_link)

    rank = random_forest_importance_ranking(X, y)

    # put plots, tables, etc in table for final report
    df = pd.DataFrame(
        list(
            zip(
                predictor_names,
                predictor_types,
                plots,
                t_values,
                p_values,
                diff_mean_plots,
                diff_mean_tables,
                rank,
            )
        ),
        columns=[
            "Name",
            "Type",
            "Plot",
            "t-score",
            "p-value",
            "diff mean plots",
            "diff mean tables",
            "rank",
        ],
    )
    df_sorted = df.sort_values("rank")
    df_final = df_sorted.style.format(
        {"Plot": fun, "diff mean plots": fun, "diff mean tables": fun}
    )

    final_report = df_final.to_html()

    w_file = open("Assignment4.html", "w")
    w_file.write(final_report)


if __name__ == "__main__":
    sys.exit(main())
