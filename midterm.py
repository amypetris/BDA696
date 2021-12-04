import os
import sys

import numpy as np
import pandas as pd
import scipy
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from scipy.stats import binned_statistic_2d
from sklearn.metrics import confusion_matrix

import hw4_data
from cat_correlation import cat_cont_correlation_ratio, cat_correlation


def classify_type(data):
    threshold = 0.05
    if len(set(data)) > len(data) * threshold:
        return "con"
    else:
        return "cat"


def plot_cat_response_cat_predictor(
    pred_data,
    pred_name,
    response,
):
    file_name = f"plots/{pred_name}-plot.html"
    pred_data, _ = pd.factorize(pred_data)
    response, _ = pd.factorize(response)
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


def con_con_correlation(pred1_data, pred1_name, pred2_data, pred2_name):
    correlation_stats, _ = scipy.stats.pearsonr(pred1_data, pred2_data)
    return correlation_stats


def con_cat_correlation(
    pred1_data, pred1_name, pred1_type, pred2_data, pred2_name, pred2_type
):
    if len(set(pred1_data)) == 2 or len(set(pred2_data)) == 2:
        pred1_data, _ = pd.factorize(pred1_data)
        correlation_stats, _ = scipy.stats.pointbiserialr(pred1_data, pred2_data)
    elif pred1_type == "cat":
        correlation_stats = cat_cont_correlation_ratio(
            np.array(pred1_data), np.array(pred2_data)
        )
    elif pred2_type == "cat":
        correlation_stats = cat_cont_correlation_ratio(
            np.array(pred2_data), np.array(pred1_data)
        )
    return correlation_stats


def cat_cat_correlation(pred1_data, pred1_name, pred2_data, pred2_name):
    correlation_stats = cat_correlation(pred1_data, pred2_data)
    return correlation_stats


def plot_difference_with_mean_unweighted(
    x1, x2, y, num_bins, pred1_name, pred2_name, pred1_type, pred2_type
):
    file_name = (
        f"diffmeanplots/DiffmeanplotUnweighted - {pred1_name} - {pred2_name}.html"
    )
    if pred1_type == "cat":
        x1, _ = pd.factorize(x1)
    if pred2_type == "cat":
        x2, _ = pd.factorize(x2)
    hist, xbins, ybins = np.histogram2d(x1, x2, bins=num_bins)
    _, x1bins = np.histogram(x1, num_bins)
    _, x2bins = np.histogram(x2, num_bins)
    bin_means, binx_edges, biny_edges, binnumber = binned_statistic_2d(
        x1, x2, y, statistic="mean", bins=[x1bins, x2bins], range=None
    )
    population_mean = np.average(y)
    mean_squared_diff = np.power((bin_means - population_mean), 2)
    fig = px.imshow(mean_squared_diff)
    fig.write_html(
        file=file_name,
        include_plotlyjs="cdn",
        include_mathjax="cdn",
    )

    return file_name


def plot_difference_with_mean_weighted(
    x1, x2, y, num_bins, pred1_name, pred2_name, pred1_type, pred2_type
):
    file_name = f"diffmeanplots/DiffmeanplotWeighted - {pred1_name} - {pred2_name}.html"
    if pred1_type == "cat":
        x1, _ = pd.factorize(x1)
    if pred2_type == "cat":
        x2, _ = pd.factorize(x2)
    hist, xbins, ybins = np.histogram2d(x1, x2, bins=num_bins)
    _, x1bins = np.histogram(x1, num_bins)
    _, x2bins = np.histogram(x2, num_bins)
    bin_means, binx_edges, biny_edges, binnumber = binned_statistic_2d(
        x1, x2, y, statistic="mean", bins=[x1bins, x2bins], range=None
    )
    population_mean = np.average(y)
    pop_proportion = hist / len(y)
    mean_squared_diff = np.power((bin_means - population_mean), 2)

    mean_squared_diff_weighted = mean_squared_diff * pop_proportion
    fig = px.imshow(mean_squared_diff_weighted)
    fig.write_html(
        file=file_name,
        include_plotlyjs="cdn",
        include_mathjax="cdn",
    )
    return file_name


def difference_with_mean_2d(
    x1, x2, y, num_bins, pred1_name, pred2_name, pred1_type, pred2_type
):
    if pred1_type == "cat":
        x1, _ = pd.factorize(x1)
    if pred2_type == "cat":
        x2, _ = pd.factorize(x2)
    hist, xbins, ybins = np.histogram2d(x1, x2, bins=num_bins)
    _, x1bins = np.histogram(x1, num_bins)
    _, x2bins = np.histogram(x2, num_bins)
    bin_means, binx_edges, biny_edges, binnumber = binned_statistic_2d(
        x1, x2, y, statistic="mean", bins=[x1bins, x2bins], range=None
    )
    population_mean = np.average(y)
    pop_proportion = hist / len(y)
    mean_squared_diff = np.power((bin_means - population_mean), 2)
    mean_squared_diff_sum = np.nansum(mean_squared_diff)
    mean_squared_diff_weighted = mean_squared_diff * pop_proportion
    mean_squared_diff_weighted_sum = np.nansum(mean_squared_diff_weighted)
    unweighted_plot = plot_difference_with_mean_unweighted(
        x1, x2, y, 10, pred1_name, pred2_name, pred1_type, pred2_type
    )
    weighted_plot = plot_difference_with_mean_weighted(
        x1, x2, y, 10, pred1_name, pred2_name, pred1_type, pred2_type
    )
    df = pd.DataFrame(
        list(
            zip(
                [pred1_name],
                [pred2_name],
                [pred1_type],
                [pred2_type],
                [population_mean],
                [mean_squared_diff_sum],
                [mean_squared_diff_weighted_sum],
                [unweighted_plot],
                [weighted_plot],
            )
        ),
        columns=[
            "Predictor 1",
            "Predictor 2",
            "Predictor 1 Type",
            "Predictor 2 Type",
            "PopulationMean",
            "MeanSquaredDiff",
            "MeanSquaredDiffWeighted",
            "Unweighted Plot",
            "Weighted Plot",
        ],
    )
    return df


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

    df, predictors, response = hw4_data.get_test_data_set("diabetes")
    X = df[predictors]
    y = df[response]

    response_type = classify_type(y)

    pred1_names = []
    pred2_names = []
    pred1_types = []
    pred2_types = []
    correlation_coeffients = []
    pred1_plots = []
    pred2_plots = []
    diff_mean_dataframes = []

    for i in range(len(X.T) - 1):
        for j in range(i + 1, len(X.T)):
            pred1_name = predictors[i]
            pred2_name = predictors[j]
            pred1_data = X[pred1_name]
            pred2_data = X[pred2_name]
            pred1_type = classify_type(pred1_data)
            pred2_type = classify_type(pred2_data)

            if pred1_type == "con" and pred2_type == "con":
                corr_coef = con_con_correlation(
                    pred1_data, pred1_name, pred2_data, pred2_name
                )
                if response_type == "cat":
                    pred1_plot = plot_cat_resp_con_predictor(pred1_data, pred1_name, y)
                    pred2_plot = plot_cat_resp_con_predictor(pred2_data, pred2_name, y)
                elif response_type == "con":
                    pred1_plot = plot_con_response_con_predictor(
                        pred1_data, pred1_name, y
                    )
                    pred2_plot = plot_con_response_con_predictor(
                        pred2_data, pred2_name, y
                    )
            elif pred1_type == "cat" and pred2_type == "con":
                corr_coef = con_cat_correlation(
                    pred1_data,
                    pred1_name,
                    pred1_type,
                    pred2_data,
                    pred2_name,
                    pred2_type,
                )
                if response_type == "cat":
                    pred1_plot = plot_cat_response_cat_predictor(
                        pred1_data, pred1_name, y
                    )
                    pred2_plot = plot_con_response_con_predictor(
                        pred2_data, pred2_name, y
                    )
                elif response_type == "con":
                    pred1_plot = plot_con_resp_cat_predictor(pred1_data, pred1_name, y)
                    pred2_plot = plot_con_response_con_predictor(
                        pred2_data, pred2_name, y
                    )
            elif pred1_type == "con" and pred2_type == "cat":
                corr_coef = con_cat_correlation(
                    pred1_data,
                    pred1_name,
                    pred1_type,
                    pred2_data,
                    pred2_name,
                    pred2_type,
                )
                if response_type == "cat":
                    pred1_plot = plot_cat_resp_con_predictor(pred1_data, pred1_name, y)
                    pred2_plot = plot_cat_response_cat_predictor(
                        pred2_data, pred2_name, y
                    )
                elif response_type == "con":
                    pred1_plot = plot_con_response_con_predictor(
                        pred1_data, pred1_name, y
                    )
                    pred2_plot = plot_con_resp_cat_predictor(pred2_data, pred2_name, y)
            elif pred1_type == "cat" and pred2_type == "cat":
                categories1 = np.array(list(set(pred1_data)))
                categories2 = np.array(list(set(pred2_data)))
                pred1_data = pd.Categorical(
                    pred1_data, categories=categories1, ordered=True
                )
                pred2_data = pd.Categorical(
                    pred2_data, categories=categories2, ordered=True
                )
                corr_coef = cat_cat_correlation(
                    pred1_data, pred1_name, pred2_data, pred2_name
                )
                if response_type == "cat":
                    pred1_plot = plot_cat_response_cat_predictor(
                        pred1_data, pred1_name, y
                    )
                    pred2_plot = plot_cat_response_cat_predictor(
                        pred2_data, pred2_name, y
                    )
                elif response_type == "con":
                    pred1_plot = plot_con_resp_cat_predictor(pred1_data, pred1_name, y)
                    pred2_plot = plot_con_resp_cat_predictor(pred2_data, pred2_name, y)
            df_diff_mean = difference_with_mean_2d(
                pred1_data,
                pred2_data,
                y,
                4,
                pred1_name,
                pred2_name,
                pred1_type,
                pred2_type,
            )
            diff_mean_dataframes.append(df_diff_mean)
            pred1_names.append(pred1_name)
            pred2_names.append(pred2_name)
            pred1_types.append(pred1_type)
            pred2_types.append(pred2_type)
            correlation_coeffients.append(corr_coef)
            pred1_plots.append(pred1_plot)
            pred2_plots.append(pred2_plot)

    diff_mean_df_concat = pd.concat(diff_mean_dataframes)

    # Build con/con brute force table
    con_con_diff_mean_df = diff_mean_df_concat[
        diff_mean_df_concat["Predictor 1 Type"] == "con"
    ]
    con_con_diff_mean_df = con_con_diff_mean_df[
        con_con_diff_mean_df["Predictor 2 Type"] == "con"
    ]
    con_con_diff_mean_df_sorted = con_con_diff_mean_df.sort_values(
        "MeanSquaredDiffWeighted", ascending=False
    )
    con_con_diff_mean_df_sorted_styled = con_con_diff_mean_df_sorted.style.format(
        {"Unweighted Plot": fun, "Weighted Plot": fun}
    )
    con_con_table = con_con_diff_mean_df_sorted_styled.render()
    text_file = open("con_con_diff_mean.html", "w")
    text_file.write(con_con_table)

    # Build con/con brute force table
    con_cat_correlation_df = diff_mean_df_concat[
        diff_mean_df_concat["Predictor 1 Type"] == "con"
    ]
    con_cat_correlation_df = con_cat_correlation_df[
        con_cat_correlation_df["Predictor 2 Type"] == "cat"
    ]
    cat_con_correlation_df = diff_mean_df_concat[
        diff_mean_df_concat["Predictor 1 Type"] == "cat"
    ]
    cat_con_correlation_df = cat_con_correlation_df[
        cat_con_correlation_df["Predictor 2 Type"] == "con"
    ]
    con_cat_correlation_df_combo = pd.concat(
        [con_cat_correlation_df, cat_con_correlation_df]
    )
    con_cat_diff_mean_df_sorted = con_cat_correlation_df_combo.sort_values(
        "MeanSquaredDiffWeighted", ascending=False
    )
    con_cat_diff_mean_df_sorted_styled = con_cat_diff_mean_df_sorted.style.format(
        {"Unweighted Plot": fun, "Weighted Plot": fun}
    )
    con_cat_table = con_cat_diff_mean_df_sorted_styled.render()
    text_file = open("con_cat_diff_mean.html", "w")
    text_file.write(con_cat_table)

    # build cat/cat brute force table
    cat_cat_diff_mean_df = diff_mean_df_concat[
        diff_mean_df_concat["Predictor 1 Type"] == "cat"
    ]
    cat_cat_diff_mean_df = cat_cat_diff_mean_df[
        cat_cat_diff_mean_df["Predictor 2 Type"] == "cat"
    ]
    cat_cat_diff_mean_df_sorted = cat_cat_diff_mean_df.sort_values(
        "MeanSquaredDiffWeighted", ascending=False
    )
    cat_cat_diff_mean_df_sorted_styled = cat_cat_diff_mean_df_sorted.style.format(
        {"Unweighted Plot": fun, "Weighted Plot": fun}
    )
    cat_cat_table = cat_cat_diff_mean_df_sorted_styled.render()
    text_file = open("cat_cat_diff_mean.html", "w")
    text_file.write(cat_cat_table)

    df = pd.DataFrame(
        list(
            zip(
                pred1_names,
                pred2_names,
                pred1_types,
                pred2_types,
                correlation_coeffients,
                pred1_plots,
                pred2_plots,
            )
        ),
        columns=[
            "Predictor 1 Name",
            "Predictor 2 Name",
            "Predictor 1 Type",
            "Predictor 2 Type",
            "Correlation Coefficient",
            "Predictor 1 Plot",
            "Predictor 2 Plot",
        ],
    )

    # BUILD CON/CON CORRELATION TABLE
    con_con_correlation_df = df[df["Predictor 1 Type"] == "con"]
    con_con_correlation_df = con_con_correlation_df[
        con_con_correlation_df["Predictor 2 Type"] == "con"
    ]
    con_con_correlation_df_sorted = con_con_correlation_df.sort_values(
        "Correlation Coefficient", ascending=False
    )
    con_con_correlation_df_sorted = con_con_correlation_df_sorted.style.format(
        {"Predictor 1 Plot": fun, "Predictor 2 Plot": fun}
    )
    con_con_table = con_con_correlation_df_sorted.render()
    text_file = open("con_con_correlation_table.html", "w")
    text_file.write(con_con_table)

    # BUILD CON/CAT CORRELATION TABLE
    con_cat_correlation_df = df[df["Predictor 1 Type"] == "con"]
    con_cat_correlation_df = con_cat_correlation_df[
        con_cat_correlation_df["Predictor 2 Type"] == "cat"
    ]
    cat_con_correlation_df = df[df["Predictor 1 Type"] == "cat"]
    cat_con_correlation_df = cat_con_correlation_df[
        cat_con_correlation_df["Predictor 2 Type"] == "con"
    ]
    con_cat_correlation_df_combo = pd.concat(
        [con_cat_correlation_df, cat_con_correlation_df]
    )
    con_cat_correlation_df_sorted = con_cat_correlation_df_combo.sort_values(
        "Correlation Coefficient", ascending=False
    )
    con_cat_correlation_df_sorted = con_cat_correlation_df_sorted.style.format(
        {"Predictor 1 Plot": fun, "Predictor 2 Plot": fun}
    )
    con_cat_table = con_cat_correlation_df_sorted.render()
    text_file = open("con_cat_correlation_table.html", "w")
    text_file.write(con_cat_table)

    # BUILD CAT/CAT CORRELATION TABLE
    cat_cat_correlation_df = df[df["Predictor 1 Type"] == "cat"]
    cat_cat_correlation_df = cat_cat_correlation_df[
        cat_cat_correlation_df["Predictor 2 Type"] == "cat"
    ]
    cat_cat_correlation_df_sorted = cat_cat_correlation_df.sort_values(
        "Correlation Coefficient", ascending=False
    )
    cat_cat_correlation_df_sorted = cat_cat_correlation_df_sorted.style.format(
        {"Predictor 1 Plot": fun, "Predictor 2 Plot": fun}
    )
    cat_cat_table = cat_cat_correlation_df_sorted.render()
    text_file = open("cat_cat_correlation_table.html", "w")
    text_file.write(cat_cat_table)

    # CORRELATION MATRICES
    pred_types = [(pred_name, classify_type(X[pred_name])) for pred_name in predictors]
    con_predictors = [x for (x, y) in pred_types if y == "con"]
    con_data = X[con_predictors]
    cat_predictors = [x for (x, y) in pred_types if y == "cat"]
    cat_data = X[cat_predictors]

    # cat/con
    correlation_matrix = X.corr()
    cat_con_fig = px.imshow(correlation_matrix)
    cat_con_fig.write_html(
        file="plots/cat_con_correlation_matrix_plot.html",
        include_plotlyjs="cdn",
        include_mathjax="cdn",
    )

    # con/con
    correlation_matrix_con = con_data.corr()
    con_con_fig = px.imshow(correlation_matrix_con)
    con_con_fig.write_html(
        file="plots/con_con_correlation_matrix_plot.html",
        include_plotlyjs="cdn",
        include_mathjax="cdn",
    )

    # cat/cat
    correlation_matrix_cat = cat_data.corr()
    cat_cat_fig = px.imshow(correlation_matrix_cat)
    cat_cat_fig.write_html(
        file="plots/cat_cat_correlation_matrix_plot.html",
        include_plotlyjs="cdn",
        include_mathjax="cdn",
    )

    correlation_matrix_df = pd.DataFrame(
        list(
            zip(
                ["plots/con_con_correlation_matrix_plot.html"],
                ["plots/con_con_correlation_matrix_plot.html"],
                ["plots/cat_cat_correlation_matrix_plot.html"],
            )
        ),
        columns=[
            "Cat/Con Correlation Matrix",
            "Con/Con Correlation Matrix",
            "Cat/Cat Correlation Matrix",
        ],
    )
    correlation_matrix_df_styled = correlation_matrix_df.style.format(
        {
            "Cat/Con Correlation Matrix": fun,
            "Con/Con Correlation Matrix": fun,
            "Cat/Cat Correlation Matrix": fun,
        }
    )
    correlation_matrix_table = correlation_matrix_df_styled.render()
    text_file = open("correlation_matrices.html", "w")
    text_file.write(correlation_matrix_table)

    return


if __name__ == "__main__":
    sys.exit(main())
