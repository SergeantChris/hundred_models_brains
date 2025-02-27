import math
from copy import deepcopy
import ast
from time import sleep
from functools import partial
import warnings
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import to_rgb
import seaborn as sns
from repralign.datasets.constants.bmd import BMDConstants
from repralign.utils.common import residuals
from repralign.utils.statistical_significance import compute_significance_against_zero, compute_pairwise_significance

ROI_GROUPS = BMDConstants.roi_groups


def idx_transform_patch_to_df(idx, n_hue, n_x):
    return n_hue * (idx % n_x) + idx // n_x


def idx_transform_df_to_patch(idx, n_hue):
    return idx // n_hue + idx % n_hue


def convert_null_distrib_dict_to_df(d):
    rows = []
    for model_name, layers_rois in d.items():
        for layer_roi_names, null_distrib in layers_rois.items():
            layer_name, roi_name = layer_roi_names.split("_fmri_")
            row = {"Model": model_name, "Layer_FullNames": layer_name, "ROI": roi_name, "NullDistrib": null_distrib}
            rows.append(row)
    return pd.DataFrame(rows)


def get_max_corr_layer_df(plotter):
    max_dataframes = []
    for dataframe in plotter.dataframes:
        dataframe = plotter.prepare_dataframe(dataframe, "R")
        max_dataframes.append(dataframe.loc[dataframe.groupby("ROI")["R"].idxmax()])
    return pd.concat(max_dataframes, ignore_index=True)


def sort_by_roi_and_model_order(plotting_df):
    model_order = plotting_df["Model"].unique()
    plotting_df["CustomOrder"] = plotting_df["Model"].map({model: i for i, model in enumerate(model_order)})
    plotting_df = plotting_df.sort_values(by=["ROI", "CustomOrder"]).reset_index(drop=True)
    plotting_df.drop("CustomOrder", axis=1, inplace=True)
    return plotting_df


def plot_base(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("tab10", n_colors=len(df["Model"].unique()))
    sns.barplot(data=df, x="ROI", y="R", hue="Model", palette=palette, alpha=0.6, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)

    n_rois = df["ROI"].nunique()
    for index, row in df.iterrows():
        patch = ax.patches[index * n_rois % len(ax.patches) + index * n_rois // len(ax.patches)]
        x = patch.get_x()
        width = patch.get_width()
        x_middle = x + width / 2
        y = patch.get_height()
        lower_limit = max(y - row["SEM"], 0)
        upper_limit = y + row["SEM"]
        ax.errorbar(x_middle, y, yerr=[[y - lower_limit], [upper_limit - y]], fmt=" ", c="k", capsize=width)
        if row["pvalue"] < 0.05:
            ax.text(x_middle, upper_limit, "*", ha="center", va="bottom", c="k")
        if np.isnan(row["LNC"]) or np.isnan(row["UNC"]):
            continue
        ax.hlines(y=row["LNC"], xmin=x, xmax=x + width, linewidth=1, color="k", linestyle="dashed")
        ax.hlines(y=row["UNC"], xmin=x, xmax=x + width, linewidth=1, color="k", linestyle="dashed")
    plt.subplots_adjust(right=0.85)
    plt.show()


def get_roi_mg_sigzero_sigpairs(df, group_name, layers=False):
    if "NullDistrib" not in df.columns:
        raise ValueError("The dataframe does not contain the null distributions")
    roi_mg_pairs = {}
    for roi in df["ROI"].unique():
        roi_df = df[df["ROI"] == roi]
        mg_null_distribs = {}
        for mg in roi_df[group_name].unique():
            mg_df = roi_df[roi_df[group_name] == mg]
            if not layers:
                mg_null_distrib = np.median(np.stack(mg_df["NullDistrib"].values), axis=0)
                median_r = np.median(mg_df["R"].values)
                df.loc[(df["ROI"] == roi) & (df[group_name] == mg), "mg_pvalue"] = compute_significance_against_zero(
                    mg_null_distrib, median_r
                )
            else:
                raise NotImplementedError("Layers are not supported yet in computing model group significance scores")
            mg_null_distribs[mg] = mg_null_distrib
        if "Kinetics-710" in mg_null_distribs.keys():
            bonferroni = 1
        else:
            bonferroni = math.comb(len(mg_null_distribs), 2)
        mg_pairs = {}
        for mg in roi_df[group_name].unique():
            for mg_other in roi_df[group_name].unique():
                if mg == mg_other:
                    continue
                null_distrib1 = mg_null_distribs[mg]
                null_distrib2 = mg_null_distribs[mg_other]
                median_r1 = np.median(roi_df[roi_df[group_name] == mg]["R"].values)
                median_r2 = np.median(roi_df[roi_df[group_name] == mg_other]["R"].values)
                mg_pairs[(mg, mg_other)] = (
                    compute_pairwise_significance(null_distrib1, null_distrib2, median_r1, median_r2) * bonferroni
                )
        roi_mg_pairs[roi] = mg_pairs
    return df, roi_mg_pairs


def change_luminance(color, factor):
    rgb = np.array(to_rgb(color))
    adjusted = np.clip(rgb * factor, 0, 1)
    return adjusted


def plot_model_groups(df, group_name, ax):
    if "NullDistrib" in df.columns:
        df, roi_mg_pairs = get_roi_mg_sigzero_sigpairs(df, group_name)
    n_model_groups = df[group_name].nunique()
    if group_name == "ArchType":
        palette = sns.color_palette("tab10")[3:5]
    elif group_name == "ModelTypeFolder":
        custom_shades = []
        for color in sns.color_palette("tab10"):
            custom_shades.extend([change_luminance(color, 1.66), change_luminance(color, 1.33), color])
        palette = sns.color_palette(custom_shades, n_colors=n_model_groups)
    else:
        palette = sns.color_palette("tab10", n_colors=n_model_groups)
        # palette = [
        #     sns.color_palette("tab10")[-2],
        #     sns.color_palette("tab10")[-3],
        #     sns.color_palette("tab10")[-2],
        #     sns.color_palette("tab10")[-1],
        # ]
    sns.boxplot(
        data=df,
        x="ROI",
        y="R",
        hue=group_name,
        palette=palette,
        whis=0,
        fliersize=0,
        showcaps=False,
        medianprops={"color": "gray"},
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x="ROI",
        y="R",
        hue=group_name,
        palette=palette,
        dodge=True,
        jitter=True,
        alpha=0.4,
        edgecolor="gray",
        linewidth=1,
        legend=False,
        ax=ax,
    )
    df_roi_mg = (
        df.groupby(["ROI", group_name])
        .mean()
        .reset_index()
        .sort_values(by=["CustomROIOrder", "CustomModelGroupOrder"])
        .reset_index(drop=True)
    )
    pathpatches = [patch for patch in ax.patches if isinstance(patch, PathPatch)]
    for i, patch in enumerate(pathpatches):
        facecolor = patch.get_facecolor()
        patch.set_edgecolor(facecolor)
        patch.set_facecolor("none")
        vertices = patch.get_path().vertices
        row = df_roi_mg.iloc[i]
        prev_row = df_roi_mg.iloc[max(0, i - 1)]
        if row[group_name] == "Kinetics-400 (2)" or prev_row[group_name] == "Kinetics-400 (2)":
            x_left_bottom = deepcopy(vertices[0][0])
            vertices[:, 0] += 0.03
            y_left_bottom = vertices[0][1]
            y_left_top = vertices[3][1]
            for line in ax.lines:
                x_data = line.get_xdata()
                y_data = line.get_ydata()
                if y_data[0] > y_left_bottom and y_data[1] < y_left_top:
                    if np.isclose(x_data[0], x_left_bottom):
                        line.set_xdata([x_data[0] + 0.03, x_data[1] + 0.03])
        if row[group_name] == "Kinetics-400 (1)" or prev_row[group_name] == "Kinetics-400 (1)":
            x_left_bottom = deepcopy(vertices[0][0])
            vertices[:, 0] -= 0.03
            y_left_bottom = vertices[0][1]
            y_left_top = vertices[3][1]
            for line in ax.lines:
                x_data = line.get_xdata()
                y_data = line.get_ydata()
                if y_data[0] > y_left_bottom and y_data[1] < y_left_top:
                    if np.isclose(x_data[0], x_left_bottom):
                        line.set_xdata([x_data[0] - 0.03, x_data[1] - 0.03])
        x = vertices[0][0]
        width = vertices[1][0] - x
        x_middle = x + width / 2
        if "NullDistrib" in df.columns:
            if row["mg_pvalue"] < 0.05:
                ax.text(x_middle, 0.002, "*", ha="center", va="bottom", c="k")
        ax.hlines(y=row["LNC"], xmin=x, xmax=x + width, linewidth=1, color="k", linestyle="dashed")
        if "UNC" in df_roi_mg.columns:
            ax.hlines(y=row["UNC"], xmin=x, xmax=x + width, linewidth=1, color="k", linestyle="dashed")
    if "NullDistrib" in df.columns:
        for roi in roi_mg_pairs.keys():
            deduplicated_roi_mg_pairs = {tuple(sorted(key)): value for key, value in roi_mg_pairs[roi].items()}
            for mg_pair, pvalue in deduplicated_roi_mg_pairs.items():
                if pvalue < 0.05:
                    mg1, mg2 = mg_pair
                    if mg1.endswith("(1)") and mg2.endswith("(2)"):
                        continue
                    if mg1 == "Sth-Sth-v2" and mg2 == "Kinetics-710":
                        continue
                    if mg1.endswith("(1)") and mg2 == "Sth-Sth-v2":
                        continue
                    if mg1.endswith("(2)") and mg2 == "Kinetics-710":
                        continue
                    [i1] = df_roi_mg.index[(df_roi_mg["ROI"] == roi) & (df_roi_mg[group_name] == mg1)].values
                    [i2] = df_roi_mg.index[(df_roi_mg["ROI"] == roi) & (df_roi_mg[group_name] == mg2)].values
                    vertices1 = pathpatches[i1].get_path().vertices
                    vertices2 = pathpatches[i2].get_path().vertices
                    x1 = vertices1[0][0] + (vertices1[1][0] - vertices1[0][0]) / 2
                    x2 = vertices2[0][0] + (vertices2[1][0] - vertices2[0][0]) / 2
                    y = max(vertices1[2][1], vertices2[2][1]) + 0.1
                    ax.plot([x1, x1, x2, x2], [y - 0.02, y, y, y - 0.02], lw=1, c="k", alpha=0.33)
                    if pvalue < 0.001:
                        ax.text((x1 + x2) / 2 - 0.004, y + 0.01, "***", ha="left", va="bottom", c="k", rotation=90)
                    elif pvalue < 0.01:
                        ax.text((x1 + x2) / 2 - 0.004, y + 0.01, "**", ha="left", va="bottom", c="k", rotation=90)
                    else:
                        ax.text((x1 + x2) / 2, y + 0.002, "*", ha="center", va="bottom", c="k")
    ax.set_ylim(bottom=0)
    ax.grid(False)
    xticklabels = ax.get_xticklabels()
    for label in xticklabels:
        if label.get_text() == "IPS1":
            label.set_text("IPS1-3")
    ax.set_xticklabels(xticklabels, fontsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.set_ylabel("")
    ax.set_xlabel("")
    return ax


def bmd_plot_every_model(plotter):
    plotting_df = get_max_corr_layer_df(plotter)
    plotting_df["R_array"] = plotting_df["R_array"].apply(ast.literal_eval)
    plotting_df = plotting_df.explode("R_array", ignore_index=True)
    rois = []
    for group in ROI_GROUPS.values():
        rois.extend(group)
    for i, roi in enumerate(rois):
        roi_df = plotting_df[plotting_df["ROI"] == roi].reset_index(drop=True)
        roi_df = roi_df.sort_values(by=["R"]).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(11, 8))
        palette = sns.color_palette("tab10", n_colors=len(roi_df["ModelType"].unique()))
        palette = {"image-in1k": palette[0], "image-k400": palette[1], "video-k400": palette[2]}
        image_in1k_medianR = roi_df[roi_df["ModelType"] == "image-in1k"]["R"].median()
        image_in1k_upperR = roi_df[roi_df["ModelType"] == "image-in1k"]["R"].quantile(0.75)
        image_in1k_lowerR = roi_df[roi_df["ModelType"] == "image-in1k"]["R"].quantile(0.25)
        plt.axhline(y=image_in1k_medianR, color=palette["image-in1k"], linestyle="-", label="image-in1k", linewidth=2)
        plt.axhline(y=image_in1k_upperR, color=palette["image-in1k"], linestyle="-", linewidth=1)
        plt.axhline(y=image_in1k_lowerR, color=palette["image-in1k"], linestyle="-", linewidth=1)
        roi_df = roi_df[roi_df["ModelType"] != "image-in1k"]
        sns.boxplot(
            data=roi_df, x="Model", y="R_array", hue="ModelType", palette=palette, width=0.8, dodge=False, ax=ax
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=270, fontsize=14)
        plt.yticks(fontsize=14)
        ax.grid(False)
        pathpatches = [patch for patch in ax.patches if isinstance(patch, PathPatch)]
        for i, patch in enumerate(pathpatches):
            vertices = patch.get_path().vertices
            x = vertices[0][0]
            width = vertices[1][0] - x
            x_middle = x + width / 2
            row = roi_df.iloc[i]
            if row["pvalue"] < 0.05:
                ax.text(x_middle, 0.002, "*", ha="center", va="bottom", c="k")
        plt.axhline(y=row["LNC"], linewidth=1, color="k", linestyle="--")
        plt.axhline(y=row["UNC"], linewidth=1, color="k", linestyle="--")
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Spearman's rho", fontsize=20)
        ax.set_xlabel("")
        ax.set_title(roi, fontsize=20)
        handles, labels = plt.gca().get_legend_handles_labels()
        ordered_labels = list(palette.keys())
        ordered_handles = [handles[labels.index(label)] for label in ordered_labels]
        labels = ["Image Models (ImageNet-1k)", "Image Models (Kinetics-400)", "Video Models (Kinetics-400)"]
        plt.legend(ordered_handles, labels, loc="upper right", fontsize=14)
        plt.subplots_adjust(bottom=0.3, right=0.95, top=0.93, left=0.1)
        plt.show()
        # plt.savefig(f"all_models_{roi}.pdf", format="pdf")


def plot_against_continuous_variable(df, continuous_variable):
    g = sns.lmplot(data=df, x=continuous_variable, y="R", col="ROI", col_wrap=2)
    g.set_ylabels("Spearman's rho", fontsize=22)
    g.set_xlabels(continuous_variable, fontsize=22)
    for ax in g.axes.flat:
        ax.grid(False)
        ax.set_ylim(bottom=0, top=0.3)
        ax.set_title(ax.get_title(), fontsize=20)
        ax.set_xticklabels([int(t) for t in ax.get_xticks()], fontsize=18)
        ax.set_yticklabels([round(t, 2) for t in ax.get_yticks()], fontsize=18)
        col_value = ax.get_title().split("= ")[-1]
        roi_df = df[df["ROI"] == col_value]
        corr, p = stats.pearsonr(roi_df[continuous_variable], roi_df["R"])
        ax.text(0.05, 0.95, f"Correlation = {corr:.2f}, p = {p:.4f}", transform=ax.transAxes, fontsize=18)
    plt.tight_layout(rect=(0.02, 0.02, 1, 0.98))
    plt.show()
    # plt.savefig("best_{cont_var}.pdf", format="pdf")
    g = sns.lmplot(data=df, x=continuous_variable, y="R", scatter=False, ci=90, hue="ROI", palette="tab20", height=6)
    g.set_ylabels("Spearman's rho")
    g.set_xlabels(continuous_variable)
    ax = g.ax
    ax.grid(False)
    ax.set_ylim(top=0.34)
    ax.set_xlabel(ax.get_xlabel(), fontsize=20)
    ax.set_ylabel(ax.get_ylabel(), fontsize=20)
    ax.set_xticklabels([int(t) for t in ax.get_xticks()], fontsize=18)
    ax.set_yticklabels([round(t, 2) for t in ax.get_yticks()], fontsize=18)
    plt.tight_layout(rect=(0.02, 0.02, 0.85, 1))
    g._legend.get_title().set_fontsize(20)
    for text in g._legend.texts:
        text.set_fontsize(18)
    plt.show()
    # plt.savefig(f"all_{cont_var}.pdf", format="pdf")


def bmd_plot_per_roi_group(
    plotter, normUNC=False, model_groups=None, all_models_null_distribs=None, continuous_variables=()
):
    plotting_df = get_max_corr_layer_df(plotter)
    model_order = plotting_df["Model"].unique()
    roi_names = plotting_df["ROI"].unique()

    if sum([not s.startswith("r") and not s.startswith("l") for s in roi_names]) == 2:
        right_left = plotting_df[
            plotting_df["ROI"].str.startswith("r") | plotting_df["ROI"].str.startswith("l")
        ].reset_index(drop=True)
        rest_df = plotting_df[
            ~plotting_df["ROI"].str.startswith("r") & ~plotting_df["ROI"].str.startswith("l")
        ].reset_index(drop=True)
        right_left["ROI"] = right_left["ROI"].str[1:]
        rl_mean = right_left.groupby(["ROI", "Model"]).mean().reset_index()
        plotting_df = pd.concat([rl_mean, rest_df]).reset_index(drop=True)
    if "V1d" in plotting_df["ROI"].values:
        early_divided = plotting_df[plotting_df["ROI"].isin(["V1d", "V1v", "V2d", "V2v"])].reset_index(drop=True)
        rest_df = plotting_df[~plotting_df["ROI"].isin(["V1d", "V1v", "V2d", "V2v"])].reset_index(drop=True)
        early_divided["ROI"] = early_divided["ROI"].str[:-1]
        early_mean = early_divided.groupby(["ROI", "Model"]).mean().reset_index()
        plotting_df = pd.concat([early_mean, rest_df]).reset_index(drop=True)

    if "ModelAccuracy" in continuous_variables:
        plt.scatter(plotting_df["ModelAccuracy"], plotting_df["ModelParams"])
        corr, p = stats.pearsonr(plotting_df["ModelAccuracy"], plotting_df["ModelParams"])
        plt.text(0.4, 0.9, f"Correlation: {corr:.2f}, p-value: {p}", transform=plt.gca().transAxes)
        plt.show()
        plt.scatter(plotting_df["ModelAccuracy"], plotting_df["ModelFlops"])
        corr, p = stats.pearsonr(plotting_df["ModelAccuracy"], plotting_df["ModelFlops"])
        plt.text(0.4, 0.9, f"Correlation: {corr:.2f}, p-value: {p}", transform=plt.gca().transAxes)
        plt.show()
        acc_p, r_p = residuals(
            plotting_df["ModelAccuracy"].values.reshape(-1, 1),
            plotting_df["R"].values.reshape(-1, 1),
            plotting_df["ModelParams"].values.reshape(-1, 1),
        )
        acc_f, r_f = residuals(
            plotting_df["ModelAccuracy"].values.reshape(-1, 1),
            plotting_df["R"].values.reshape(-1, 1),
            plotting_df["ModelFlops"].values.reshape(-1, 1),
        )
        acc_pf, r_pf = residuals(
            plotting_df["ModelAccuracy"].values.reshape(-1, 1),
            plotting_df["R"].values.reshape(-1, 1),
            np.stack([plotting_df["ModelParams"].values, plotting_df["ModelFlops"].values], axis=1),
        )
        plotting_df["Accuracy~Params"] = acc_p.squeeze()
        plotting_df["Accuracy~Flops"] = acc_f.squeeze()
        plotting_df["Accuracy~Params+Flops"] = acc_pf.squeeze()
        plotting_df["R~Params"] = r_p.squeeze()
        plotting_df["R~Flops"] = r_f.squeeze()
        plotting_df["R~Params+Flops"] = r_pf.squeeze()
        continuous_variables = list(continuous_variables) + [
            "Accuracy~Params",
            "Accuracy~Flops",
            "Accuracy~Params+Flops",
        ]

    if all_models_null_distribs is not None:
        all_models_null_distribs = convert_null_distrib_dict_to_df(all_models_null_distribs)
        plotting_df = plotting_df.merge(all_models_null_distribs, on=["Model", "Layer_FullNames", "ROI"])

    if normUNC:
        plotting_df["R"] = plotting_df["R"] / plotting_df["UNC"]
        plotting_df["LNC"] = plotting_df["LNC"] / plotting_df["UNC"]
        plotting_df.drop("UNC", axis=1, inplace=True)
    if model_groups is not None:
        roi_groups = len(ROI_GROUPS.keys())
        rows = min(2, roi_groups)
        columns = np.ceil(roi_groups / rows).astype(int)
        fig, axes = plt.subplots(rows, columns, figsize=(7 * rows, 4.2 * columns), squeeze=False, sharey="row")
        # fig, axes = plt.subplots(rows, columns, figsize=(7 * rows, 4.2 * columns), squeeze=False, sharey="row")
        # fig, axes = plt.subplots(rows, columns, figsize=(8.5 * rows, 5 * columns), squeeze=False, sharey="row")
        axes = axes.flatten("F")
    for i, group in enumerate(ROI_GROUPS.values()):
        group_df = plotting_df[plotting_df["ROI"].isin(group)].reset_index(drop=True)
        group_df["CustomROIOrder"] = group_df["ROI"].map({roi: i for i, roi in enumerate(group)})
        group_df["CustomModelOrder"] = group_df["Model"].map({model: i for i, model in enumerate(model_order)})
        group_df = group_df.sort_values(by=["CustomROIOrder", "CustomModelOrder"]).reset_index(drop=True)
        print(group_df.groupby("ROI", group_keys=False).apply(lambda x: x.nlargest(5, "R"))[["ROI", "Model"]])
        print(
            group_df.groupby("ROI", group_keys=False).apply(lambda x: x[x["R"] > x["R"].quantile(0.93)])[
                ["ROI", "Model"]
            ]
        )
        for j, cont_var in enumerate(continuous_variables):
            plot_against_continuous_variable(group_df, cont_var)
            sleep(1.5**j)
        if model_groups is not None:
            mg_order = group_df[model_groups].unique()
            group_df["CustomModelGroupOrder"] = group_df[model_groups].map({mg: i for i, mg in enumerate(mg_order)})
            axes[i] = plot_model_groups(group_df, model_groups, axes[i])
        else:
            plot_base(group_df)
    if model_groups is not None:
        fig.canvas.draw()
        ylabel = "Normalized Spearman's rho" if normUNC else "Spearman's rho"
        fig.supylabel(ylabel, fontsize=22)
        fig.supxlabel("Brain Regions", fontsize=22, x=0.55)
        handles, labels = axes[0].get_legend_handles_labels()
        if model_groups == "ModelType":
            label_pretty = {
                "image-in1k": "Image Models (ImageNet-1k)",
                "image-k400": "Image Models (Kinetics-400)",
                "video-k400": "Video Models (Kinetics-400)",
            }
        elif model_groups == "ModelTypeFolder":
            label_pretty = {
                "image-in1k-models_no_dimred": "Image Models (ImageNet-1k) - no dim. reduction",
                "image-in1k-models": "Image Models (ImageNet-1k) - SRP",
                "image-in1k-models_pca_rerun": "Image Models (ImageNet-1k) - PCA",
                "image-k400-models_no_dimred": "Image Models (Kinetics-400) - no dim. reduction",
                "image-k400-models": "Image Models (Kinetics-400) - SRP",
                "image-k400-models_pca_rerun": "Image Models (Kinetics-400) - PCA",
                "video-k400-models_no_dimred": "Video Models (Kinetics-400) - no dim. reduction",
                "video-k400-models": "Video Models (Kinetics-400) - SRP",
                "video-k400-models_pca_rerun": "Video Models (Kinetics-400) - PCA",
            }
        else:
            label_pretty = None
        labels = [label_pretty[label] for label in labels] if label_pretty is not None else labels
        by_label = dict(zip(labels, handles))
        fig.legend(
            by_label.values(), by_label.keys(), loc="upper center", fontsize=18, ncol=3, bbox_to_anchor=(0.53, 1)
        )
        titles = ["Early Visual Cortex", "Ventral-Occipital Stream", "Dorsal Stream", "Lateral Stream"]
        for ax, t in zip(axes, titles):
            ax.legend_.remove() if ax.legend_ is not None else None
            ax.set_title(t, fontsize=20)
        for ax in axes:
            ax.legend_.remove() if ax.legend_ is not None else None
            xmin, xmax = ax.get_xlim()
            ax.set_xlim(xmin - 0.1, xmax + 0.1)
        plt.tight_layout(rect=(0.02, 0, 1, 0.93))
        # plt.show()
        plt.savefig("best_layer.pdf", format="pdf")


def adjust_layers(df, only_sf=False):
    def custom_agg(series, agg="mean"):
        if pd.api.types.is_array_like(series.iloc[0]):
            if agg == "mean":
                return np.stack(series.values).mean(axis=0)
            elif agg == "max":
                return np.stack(series.values).max(axis=0)
            else:
                raise ValueError("Invalid aggregation method, must be 'mean' or 'max'")
        elif series.nunique() == 1:
            # All values are the same, keep the first value
            return series.iloc[0]
        elif pd.api.types.is_integer_dtype(series):
            # If the values are integers, keep the first value
            return series.iloc[0]
        elif pd.api.types.is_numeric_dtype(series):
            # If the values are numeric (but not integer), agg them
            if agg == "mean":
                return series.mean()
            elif agg == "max":
                return series.max()
            else:
                raise ValueError("Invalid aggregation method, must be 'mean' or 'max'")
        else:
            return np.nan

    def slowfast_max_streams(group):
        if "slowfast" in group["Model"].unique()[0]:
            group = group.reset_index(drop=True)
            group["LayerSuffix"] = group["Layer_FullNames"].apply(
                lambda x: x.split("_lateral")[0].split("_")[-1] if "_lateral" in x else x.split("_")[-1]
            )
            p = partial(custom_agg, agg="max")
            group = group.groupby("LayerSuffix").apply(lambda x: x.apply(p)).reset_index(drop=True)
            group["Layer_FullNames"] = group["LayerSuffix"]
            group.drop("LayerSuffix", axis=1, inplace=True)
        return group

    df = df.sort_values("Layer").reset_index(drop=True)
    df = df.groupby("Model").apply(slowfast_max_streams).reset_index(drop=True)
    if only_sf:
        return df

    # Determine the most common number of layers
    layer_counts = df.groupby("Model")["Layer"].nunique()
    most_common_layers = layer_counts.mode().iloc[0]

    def average_layers(group):
        group = group.reset_index(drop=True)
        middle_layers = group.iloc[1:-1]
        while len(middle_layers) > most_common_layers - 2:
            new_middle_layers = []
            for i in range(len(middle_layers) - 1):
                pair = middle_layers.iloc[i : i + 2]
                new_middle_layers.append(pair.apply(custom_agg))
            middle_layers = pd.DataFrame(new_middle_layers)
            group.loc[len(group) - 1, "Layer"] = group.loc[len(group) - 1, "Layer"] - 1
        new_layer_indices = np.linspace(1, most_common_layers - 2, len(middle_layers), dtype=int)
        middle_layers["Layer"] = new_layer_indices
        group.loc[0, "Layer"] = 0
        group.loc[len(group) - 1, "Layer"] = most_common_layers - 1
        group = pd.concat([group.iloc[:1], middle_layers, group.iloc[-1:]])
        return group

    df = df.groupby("Model").apply(average_layers).reset_index(drop=True)
    df = df.reset_index(drop=True)
    return df


def normalize_to_model_depth(roi_df):
    max_layers = roi_df.groupby("Model")["Layer"].max().reset_index()
    max_layers.rename(columns={"Layer": "MaxLayer"}, inplace=True)
    roi_df = roi_df.merge(max_layers, on="Model", how="left")
    roi_df["Layer"] = roi_df["Layer"] / roi_df["MaxLayer"]
    return roi_df


def bmd_plot_all_layers_line_roi_groups(plotter, model_groups=None, separate=False, all_models_null_distribs=None):
    dataframes = []
    for dataframe in plotter.dataframes:
        dataframes.append(plotter.prepare_dataframe(dataframe, "R"))
    plotting_df = pd.concat(dataframes, ignore_index=True)
    plotting_df.drop("R_array", axis=1, inplace=True)

    roi_names = plotting_df["ROI"].unique()
    if sum([not s.startswith("r") and not s.startswith("l") for s in roi_names]) == 2:
        warnings.warn(
            "This function is not supported for hemispheres because they do not necessarily share the best layer. "
            "Skipping plot."
        )
        return
    if "V1d" in plotting_df["ROI"].values:
        warnings.warn(
            "This function is not supported for stream-divided early visual areas because they do not "
            "necessarily share the best layer. Skipping plot."
        )
        return

    if all_models_null_distribs is not None:
        all_models_null_distribs = convert_null_distrib_dict_to_df(all_models_null_distribs)
        plotting_df = plotting_df.merge(all_models_null_distribs, on=["Model", "Layer_FullNames", "ROI"])

    for i, group in enumerate(ROI_GROUPS.values()):
        n_rois = len(group)
        # rows = min(2, n_rois)
        # columns = np.ceil(n_rois / rows).astype(int)
        rows = 1
        columns = n_rois
        fig, axes = plt.subplots(
            rows, columns, figsize=(columns * 5, rows * 4.2), squeeze=False, sharex=True, sharey=True
        )
        axes = axes.flatten("F")
        for j, roi in enumerate(group):
            ax = axes[j]
            roi_df = plotting_df[plotting_df["ROI"] == roi]
            model_order = roi_df["Model"].unique()
            if model_groups is not None:
                # only_sf_adjust = partial(adjust_layers, only_sf=True)
                # old_roi_df = roi_df.groupby(model_groups).apply(only_sf_adjust).reset_index(drop=True)
                # old_roi_df = normalize_to_model_depth(old_roi_df)
                roi_df = roi_df.groupby(model_groups).apply(adjust_layers).reset_index(drop=True)
            roi_df = normalize_to_model_depth(roi_df)
            roi_df["CustomOrder"] = roi_df["Model"].map({model: k for k, model in enumerate(model_order)})
            mg_order = plotting_df[model_groups].unique()
            roi_df["CustomModelGroupOrder"] = roi_df[model_groups].map({mg: i for i, mg in enumerate(mg_order)})
            roi_df = roi_df.sort_values(["CustomModelGroupOrder", "CustomOrder", "Layer"]).reset_index(drop=True)
            roi_df.drop("CustomOrder", axis=1, inplace=True)
            hue = "Model" if model_groups is None else model_groups
            if model_groups == "ArchType":
                palette = sns.color_palette("tab10")[3:5]
            elif model_groups == "ModelTypeFolder":
                custom_shades = []
                for color in sns.color_palette("tab10"):
                    custom_shades.extend([change_luminance(color, 1.66), change_luminance(color, 1.33), color])
                palette = sns.color_palette(custom_shades, n_colors=roi_df[model_groups].nunique())
            else:
                palette = sns.color_palette("tab10", n_colors=len(roi_df[hue].unique()))
                # palette = [sns.color_palette("tab10")[-2], sns.color_palette("tab10")[-1]]
            sns.lineplot(
                data=roi_df,
                x="Layer",
                y="R",
                hue=hue,
                palette=palette,
                ax=ax,  # , estimator="median", errorbar=("pi",50)
                errorbar=None if separate else ("ci", 95),
            )
            if separate:
                for hue_value, plt_group in roi_df.groupby(hue):
                    for model_name, model_data in plt_group.groupby("Model"):
                        # if "uniformer" in model_name.lower() or "convit" in model_name.lower():
                        #     color = "black"
                        #     alpha = 1
                        # else:
                        #     color = palette[roi_df[hue].unique().tolist().index(hue_value)]
                        #     alpha = 0.2
                        alpha = 0.33
                        color = palette[roi_df[hue].unique().tolist().index(hue_value)]
                        ax.plot(
                            model_data["Layer"],
                            model_data["R"],
                            alpha=alpha,
                            linestyle="--",
                            color=color,
                        )
            ax.set_title(roi, fontsize=20) if roi != "IPS1" else ax.set_title("IPS1-3", fontsize=20)
            ax.set_ylabel("")
            ax.set_xlabel("")
            ax.tick_params(axis="y", labelsize=18)
            ax.tick_params(axis="x", labelsize=18)
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda x, _: "{:.0f}".format(x) if x in [0, 1] else "{:.2f}".format(x))
            )
            ax.get_legend().remove()
            ax.grid(False)
            ax.set_xlim(0, 1)
            ax.margins(x=0)
            if model_groups is not None and all_models_null_distribs is not None:
                mg_order = roi_df[model_groups].unique()
                roi_df["CustomModelGroupOrder"] = roi_df[model_groups].map({mg: i for i, mg in enumerate(mg_order)})
                roi_df, roi_mg_pairs = get_roi_mg_sigzero_sigpairs(roi_df, model_groups)
                df_lay_mg = (
                    roi_df.groupby(["Layer", model_groups])
                    .mean()
                    .reset_index()
                    .sort_values(by=["Layer", "CustomModelGroupOrder"])
                    .reset_index(drop=True)
                )
                ylowlim = ax.get_ylim()[0]
                sig_ylocs = {mg: ylowlim + i * 0.002 for i, mg in enumerate(mg_order)}
                _, labels = ax.get_legend_handles_labels()
                for line, mg in zip(ax.lines, labels):
                    line_color = line.get_color()
                    for line_other, mg_other in zip(ax.lines, labels):
                        if mg_other == mg:
                            continue
                        for (x, x_other), (y, y_other) in zip(
                            zip(line.get_xdata(), line_other.get_xdata()), zip(line.get_ydata(), line_other.get_ydata())
                        ):
                            row = df_lay_mg[(df_lay_mg["Layer"] == x) & (df_lay_mg[model_groups] == mg)]
                            [mg_pvalue] = row["mg_pvalue"].values
                            if mg_pvalue < 0.05:
                                ax.text(x, sig_ylocs[mg], "*", ha="center", va="bottom", c=line_color)
                            if abs(x - x_other) < 0.1:
                                pair_pvalue = roi_mg_pairs[roi][(mg, mg_other)]
                                if pair_pvalue < 0.0001:
                                    x_ofs = x + 0.01
                                    ymin = min(y, y_other)
                                    ymax = max(y, y_other)
                                    ax.plot(
                                        [x_ofs - 0.01, x_ofs, x_ofs, x_ofs - 0.01],
                                        [ymax, ymax, ymin, ymin],
                                        lw=1,
                                        c="k",
                                        alpha=0.33,
                                    )
                                    ax.text(x_ofs + 0.01, (y + y_other) / 2, "***", ha="left", va="center", c="k")
        # plt.subplots_adjust(hspace=0.4, right=0.45)
        fig.supylabel("Spearman's rho", fontsize=22)
        fig.supxlabel("Depth", fontsize=22, x=0.53)
        handles, labels = axes[0].get_legend_handles_labels()
        if model_groups == "ModelType":
            label_pretty = {
                "image-in1k": "Image Models (ImageNet-1k)",
                "image-k400": "Image Models (Kinetics-400)",
                "video-k400": "Video Models (Kinetics-400)",
            }
        elif model_groups == "ModelTypeFolder":
            label_pretty = {
                "image-in1k-models_no_dimred": "Image Models (ImageNet-1k) - no dim. reduction",
                "image-in1k-models": "Image Models (ImageNet-1k) - SRP",
                "image-in1k-models_pca_rerun": "Image Models (ImageNet-1k) - PCA",
                "image-k400-models_no_dimred": "Image Models (Kinetics-400) - no dim. reduction",
                "image-k400-models": "Image Models (Kinetics-400) - SRP",
                "image-k400-models_pca_rerun": "Image Models (Kinetics-400) - PCA",
                "video-k400-models_no_dimred": "Video Models (Kinetics-400) - no dim. reduction",
                "video-k400-models": "Video Models (Kinetics-400) - SRP",
                "video-k400-models_pca_rerun": "Video Models (Kinetics-400) - PCA",
            }
        else:
            label_pretty = None
        labels = [label_pretty[label] for label in labels] if label_pretty is not None else labels
        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.9, 0.9), fontsize=18)
        fig.tight_layout(rect=[0.02, 0, 1, 1])
        # plt.show()
        plt.savefig(f"all_layers_{i}.pdf", format="pdf")
        sleep(1.5**i)
