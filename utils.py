import pandas
import plotly.express as px
import streamlit as st
from scipy.stats import kendalltau


def aggregate_results(data, target_property):
    aggregation = data.melt(
        value_vars=st.session_state["models"],
        id_vars=['linguistic phenomena', 'probing dataset', 'probe type','linguistic competencies']
    )[[target_property, "variable", "value"]].pivot_table(index=[target_property, "variable"], values="value").reset_index()

    aggregation.columns = [target_property, "model", "score"]

    return aggregation

def read_data(path, train_portions=[1]):
    frame = pandas.read_csv(path, index_col=0)
    frame = frame[frame["train portion"].isin(train_portions)]
    frame["linguistic competencies"] = frame["linguistic subfield"]
    del frame["linguistic subfield"]

    frame = frame.groupby(["probing dataset", "linguistic phenomena", "probe type", "probe", "linguistic competencies"]).mean().reset_index()
    del frame["seed"]
    del frame["train portion"]

    return frame


def get_rankings(data, model_columns):
    rankings = data[model_columns].rank(axis=1, method="max", ascending=True)
    normalized_rankings = rankings.apply(lambda row: (row-row.min()) / (row.max() - row.min()),axis=1)
    ranking_data = data.copy()
    ranking_data[model_columns] = normalized_rankings

    return ranking_data


def get_polar_plot(data, target_column, title):
    return px.line_polar(
        data, r='score', theta=target_column, markers=True,
        line_close=True, color="model", width=500, title=title
    )


def update_rankings(rankings, selected_models, add_average=True):
    rankings[selected_models] = rankings[selected_models] * 100

    mean_ranking = rankings[selected_models].mean(axis=0).values
    rankings["deviation"] = rankings[selected_models].std(axis=1)
    rankings["discriminability"] = rankings.apply(lambda row: kendalltau(mean_ranking, row[selected_models].values).pvalue, axis=1).fillna(0)

    if add_average:
        average = dict(rankings.mean(numeric_only=True))
        average["probing dataset"] = "Average"
        rankings = pandas.concat([rankings, pandas.DataFrame([average])], ignore_index = True).fillna("")

    return rankings

def update_rankings(rankings, selected_models):
    mean_ranking = rankings[selected_models].mean(axis=0).values
    rankings["deviation"] = rankings[selected_models].std(axis=1)
    rankings["discriminability"] = rankings.apply(lambda row: kendalltau(mean_ranking, row[selected_models].values).pvalue, axis=1)

    return rankings