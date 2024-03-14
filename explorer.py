import pandas
import seaborn as sns
import streamlit as st
from scipy.stats import kendalltau

from utils import read_data, get_rankings, aggregate_results, get_polar_plot


st.set_page_config(layout="wide")
st.title("Holmes Explorer")

st.markdown("""
    This page allows you do analyze results of Holmes regarding language models and datasets. 
    
    ### Overall Results
    
    This part shows you how the different language models perform. 
    You can analyze the `mean winning rate` of the selected language models for `linguistic competencies`, `linguistic phenomena`, or `probing datasets`.
    Note, these plots are not connected to each other. You can select syntax phenomena while inspecting all `competencies` on the same time. 

""")



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

if "data" not in st.session_state:
    st.session_state["raw_data"] = {
        "f1": read_data("data/holmes_results_f1_raw_v4.0.csv"),
        #"f1_std": read_data("data/holmes_results_f1-std.csv"),
        #"compression": read_data("data/holmes_results_compression.csv"),
    }

    st.session_state["probing_datasets"] = list(st.session_state["raw_data"]["f1"]["probing dataset"].unique())
    st.session_state["linguistic_competencies"] = list(st.session_state["raw_data"]["f1"]["linguistic competencies"].unique())
    st.session_state["linguistic_phenomena"] = list(st.session_state["raw_data"]["f1"]["linguistic phenomena"].unique())
    st.session_state["models"] = st.session_state["raw_data"]["f1"].columns[6:-1]

    st.session_state["rankings_f1"] = get_rankings(st.session_state["raw_data"]["f1"], st.session_state["models"])
    #st.session_state["rankings_f1_std"] = get_rankings(st.session_state["raw_data"]["f1_std"], st.session_state["models"])
    #st.session_state["rankings_compression"] = get_rankings(st.session_state["raw_data"]["compression"], st.session_state["models"])

    st.session_state["aggregated_by_competencies"] = aggregate_results(st.session_state["rankings_f1"], target_property='linguistic competencies')
    st.session_state["aggregated_by_phenomena"] = aggregate_results(st.session_state["rankings_f1"], target_property= 'linguistic phenomena')
    st.session_state["aggregated_by_datasets"] = aggregate_results(st.session_state["rankings_f1"], target_property='probing dataset')


st.multiselect(
    label="Select models to compare",
    options=st.session_state["models"],
    key="selected_models",
    default=["google/ul2", "google/flan-ul2", "bert-base-uncased", "EleutherAI/pythia-12b"]
)



with st.expander("Overall results", expanded=True):

    selected_models = st.session_state["selected_models"]




    tab1, tab2, tab3 = st.tabs(["Linguistic Competencies", "Linguistic Phenomena", "Probing Datasets"])

    with tab1:
        st.multiselect(
            label="Select linguistic competencies to analyze",
            options=st.session_state["linguistic_competencies"],
            key="selected_competencies",
            default=st.session_state["linguistic_competencies"]
        )

        relevant_competencies_data = st.session_state["aggregated_by_competencies"][st.session_state["aggregated_by_competencies"]["model"].isin(selected_models)]
        relevant_competencies_data = relevant_competencies_data[relevant_competencies_data['linguistic competencies'].isin(st.session_state["selected_competencies"])]

        fig = get_polar_plot(data=relevant_competencies_data, target_column='linguistic competencies', title="Competencies Comparison")
        st.plotly_chart(fig)

    with tab2:
        st.multiselect(
            label="Select linguistic phenomena to analyze",
            options=st.session_state["linguistic_phenomena"],
            key="selected_phenomena",
            default=["negation", "part-of-speech", "binding"]
        )

        relevant_phenomena_data = st.session_state["aggregated_by_phenomena"][st.session_state["aggregated_by_phenomena"]["model"].isin(selected_models)]
        relevant_phenomena_data = relevant_phenomena_data[relevant_phenomena_data['linguistic phenomena'].isin(st.session_state["selected_phenomena"])]

        fig = get_polar_plot(data=relevant_phenomena_data, target_column='linguistic phenomena', title="Phenomena Comparison")
        st.plotly_chart(fig)

    with tab3:

        st.multiselect(
            label="Select datasets to analyze",
            options=st.session_state["probing_datasets"],
            key="selected_datasets",
            default=["pos", "xpos", "upos"]
        )

        relevant_dataset_data = st.session_state["aggregated_by_datasets"][st.session_state["aggregated_by_datasets"]["model"].isin(selected_models)]
        relevant_dataset_data = relevant_dataset_data[relevant_dataset_data['probing dataset'].isin(st.session_state["selected_datasets"])]

        fig = get_polar_plot(data=relevant_dataset_data, target_column='probing dataset', title="Dataset Comparison")
        st.plotly_chart(fig)


rankings = st.session_state["rankings_f1"]

cm = sns.color_palette("blend:white,green", as_cmap=True)

def apply_gradient(s, cmap):
    # Normalize the column values to 0-1 range
    norm = (s - s.min()) / (s.max() - s.min())
    # Convert normalized values to colors from the colormap
    colors = [f'background-color: {cmap(x)}' for x in norm]
    return colors

def get_cmap(name):
    import matplotlib
    return matplotlib.colormaps.get_cmap(name)

def style_rankings(rankings, selected_models):

    formats = {
        "deviation":"{:.1f}",
        "discriminability":"{:.2f}",
    }

    for model in selected_models:
        formats[model] = "{:.1f}"

    colored_rankings = rankings.style.format(formats)\
        .background_gradient(cmap='Blues', subset=selected_models, axis=1) \
        .background_gradient(cmap='Greens', subset=["deviation"], axis=0) \
        .background_gradient(cmap='Reds', subset=["discriminability"], axis=0)

    styled_rankings = colored_rankings.set_table_styles([
        {'selector': 'thead', 'props': 'border-top: 5px solid rgba(49, 51, 63, 0.2)'},
        {'selector': 'thead', 'props': 'border-bottom: 3px double rgba(49, 51, 63, 0.2)'}
    ])
    n_rows = rankings.shape[0] - 1

    styled_rankings = styled_rankings.set_table_styles([
        {'selector': f'td.row{n_rows}', 'props': 'border-top: 5px double rgba(49, 51, 63, 0.2)'},
        {'selector': f'td.row{n_rows}', 'props': 'border-bottom: 3px solid rgba(49, 51, 63, 0.2)'},
        {'selector': f'th.row{n_rows}', 'props': 'border-top: 5px double rgba(49, 51, 63, 0.2)'},
        {'selector': f'th.row{n_rows}', 'props': 'border-bottom: 3px solid rgba(49, 51, 63, 0.2)'},
    ], overwrite=False)


    styled_rankings = styled_rankings.set_table_styles({
            'deviation': [
                {'selector': 'th', 'props': 'border-left: 2px solid black'},
                {'selector': 'td', 'props': 'border-left: 2px solid black'}
            ]
    }, overwrite=False)

    styled_rankings = styled_rankings.set_table_styles({
        'linguistic phenomena': [
            {'selector': 'th', 'props': 'border-right: 2px solid black'},
            {'selector': 'td', 'props': 'border-right: 2px solid black'}
        ]
    }, overwrite=False)

    return styled_rankings

st.markdown("""
    ### Competencies Details
    
    The second part shows details of `competencies` separately. This includes the `winning rate` for a specific model and the `probing datasets` under test. In addition, `deviation` tells how much the selected language models deviate among each other for a specfic `probing dataset`. With `discriminability`, we report the agreement of a given `probing dataset` with the average rankings of the specific `linguistic competence`. For example, `discriminability=1` means that the model rankings of the specific `probing dataset` agrees 100% with average rankings of the specific `linguistic competence`.     
""")


tab_morphology, tab_syntax, tab_semantics, tab_reasoning, tab_discourse = st.tabs(["Morphology", "Syntax", "Semantics", "Reasoning", "Discourse"])


with tab_morphology:
    morphology_rankings = rankings[rankings["linguistic competencies"] == "morphology"]
    morphology_rankings = morphology_rankings[["probing dataset", "linguistic phenomena"] + selected_models]
    morphology_rankings = update_rankings(morphology_rankings, selected_models)

    styled_rankings = style_rankings(morphology_rankings, selected_models)

    st.write(styled_rankings.to_html(), unsafe_allow_html=True)



with tab_syntax:
    syntax_rankings = rankings[rankings["linguistic competencies"] == "syntax"]
    syntax_rankings = syntax_rankings[["probing dataset", "linguistic phenomena"] + selected_models]
    syntax_rankings = update_rankings(syntax_rankings, selected_models)

    styled_rankings = style_rankings(syntax_rankings, selected_models)

    st.write(styled_rankings.to_html(), unsafe_allow_html=True)


with tab_semantics:
    semantics_rankings = rankings[rankings["linguistic competencies"] == "semantics"]
    semantics_rankings = semantics_rankings[["probing dataset", "linguistic phenomena"] + selected_models]
    semantics_rankings = update_rankings(semantics_rankings, selected_models)

    styled_rankings = style_rankings(semantics_rankings, selected_models)

    st.write(styled_rankings.to_html(), unsafe_allow_html=True)


with tab_reasoning:
    reasoning_rankings = rankings[rankings["linguistic competencies"] == "reasoning"]
    reasoning_rankings = reasoning_rankings[["probing dataset", "linguistic phenomena"] + selected_models]
    reasoning_rankings = update_rankings(reasoning_rankings, selected_models)

    styled_rankings = style_rankings(reasoning_rankings, selected_models)

    st.write(styled_rankings.to_html(), unsafe_allow_html=True)


with tab_discourse:
    discourse_rankings = rankings[rankings["linguistic competencies"] == "discourse"]
    discourse_rankings = discourse_rankings[["probing dataset", "linguistic phenomena"] + selected_models]
    discourse_rankings = update_rankings(discourse_rankings, selected_models)

    styled_rankings = style_rankings(discourse_rankings, selected_models)

    st.write(styled_rankings.to_html(), unsafe_allow_html=True)


