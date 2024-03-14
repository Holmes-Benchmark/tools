import json
import random
from collections import defaultdict
import pandas
import plotly.express as px

import streamlit as st
import plotly.express as px
from scipy.stats import kendalltau
from st_aggrid import AgGrid, AgGridTheme

from st_keyup import st_keyup
import redis

from app.pages.utils import read_data, get_rankings, aggregate_results

st.set_page_config(layout="wide")
st.title("Holmes Leaderboard")


parameters = {
    'sentence-transformers/average_word_embeddings_glove.6B.300d': 300,
    'sentence-transformers/average_word_embeddings_glove.840B.300d': 300,
    'microsoft/phi-2': 2700000000,
    'EleutherAI/gpt-j-6b': 6000000000,
    'EleutherAI/pythia-1b': 1000000000,
    'EleutherAI/pythia-1b-deduped': 1000000000,
    'EleutherAI/pythia-1.4b': 1400000000,
    'EleutherAI/pythia-1.4b-deduped': 1400000000,
    'EleutherAI/pythia-12b': 12000000000,
    'EleutherAI/pythia-12b-deduped': 12000000000,
    'EleutherAI/pythia-160m': 160000000,
    'EleutherAI/pythia-160m-deduped': 160000000,
    'EleutherAI/pythia-2.8b': 2800000000,
    'EleutherAI/pythia-2.8b-deduped': 2800000000,
    'EleutherAI/pythia-410m': 410000000,
    'EleutherAI/pythia-410m-deduped': 410000000,
    'EleutherAI/pythia-6.9b': 6900000000,
    'EleutherAI/pythia-6.9b-deduped': 6900000000,
    'EleutherAI/pythia-70m': 70000000,
    'EleutherAI/pythia-70m-deduped': 70000000,
    'albert-base-v2': 12000000,
    'bert-base-uncased': 108000000,
    'google/electra-base-discriminator': 110000000,
    'google/flan-t5-xxl': 11000000000,
    'google/flan-t5-base': 220000000,
    'google/flan-t5-small': 60000000,
    'google/flan-t5-large': 770000000,
    'google/flan-t5-xl': 3000000000,
    'google/flan-ul2': 20000000000,
    'google/t5-base-lm-adapt': 220000000,
    'google/t5-small-lm-adapt': 60000000,
    'google/t5-large-lm-adapt': 770000000,
    'google/t5-xl-lm-adapt': 3000000000,
    'google/t5-xxl-lm-adapt': 11000000000,
    'google/ul2': 20000000000,
    'gpt2': 45000000,
    'facebook/bart-base': 140000000,
    'meta-llama/Llama-2-13b-chat-hf': 13000000000,
    'meta-llama/Llama-2-13b-hf': 13000000000,
    'meta-llama/Llama-2-70b-chat-hf': 70000000000,
    'meta-llama/Llama-2-70b-hf': 70000000000,
    'meta-llama/Llama-2-7b-chat-hf': 7000000000,
    'meta-llama/Llama-2-7b-hf': 7000000000,
    'microsoft/deberta-base': 100000000,
    'microsoft/deberta-v3-base': 86000000,
    'roberta-base': 125000000,
    'databricks/dolly-v2-12b': 12000000000,
    'allenai/tk-instruct-11b-def': 11000000000,
    'allenai/tulu-2-dpo-13b': 13000000000,
    'allenai/tulu-2-13b': 13000000000,
    'microsoft/Orca-2-13b': 13000000000,
    'lmsys/vicuna-13b-v1.5': 13000000000,
    'mistralai/Mixtral-8x7B-Instruct-v0.1': 56000000000,
    'mistralai/Mixtral-8x7B-v0.1': 56000000000,
}


architecture = {
    'sentence-transformers/average_word_embeddings_glove.6B.300d': "static",
    'sentence-transformers/average_word_embeddings_glove.840B.300d': "static",
    'microsoft/phi-2': "decoder",
    'EleutherAI/gpt-j-6b': "decoder",
    'EleutherAI/pythia-1b': "decoder",
    'EleutherAI/pythia-1b-deduped': "decoder",
    'EleutherAI/pythia-1.4b': "decoder",
    'EleutherAI/pythia-1.4b-deduped': "decoder",
    'EleutherAI/pythia-12b': "decoder",
    'EleutherAI/pythia-12b-deduped': "decoder",
    'EleutherAI/pythia-160m': "decoder",
    'EleutherAI/pythia-160m-deduped': "decoder",
    'EleutherAI/pythia-2.8b': "decoder",
    'EleutherAI/pythia-2.8b-deduped': "decoder",
    'EleutherAI/pythia-410m': "decoder",
    'EleutherAI/pythia-410m-deduped': "decoder",
    'EleutherAI/pythia-6.9b': "decoder",
    'EleutherAI/pythia-6.9b-deduped': "decoder",
    'EleutherAI/pythia-70m': "decoder",
    'EleutherAI/pythia-70m-deduped': "decoder",
    'albert-base-v2': "encoder",
    'bert-base-uncased': "encoder",
    'google/electra-base-discriminator': "encoder",
    'google/flan-t5-xxl': "encoder-decoder",
    'google/flan-t5-base': "encoder-decoder",
    'google/flan-t5-small': "encoder-decoder",
    'google/flan-t5-large': "encoder-decoder",
    'google/flan-t5-xl': "encoder-decoder",
    'google/flan-ul2': "encoder-decoder",
    'google/t5-base-lm-adapt': "encoder-decoder",
    'google/t5-small-lm-adapt': "encoder-decoder",
    'google/t5-large-lm-adapt': "encoder-decoder",
    'google/t5-xl-lm-adapt': "encoder-decoder",
    'google/t5-xxl-lm-adapt': "encoder-decoder",
    'google/ul2': "encoder-decoder",
    'gpt2': "decoder",
    'facebook/bart-base': "encoder",
    'meta-llama/Llama-2-13b-chat-hf': "decoder",
    'meta-llama/Llama-2-13b-hf': "decoder",
    'meta-llama/Llama-2-70b-chat-hf': "decoder",
    'meta-llama/Llama-2-70b-hf': "decoder",
    'meta-llama/Llama-2-7b-chat-hf': "decoder",
    'meta-llama/Llama-2-7b-hf': "decoder",
    'microsoft/deberta-base': "encoder",
    'microsoft/deberta-v3-base': "encoder",
    'roberta-base': "encoder",
    'databricks/dolly-v2-12b': "decoder",
    'allenai/tk-instruct-11b-def': "encoder-decoder",
    'allenai/tulu-2-dpo-13b': "decoder",
    'allenai/tulu-2-13b': "decoder",
    'microsoft/Orca-2-13b': "decoder",
    'lmsys/vicuna-13b-v1.5': "decoder",
    'mistralai/Mixtral-8x7B-Instruct-v0.1': "decoder",
    'mistralai/Mixtral-8x7B-v0.1': "decoder",
}

holmes_version = st.sidebar.selectbox("Select Holmes Version", options=["Holmes", "FlashHolmes"])
free_holmes = st.sidebar.checkbox("Only Freely-Available Datasets")

if free_holmes:
    holmes_version = f"Free{holmes_version}"

if "data" not in st.session_state:
    st.session_state["raw_data"] = {
        "Holmes": read_data("data/holmes_results_f1_raw_v4.0.csv"),
        "FlashHolmes": read_data("data/holmes_results_f1_raw_v4.0.csv", train_portions=[0.0625]),
        "FreeHolmes": read_data("data/holmes_results_f1_raw_free_v4.0.csv"),
        "FreeFlashHolmes": read_data("data/holmes_results_f1_raw_free_v4.0.csv", train_portions=[0.0625]),
        #"f1_std": read_data("data/holmes_results_f1-std.csv"),
        #"compression": read_data("data/holmes_results_compression.csv"),
    }

st.session_state["probing_datasets"] = list(st.session_state["raw_data"][holmes_version]["probing dataset"].unique())
st.session_state["linguistic_competencies"] = list(st.session_state["raw_data"][holmes_version]["linguistic competencies"].unique())
st.session_state["linguistic_phenomena"] = list(st.session_state["raw_data"][holmes_version]["linguistic phenomena"].unique())
st.session_state["models"] = st.session_state["raw_data"][holmes_version].columns[6:-1]

st.session_state["rankings_f1"] = get_rankings(st.session_state["raw_data"][holmes_version], st.session_state["models"])

overall_rankings = {
    "overall": {model: st.session_state["rankings_f1"][model].mean() * 100 for model in st.session_state["models"]}
}

for competence, competence_frame in st.session_state["raw_data"][holmes_version].groupby("linguistic competencies"):
    competence_rankings = get_rankings(competence_frame, st.session_state["models"])
    overall_rankings[competence] = {model: competence_rankings[model].mean() * 100 for model in st.session_state["models"]}

leaderboard = pandas.DataFrame(overall_rankings).reset_index()
leaderboard.columns = ["model_name", "overall", "discourse", "morphology", "reasoning", "semantics", "syntax"]
leaderboard["params"] = leaderboard["model_name"].apply(lambda model_name: parameters.get(model_name, -1))
leaderboard["architecture"] = leaderboard["model_name"].apply(lambda model_name: architecture.get(model_name, -1))


leaderboard = leaderboard[["model_name", "architecture", "params", "overall", "discourse", "morphology", "reasoning", "semantics", "syntax"]]


def style_leaderboard(leaderboard):

    formats = {}

    for scope in ["overall", "discourse", "morphology", "reasoning", "semantics", "syntax"]:
        formats[scope] = "{:.1f}"

    colored_leaderboard = leaderboard.style.format(formats) \
        .background_gradient(cmap='Blues', subset=["overall", "discourse", "morphology", "reasoning", "semantics", "syntax"], axis=0)

    styled_leaderboard = colored_leaderboard.set_table_styles([
        {'selector': 'thead', 'props': 'border-top: 5px solid rgba(49, 51, 63, 0.2)'},
        {'selector': 'thead', 'props': 'border-bottom: 3px double rgba(49, 51, 63, 0.2)'}
    ])

    return styled_leaderboard


with st.expander("Leaderboard", expanded=True):
    leaderboard = leaderboard.sort_values("overall", ascending=False)

    styled_leaderboard = style_leaderboard(leaderboard)

    st.write(styled_leaderboard.to_html(), unsafe_allow_html=True)

