import pandas
import streamlit as st
from PIL import Image

from utils import read_data, get_rankings

im = Image.open("images/holmes_leaderboard_icon.png")

st.set_page_config(
    layout="wide",
    page_icon=im,
)
st.image("images/holmes_leaderboard.svg", width=400)
st.markdown("""
    This is the leaderboard of the Holmes 🔎 benchmark. 
    You can choose to consider only non licensed datasets or all of them, and how you wish to sort the table.
""")


holmes_version = st.selectbox("Select Holmes Version", options=["Holmes 🔎 - Our comprehensive version", "FlashHolmes ⚡ - Our streamlined version, built with efficiency in mind"])
free_holmes = st.checkbox("Only Freely-Available Datasets")

sort_by = st.selectbox(
    "Select Columns to Sort",
    ["overall", "params", "morphology", "syntax", "semantics", "reasoning", "discourse"]
)

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
    'microsoft/phi-2': "Dec",
    'EleutherAI/gpt-j-6b': "Dec",
    'EleutherAI/pythia-1b': "Dec",
    'EleutherAI/pythia-1b-deduped': "Dec",
    'EleutherAI/pythia-1.4b': "Dec",
    'EleutherAI/pythia-1.4b-deduped': "Dec",
    'EleutherAI/pythia-12b': "Dec",
    'EleutherAI/pythia-12b-deduped': "Dec",
    'EleutherAI/pythia-160m': "Dec",
    'EleutherAI/pythia-160m-deduped': "Dec",
    'EleutherAI/pythia-2.8b': "Dec",
    'EleutherAI/pythia-2.8b-deduped': "Dec",
    'EleutherAI/pythia-410m': "Dec",
    'EleutherAI/pythia-410m-deduped': "Dec",
    'EleutherAI/pythia-6.9b': "Dec",
    'EleutherAI/pythia-6.9b-deduped': "Dec",
    'EleutherAI/pythia-70m': "Dec",
    'EleutherAI/pythia-70m-deduped': "Dec",
    'albert-base-v2': "Enc",
    'bert-base-uncased': "Enc",
    'google/electra-base-discriminator': "Enc",
    'google/flan-t5-xxl': "Enc-Dec",
    'google/flan-t5-base': "Enc-Dec",
    'google/flan-t5-small': "Enc-Dec",
    'google/flan-t5-large': "Enc-Dec",
    'google/flan-t5-xl': "Enc-Dec",
    'google/flan-ul2': "Enc-Dec",
    'google/t5-base-lm-adapt': "Enc-Dec",
    'google/t5-small-lm-adapt': "Enc-Dec",
    'google/t5-large-lm-adapt': "Enc-Dec",
    'google/t5-xl-lm-adapt': "Enc-Dec",
    'google/t5-xxl-lm-adapt': "Enc-Dec",
    'google/ul2': "Enc-Dec",
    'gpt2': "Dec",
    'facebook/bart-base': "Enc",
    'meta-llama/Llama-2-13b-chat-hf': "Dec",
    'meta-llama/Llama-2-13b-hf': "Dec",
    'meta-llama/Llama-2-70b-chat-hf': "Dec",
    'meta-llama/Llama-2-70b-hf': "Dec",
    'meta-llama/Llama-2-7b-chat-hf': "Dec",
    'meta-llama/Llama-2-7b-hf': "Dec",
    'microsoft/deberta-base': "Enc",
    'microsoft/deberta-v3-base': "Enc",
    'roberta-base': "Enc",
    'databricks/dolly-v2-12b': "Dec",
    'allenai/tk-instruct-11b-def': "Enc-Dec",
    'allenai/tulu-2-dpo-13b': "Dec",
    'allenai/tulu-2-13b': "Dec",
    'microsoft/Orca-2-13b': "Dec",
    'lmsys/vicuna-13b-v1.5': "Dec",
    'mistralai/Mixtral-8x7B-Instruct-v0.1': "Dec",
    'mistralai/Mixtral-8x7B-v0.1': "Dec",
}


if free_holmes:
    holmes_version = f"Free{holmes_version}"

if "data" not in st.session_state:
    st.session_state["raw_data"] = {
        "Holmes 🔎 - Our comprehensive version": read_data("data/holmes_results_f1_raw_v4.0.csv"),
        "FlashHolmes ⚡ - Our streamlined version, built with efficiency in mind": read_data("data/holmes_results_f1_raw_v4.0.csv", train_portions=[0.0625]),
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

    for scope in ["Overall", "Discourse", "Morphology", "Reasoning", "Semantics", "Syntax"]:
        formats[scope] = "{:.1f}"

    colored_leaderboard = leaderboard.style.format(formats) \
        .background_gradient(cmap='Blues', subset=["Overall", "Discourse", "Morphology", "Reasoning", "Semantics", "Syntax"], axis=0)

    styled_leaderboard = colored_leaderboard.set_table_styles([
        {'selector': 'thead', 'props': 'border-top: 5px solid rgba(49, 51, 63, 0.2)'},
        {'selector': 'thead', 'props': 'border-bottom: 3px double rgba(49, 51, 63, 0.2)'}
    ])

    return styled_leaderboard


with st.expander("Leaderboard", expanded=True):
    leaderboard = leaderboard.sort_values(sort_by, ascending=False)

    leaderboard.columns = ['Model', 'Architecture', 'Params', "Overall", "Discourse", "Morphology", "Reasoning", "Semantics", "Syntax"]

    styled_leaderboard = style_leaderboard(leaderboard)

    st.write(styled_leaderboard.to_html(), unsafe_allow_html=True)

