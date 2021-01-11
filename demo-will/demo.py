import streamlit as st
import matplotlib.pyplot as plt

import numpy as np
import pickle
import json
import copy
import pandas as pd
import random
from tqdm.auto import tqdm

import torch
from scipy.spatial import distance

import recnn

tqdm.pandas()

# constants
ML20MPATH = "../data/ml-20m/"
MODELSPATH = "models/"
DATAPATH = "data/"
SHOW_TOPN_MOVIES = (
    200  # recommend me a movie. show only top ... movies, higher values lead to slow ux
)

# disable it if you get an error
from jupyterthemes import jtplot

jtplot.style(theme="grade3")

def render_header():
    st.write(
        """
        <p align="center"> 
            <img src="https://raw.githubusercontent.com/awarebayes/RecNN/master/res/logo%20big.png">
        </p>
        <p align="center"> 
        <iframe src="https://ghbtns.com/github-btn.html?user=awarebayes&repo=recnn&type=star&count=true&size=large" frameborder="0" scrolling="0" width="160px" height="30px"></iframe>
        <iframe src="https://ghbtns.com/github-btn.html?user=awarebayes&repo=recnn&type=fork&count=true&size=large" frameborder="0" scrolling="0" width="158px" height="30px"></iframe>
        <iframe src="https://ghbtns.com/github-btn.html?user=awarebayes&type=follow&count=true&size=large" frameborder="0" scrolling="0" width="220px" height="30px"></iframe>
        </p>
        <p align="center"> 
        <a href='https://circleci.com/gh/awarebayes/RecNN'>
        <img src='https://circleci.com/gh/awarebayes/RecNN.svg?style=svg' alt='Documentation Status' />
        </a>
        <a href="https://codeclimate.com/github/awarebayes/RecNN/maintainability">
        <img src="https://api.codeclimate.com/v1/badges/d3a06ffe45906969239d/maintainability" />            
        </a>
        <a href="https://colab.research.google.com/github/awarebayes/RecNN/">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" />
        </a>
        <a href='https://recnn.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/recnn/badge/?version=latest' alt='Documentation Status' />
        </a>
        </p>
        <p align="center"> 
            <b> Choose the page on the left sidebar to proceed </b>
        </p>
        <p align="center"> 
            This is my school project. It focuses on Reinforcement Learning for personalized news recommendation.
            The main distinction is that it tries to solve online off-policy learning with dynamically generated 
            item embeddings. I want to create a library with SOTA algorithms for reinforcement learning
            recommendation, providing the level of abstraction you like.
        </p>
        <p align="center">
            <a href="https://recnn.readthedocs.io">recnn.readthedocs.io</a>
        </p>
        ### 📚 Read the articles on medium!
        - Pretty much what you need to get started with this library if you know recommenders
          but don't know much about reinforcement learning:
        <p align="center"> 
           <a href="https://towardsdatascience.com/reinforcement-learning-ddpg-and-td3-for-news-recommendation-d3cddec26011">
                <img src="https://raw.githubusercontent.com/awarebayes/RecNN/master/res/article_1.png"  width="100%">
            </a>
        </p>
        - Top-K Off-Policy Correction for a REINFORCE Recommender System:
        <p align="center"> 
           <a href="https://towardsdatascience.com/top-k-off-policy-correction-for-a-reinforce-recommender-system-e34381dceef8">
                <img src="https://raw.githubusercontent.com/awarebayes/RecNN/master/res/article_2.png" width="100%">
            </a>
        </p>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### 🤖 You can play with these (more will be implemented):
        | Algorithm                             | Paper                            | Code                       |
        |---------------------------------------|----------------------------------|----------------------------|
        | Deep Deterministic Policy Gradients   | https://arxiv.org/abs/1509.02971 | examples/1.Vanilla RL/DDPG |
        | Twin Delayed DDPG (TD3)               | https://arxiv.org/abs/1802.09477 | examples/1.Vanilla RL/TD3  |
        | Soft Actor-Critic                     | https://arxiv.org/abs/1801.01290 | examples/1.Vanilla RL/SAC  |
        | REINFORCE Top-K Off-Policy Correction | https://arxiv.org/abs/1812.02353 | examples/2. REINFORCE TopK |
    """
    )


@st.cache
def load_mekd():
    return pickle.load(open(DATAPATH + "myembeddings.pickle", "rb"))


def get_embeddings():
    movie_embeddings_key_dict = load_mekd()
    movies_embeddings_tensor, key_to_id, id_to_key = recnn.data.utils.make_items_tensor(
        movie_embeddings_key_dict
    )
    return movies_embeddings_tensor, key_to_id, id_to_key


def load_models(device):
    ddpg = recnn.nn.models.Actor(290, 28, 256).to(device)

    ddpg.load_state_dict(torch.load(MODELSPATH + 'ddpg_policy.pt', map_location=device))

    return {'ddpg': ddpg}


def rank(gen_action, metric, k):
    scores = []
    movie_embeddings_key_dict = load_mekd()
    meta = load_omdb_meta()

    for i in movie_embeddings_key_dict.keys():
        if i == 0 or i == "0":
            continue
        scores.append([i, metric(movie_embeddings_key_dict[i], gen_action)])
    scores = list(sorted(scores, key=lambda x: x[1]))
    scores = scores[:k]
    ids = [i[0] for i in scores]
    for i in range(k):
        scores[i].extend(
            [
                meta[str(scores[i][0])]["omdb"][key]
                for key in ["Title", "Genre", "imdbRating"]
            ]
        )
    indexes = ["id", "score", "Title", "Genre", "imdbRating"]
    table_dict = dict(
        [(key, [i[idx] for i in scores]) for idx, key in enumerate(indexes)]
    )
    table = pd.DataFrame(table_dict)
    return table


def main():
    st.sidebar.header("📰 recnn by @awarebayes 👨‍🔧")

    if st.sidebar.checkbox("Use cuda", torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    st.sidebar.subheader("Choose a page to proceed:")
    page = st.sidebar.selectbox(
        "",
        [
            "🚀 Get Started",
            "📽 ️Recommend me a movie",
            "🔨 Test Recommendation",
            "⛏️ Test Diversity",
            "🤖 Reinforce Top K",
        ],
    )

    if page == "🚀 Get Started":
        render_header()

        st.subheader("If you have cloned this repo, here is some stuff for you:")

        st.markdown(
            """
            📁 **Downloads** + change the **constants**, so they point to this unpacked folder:
            
            - [Models](https://drive.google.com/file/d/1goGa15XZmDAp2msZvRi2v_1h9xfmnhz7/view?usp=sharing)
             **= MODELSPATH**
            - [Data for Streamlit Demo](https://drive.google.com/file/d/1nuhHDdC4mCmiB7g0fmwUSOh1jEUQyWuz/view?usp=sharing)
             **= DATAPATH**
            - [ML20M Dataset](https://grouplens.org/datasets/movielens/20m/)
             **= ML20MPATH**
             
            p.s. ml20m is only needed for links.csv, I couldn't include it in my streamlit data because of copyright.
            This is all the data you need.
            """
        )

    if page == "🔨 Test Recommendation":
    	pass

    if page == "⛏️ Test Diversity":
       pass

    if page == "📽 ️Recommend me a movie":
        st.header("📽 ️Recommend me a movie")
        st.markdown(
            """
        **Now, this is probably why you came here. Let's get you some movies suggested**
        
        You need to choose 10 movies in the bar below by typing their titles.
        Due to the client side limitations, I am only able to display top 200 movies.
        P.S. you can type to search
        """
        )

        # mov_base = get_mov_base()
        # mov_base_by_title = {v: k for k, v in mov_base.items()}
        # movies_chosen = st.multiselect("Choose 10 movies", list(mov_base.values()))


        data = pd.read_csv(DATAPATH + 'train_test.csv')
        movies_chosen = data[data.customer_id == 161].vendor_id

        st.write('Vendor id dari customer test:', movies_chosen)

        # st.markdown(
        #     "**{} chosen {} to go**".format(len(moviess_chosen), 10 - len(movies_chosen))
        # )

        # if len(movies_chosen) > 10:
        #     st.error(
        #         "Please select exactly 10 movies, you have selected {}".format(
        #             len(movies_chosen)
        #         )
        #     )
        # if len(movies_chosen) == 10:
        #     st.success("You have selected 10 movies. Now let's rate them")
        # else:
        #     st.info("Please select 10 movies in the input above")

        if True:
            st.markdown("### Rate each movie from 1 to 10")
            ratings = data[data.customer_id == 161].total
            st.write('nilai total dari vendor customer test', ratings)

            ids = movies_chosen
            # st.write('Movie indexes', list(ids))s
            embs = load_mekd()
            st.write("your embs", embs[0].shape)
            state = torch.cat(
                [
                    torch.cat([embs[i] for i in ids]),
                    torch.tensor(list(ratings)).float(),
                ]
            )
            state = state.to(device).squeeze(0)
            st.write("your state", state.shape)

            models = load_models(device)
            st.write("your model", models["ddpg"])
            algorithm = st.selectbox("Choose an algorithm", ("ddpg", "td3"))

            metric = st.selectbox(
                "Choose a metric",
                (
                    "euclidean",
                    "cosine",
                    "correlation",
                    "canberra",
                    "minkowski",
                    "chebyshev",
                    "braycurtis",
                    "cityblock",
                ),
            )

            dist = {
                "euclidean": distance.euclidean,
                "cosine": distance.cosine,
                "correlation": distance.correlation,
                "canberra": distance.canberra,
                "minkowski": distance.minkowski,
                "chebyshev": distance.chebyshev,
                "braycurtis": distance.braycurtis,
                "cityblock": distance.cityblock,
            }

            topk = st.slider(
                "TOP K items to recommend:", min_value=1, max_value=30, value=7
            )
            action = models["ddpg"].forward(state)

            st.subheader("The neural network thinks you should watch:")
            st.write(rank(action[0].detach().cpu().numpy(), dist[metric], topk))

    if page == "🤖 Reinforce Top K":
        pass


if __name__ == "__main__":
    main()
