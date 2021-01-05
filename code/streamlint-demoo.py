import streamlit as st
import pandas as pd
import pickle

import recnn
from scipy.spatial import distance
import torch


MODELSPATH = './models/'
DATAPATH = './data/'

@st.cache
def load_mekd():
    return pickle.load(open(DATAPATH + 'dataset.pickle', 'rb'))

def get_embeddings():
    movie_embeddings_key_dict = load_mekd()
    movies_embeddings_tensor, key_to_id, id_to_key = recnn.data.utils.make_items_tensor(movie_embeddings_key_dict)
    return  movies_embeddings_tensor, key_to_id, id_to_key

@st.cache
def load_links():
    return pd.read_csv(DATAPATH + 'train_test.csv', index_col='vendor_id')

@st.cache
def get_mov_base():
    links = load_links()
    movies_embeddings_tensor, key_to_id, id_to_key = get_embeddings()
    # meta = load_omdb_meta()

    # popular = pd.read_csv(DATAPATH + 'movie_counts.csv')[:SHOW_TOPN_MOVIES]
    mov_base = {}

    # for i, k in list(meta.items()):
    #     tmdid = int(meta[i]['tmdbId'])
    #     if tmdid > 0 and popular['id'].isin([i]).any():
    #         movieid = pd.to_numeric(links.loc[tmdid]['movieId'])
    #         if isinstance(movieid, pd.Series):
    #             continue
    #         mov_base[int(movieid)] = meta[i]['omdb']['Title']

    return mov_base

def load_models(device):
    ddpg = recnn.nn.models.Actor(290, 28, 256).to(device)

    ddpg.load_state_dict(torch.load(MODELSPATH + 'ddpg_policy.model', map_location=device))
    return ddpg

def main():

    if st.sidebar.checkbox('Use cuda', torch.cuda.is_available()):
        device = torch.device('cpu')
    else:
        device = torch.device('cpu')
        
    st.header("📽 ️Recommend me a Restaurant")


    mov_base = get_mov_base()
    mov_base_by_title = {v: k for k, v in mov_base.items()}
    movies_chosen = st.multiselect('Choose 10 movies', list(mov_base.values()))

        # if len(movies_chosen) == 10:
        #     st.markdown('### Rate each movie from 1 to 10')
        #     ratings = dict([(i, st.number_input(i, min_value=1, max_value=10, value=5)) for i in movies_chosen])
            # st.write('for debug your ratings are:', ratings)


        #     ids = [mov_base_by_title[i] for i in movies_chosen]
        #     # st.write('Movie indexes', list(ids))
    # embs = load_mekd()
    # state = torch.cat([torch.cat([embs[i] for i in ids]), torch.tensor(list(ratings.values())).float() - 5])
    # st.write('your state', state)
    # state = state.to(device).squeeze(0)

    models = load_models(device)
    algorithm = st.selectbox('Choose an algorithm', ('ddpg', 'td3'))

    metric = st.selectbox('Choose a metric', ('euclidean', 'cosine', 'correlation',
                                            'canberra', 'minkowski', 'chebyshev',
                                            'braycurtis', 'cityblock',))

    dist = {'euclidean': distance.euclidean, 'cosine': distance.cosine,
                    'correlation': distance.correlation, 'canberra': distance.canberra,
                    'minkowski': distance.minkowski, 'chebyshev': distance.chebyshev,
                    'braycurtis': distance.braycurtis, 'cityblock': distance.cityblock}

    topk = st.slider("TOP K items to recommend:", min_value=1, max_value=30, value=7)
    action = models[algorithm].forward(state)

    st.subheader('The neural network thinks you should watch:')
    st.write(rank(action[0].detach().cpu().numpy(), dist[metric], topk))

if __name__ == "__main__":
    main()
