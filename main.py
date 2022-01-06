import gc
import pickle
import requests
import time
import numpy as np
import streamlit as st
from bs4 import BeautifulSoup


def cos_similarity(x, y, eps=1e-16):
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)  

def retrieval(query_url):
    while True:
        res = requests.get(query_url)
        if res.status_code == 200:
            break
        time.sleep(1)
    soup = BeautifulSoup(res.text, 'html.parser')
    query_title = soup.select_one('#abs > h1').get_text().replace("Title:", "")
    query_abst = soup.select_one('#abs > blockquote').get_text().replace("Abstract: ", "")
    return query_title, query_abst


def main():
    n_split = 30
    gc.enable()

    st.write('## Content-Based Related Paper Search')
    st.write('You can search for similar machine learning papers in arXiv published by 2021 (category: cs.AI, cs.LG, stat.ML, cs.CV, and cs.CL). The similarity of title and abstract embeddings obtained by SPECTER [Cohan+, ACL 2020] are used for searching. Please enter arXiv abstract page url.')
    query_url = st.text_input('arXiv abstract page url (× pdf page url) e.g. https://arxiv.org/abs/xxxx.yyyyy', '')
    top_n = st.text_input('The number of search results', '10')
    top_n = int(top_n)

    with open('./data/raw_link.pkl', 'rb') as f:
        link_list = pickle.load(f)

    use_title = st.checkbox('Use title information', value=True)
    use_abst = st.checkbox('Use abstract information', value=True)
    ready = st.button('Go!')

    if ready:
        if query_url == '':
            raise ValueError('Enter arXiv abstract page url and try again.')

        try:
            query_idx = link_list.index(query_url)
        except:
            raise ValueError('Enter valid url and try again.')

        for i in range(n_split):
            if query_idx < (len(link_list)//n_split)*(i+1):
                title_embeddings = np.load('./data/title_embeddings{}.npy'.format(i+1))
                abst_embeddings = np.load('./data/abst_embeddings{}.npy'.format(i+1))
                query1 = title_embeddings[query_idx-(len(link_list)//n_split)*i]
                query2 = abst_embeddings[query_idx-(len(link_list)//n_split)*i]
                if use_title and use_abst:
                    query = np.concatenate([query1, query2], axis=0)
                elif use_title:
                    query = query1.copy()
                elif use_abst:
                    query = query2.copy()
                else:
                    raise ValueError('Check either title or abstract and try again.')
                del query1
                del query2
                break

        sim_list = []
        for i in range(n_split):
            title_embeddings = np.load('./data/title_embeddings{}.npy'.format(i+1))
            abst_embeddings = np.load('./data/abst_embeddings{}.npy'.format(i+1))

            if use_title and use_abst:
                embeddings = np.concatenate([title_embeddings, abst_embeddings], axis=1)
            elif use_title:
                embeddings = title_embeddings.copy()
            elif use_abst:
                embeddings = abst_embeddings.copy()
            else:
                raise ValueError('Check either title or abstract and try again.')
            del title_embeddings
            del abst_embeddings

            for vector in embeddings:
                sim_list.append(cos_similarity(query, vector))
            gc.collect()
        
        sim_list = np.array(sim_list)
        sim_idx = np.argsort(sim_list)[::-1]
        for i in range(1, top_n+1):
            result_title, result_abst = retrieval(link_list[sim_idx[i]])
            st.write('### \# {}'.format(i))
            st.write('Similarlity: {}'.format(round(sim_list[sim_idx[i]], 4)))
            st.write('Title: {}'.format(result_title))
            st.write('Link: {}'.format(link_list[sim_idx[i]]))
            st.write('Abstract: {}'.format(result_abst))


if __name__ == '__main__':
    main()