from typing import List, Dict, Tuple
import json
import os

import faiss
import torch
import tqdm
import numpy as np
from sentence_transformers.models import Transformer, Pooling
from sentence_transformers import SentenceTransformer


corpus_names = {
    "MOC": ["msd", "pubmed"], # Mixture of Corpus
    "PubMed": ["pubmed"],
    "MSD": ["msd"],
    "Textbooks": ["textbooks"],
}

retriever_names = {
    "MedCPT": "ncbi/MedCPT-Query-Encoder",
}

retriever_paths = {
    "MedCPT": "retriever/sentence_transformers/ncbi_MedCPT-Query-Encoder",
}


class CustomizeSentenceTransformer(SentenceTransformer): # change the default pooling "MEAN" to "CLS"
    def _load_auto_model(self, model_name_or_path, **kwargs):
        """
        Creates a simple Transformer + CLS Pooling model and returns the modules
        """
        print("No sentence-transformers model found with name {}. Creating a new one with CLS pooling.".format(model_name_or_path))
        transformer_model = Transformer(model_name_or_path)
        pooling_model = Pooling(transformer_model.get_word_embedding_dimension(), 'cls')
        return [transformer_model, pooling_model]


def embed(chunk_dir, index_dir, model_name, **kwargs):
    save_dir = os.path.join(index_dir, "embedding")
    model = CustomizeSentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    fnames = sorted([fname for fname in os.listdir(chunk_dir) if fname.endswith(".json") or fname.endswith(".jsonl")])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for fname in tqdm.tqdm(fnames):
            fpath = os.path.join(chunk_dir, fname)
            save_path = os.path.join(save_dir, fname.replace(".json", ".npy") if fname.endswith(".json") else fname.replace(".jsonl", ".npy"))
            if os.path.exists(save_path):
                continue
            if open(fpath).read().strip() == "":
                continue
            # texts = [json.loads(item) for item in open(fpath).read().strip().split('\n')]
            # texts = [[item["title"], item["content"]] for item in texts]
            texts = [json.loads(open(fpath).read())]
            texts = [[item["title"], item["content"]] for item in texts]
            embed_chunks = model.encode(texts, **kwargs)
            np.save(save_path, embed_chunks)
        embed_chunks = model.encode([""], **kwargs)
    return embed_chunks.shape[-1]

def construct_index(index_dir, h_dim=768):
    with open(os.path.join(index_dir, "metadatas.jsonl"), 'w') as f:
        f.write("")

    index = faiss.IndexFlatIP(h_dim)

    for fname in tqdm.tqdm(sorted(os.listdir(os.path.join(index_dir, "embedding")))):
        curr_embed = np.load(os.path.join(index_dir, "embedding", fname))
        index.add(curr_embed)
        with open(os.path.join(index_dir, "metadatas.jsonl"), 'a+') as f:
            f.write("\n".join([json.dumps({'index': i, 'source': fname.replace(".npy", "")}) for i in range(len(curr_embed))]) + '\n')

    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))
    return index


class Retriever: 
    def __init__(self, retriever_name: str="ncbi/MedCPT-Query-Encoder", retriever_path: str=None, corpus_name: str="textbooks", corpus_dir: str="./corpus", **kwargs):
        self.retriever_name = retriever_name
        self.retriever_path = retriever_path
        self.corpus_name = corpus_name

        self.corpus_dir = corpus_dir
        if not os.path.exists(self.corpus_dir):
            os.makedirs(self.corpus_dir)
        self.chunk_dir = os.path.join(self.corpus_dir, self.corpus_name, "chunk")
        self.index_dir = os.path.join(self.corpus_dir, self.corpus_name, "index", self.retriever_name.replace("Query-Encoder", "Article-Encoder"))
        if os.path.exists(os.path.join(self.index_dir, "faiss.index")):
            print("[In progress] Loading the {:s} corpus with the {:s} retriever...".format(self.corpus_name, self.retriever_name.replace("Query-Encoder", "Article-Encoder")))
            self.index = faiss.read_index(os.path.join(self.index_dir, "faiss.index"))
            self.metadatas = [json.loads(line) for line in open(os.path.join(self.index_dir, "metadatas.jsonl")).read().strip().split('\n')]
            print("[Finished] Corpus loading finished!")
        else:
            print("[In progress] Embedding the {:s} corpus with the {:s} retriever...".format(self.corpus_name, self.retriever_name.replace("Query-Encoder", "Article-Encoder")))
            if os.path.exists(self.retriever_path.replace("Query-Encoder", "Article-Encoder")):
                h_dim = embed(chunk_dir=self.chunk_dir, index_dir=self.index_dir, model_name=self.retriever_path.replace("Query-Encoder", "Article-Encoder"), **kwargs)
            else:
                h_dim = embed(chunk_dir=self.chunk_dir, index_dir=self.index_dir, model_name=self.retriever_name.replace("Query-Encoder", "Article-Encoder"), **kwargs)
            print("[In progress] Embedding finished! The dimension of the embeddings is {:d}.".format(h_dim))
            self.index = construct_index(index_dir=self.index_dir, h_dim=h_dim)
            print("[Finished] Corpus indexing finished!")
            self.metadatas = [json.loads(line) for line in open(os.path.join(self.index_dir, "metadatas.jsonl")).read().strip().split('\n')]
        if os.path.exists(self.retriever_path):
            self.embedding_function = CustomizeSentenceTransformer(self.retriever_path, device="cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.embedding_function = CustomizeSentenceTransformer(self.retriever_name, device="cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_function.eval()

    def get_relevant_documents(self, question, k=32, **kwarg):
        assert type(question) == str
        question = [question]

        with torch.no_grad():
            query_embed = self.embedding_function.encode(question, **kwarg)
        res_ = self.index.search(query_embed, k=k)
        indices = [self.metadatas[i] for i in res_[1][0]]

        texts = self.idx2txt(indices)
        embeddings = self.idx2embedding(indices)
        scores = res_[0][0].tolist()
        return texts, embeddings, scores

    def idx2txt(self, indices) -> List[Dict[str, str]]:
        '''
        Input: List of Dict( {"source": str, "index": int} )
        Output: List of str
        '''
        texts = []
        for i in indices:
            if os.path.exists(os.path.join(self.chunk_dir, i["source"]+".json")):
                texts.append(json.loads(open(os.path.join(self.chunk_dir, i["source"]+".json")).read()))
            else:
                texts.append(json.loads(open(os.path.join(self.chunk_dir, i["source"]+".jsonl")).read().strip().split('\n')[i["index"]]))
        return texts

    def idx2embedding(self, indices) -> List[np.ndarray]:
        '''
        Input: List of Dict( {"source": str, "index": int} )
        Output: List of Embedding
        '''
        return [np.load(os.path.join(self.index_dir, "embedding", i["source"]+".npy"))[i["index"]] for i in indices]


class RetrievalSystem:
    def __init__(self, retriever_name="MedCPT", corpus_name="MOC", corpus_dir="./corpus"):
        assert retriever_name in retriever_names
        assert corpus_name in corpus_names
        self.retriever_name = retriever_names[retriever_name]
        self.retriever_path = retriever_paths[retriever_name]
        self.corpus_names = corpus_names[corpus_name]
        self.retrievers = [Retriever(self.retriever_name, self.retriever_path, corpus_name, corpus_dir) for corpus_name in self.corpus_names]
    
    def retrieve(self, question, k=3) -> Tuple[List[Dict[str, str]], List[np.ndarray], List[float]]:
        '''
            Given questions, return the relevant snippets from the corpus
        '''
        assert type(question) == str
        
        retrieve_texts = []
        retrieve_embeddings = []
        retrieve_scores = []

        for retriever in self.retrievers:
            texts, embeddings, scores = retriever.get_relevant_documents(question, k=k)
            retrieve_texts += texts
            retrieve_embeddings += embeddings
            retrieve_scores += scores

        return retrieve_texts, retrieve_embeddings, retrieve_scores