import dashscope
from dashscope import TextEmbedding
import numpy as np
from typing import Union, List
import logging
import os
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs/emb_config.yaml"), 'r') as file:
    ds_config = yaml.safe_load(file).get("dashscope_config")


def generate_embedding(text, embedding_model="dashscope", **kwargs):
    # todo: support more embedding model in the future, e.g., m3e model.
    dashscope.api_key = ds_config["api_key"]
    dashscope.base_http_api_url = ds_config['base_http_api_url']
    try:
        rsp = TextEmbedding.call(model=TextEmbedding.Models.text_embedding_v1, input=text)
        embeddings = np.array([record['embedding'] for record in rsp.output['embeddings']])
        if isinstance(text, str):
            embeddings = embeddings[0]
    except TypeError as e:
        logger.warning(f"Request dashscope embedding service failed, error info {e}")
        embeddings = None
    return embeddings


def cosine_distance(a, b):
    """
    Only support `a` is an embedding vector, `b` is a vector or matrix.
    """
    dist = np.dot(a, b.T) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return dist


def l2_distance(a, b):
    dist = np.linalg.norm(a - b, axis=-1)
    return dist


def similarity_match(query: str, corpus: Union[str, List], dist_type="cosine"):
    if dist_type not in ("cosine", "l2"):
        logger.warning(f"invalid input distance type, {dist_type}, setting to cosine distance")
        dist_type = "cosine"

    query_emb = generate_embedding(query)
    corpus_emb = generate_embedding(corpus)

    if query_emb is not None and corpus_emb is not None:
        if dist_type == "l2":
            return l2_distance(query_emb, corpus_emb)
        else:
            return cosine_distance(query_emb, corpus_emb)
    else:
        return None


if __name__ == "__main__":
    queries = "请问贵州茅台最近股价如何"
    context = ["完美世界近期市场波动较大", "茅台和五粮液作为消费龙头, 2020年整体表现优于沪深300指数"]
    print(similarity_match(queries, context))
