from typing import Any, List, Optional

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle
import torch
import numpy as np

def sigmoid(_outputs):
    return 1.0 / (1.0 + np.exp(-_outputs))

class OptimumRerank(BaseNodePostprocessor):
    model: str = Field(description="Sentence transformer model name.")
    top_n: int = Field(description="Number of nodes to return sorted by score.")
    keep_retrieval_score: bool = Field(
        default=False,
        description="Whether to keep the retrieval score in metadata.",
    )

    max_length: int
    
    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()

    def __init__(
        self,
        max_length: int,
        model: str,
        top_n: int = 2,
        keep_retrieval_score: Optional[bool] = False,
    ):
        try:
            from transformers import AutoTokenizer
            from optimum.onnxruntime import ORTModelForSequenceClassification
        except ImportError:
            raise ImportError(
                "Cannot import sentence-transformers or torch package,",
                "please `pip install torch sentence-transformers`",
            )
        
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = ORTModelForSequenceClassification.from_pretrained(model)
        
        super().__init__(
            top_n=top_n,
            model=model,
            keep_retrieval_score=keep_retrieval_score,
            max_length=max_length
        )

    @classmethod
    def class_name(cls) -> str:
        return "OptimumRerank"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0:
            return []

        query_and_nodes = [
            (
                query_bundle.query_str,
                node.node.get_content(metadata_mode=MetadataMode.LLM),
            )
            for node in nodes
        ]

        with self.callback_manager.event(
            CBEventType.RERANKING,
            payload={
                EventPayload.NODES: nodes,
                EventPayload.MODEL_NAME: self.model,
                EventPayload.QUERY_STR: query_bundle.query_str,
                EventPayload.TOP_K: self.top_n,
            },
        ) as event:
            warmup_features = self._tokenizer(query_and_nodes[:10], padding=True, truncation="longest_first", return_tensors="pt", max_length=self.max_length)
            features = self._tokenizer(query_and_nodes, padding=True,  truncation="longest_first", return_tensors="pt", max_length=self.max_length)

            with torch.no_grad():
                self._model(**warmup_features).logits
                outputs = self._model(**features)
                outputs = outputs["logits"][0]
                outputs = outputs.numpy()
                
            scores = sigmoid(outputs)

            assert len(scores) == len(nodes)

            for node, score in zip(nodes, scores):
                if self.keep_retrieval_score:
                    # keep the retrieval score in metadata
                    node.node.metadata["retrieval_score"] = node.score
                node.score = score

            new_nodes = sorted(nodes, key=lambda x: -x.score if x.score else 0)[
                : self.top_n
            ]
            event.on_end(payload={EventPayload.NODES: new_nodes})

        return new_nodes