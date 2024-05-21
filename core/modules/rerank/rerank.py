from typing import Any, List, Optional

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle
import torch
import numpy as np

def sigmoid(_outputs):
    """
     @brief Sigmoid function for classification. The sigmoid function is defined as 1 / ( 1 + exp ( - _outputs ))
     @param _outputs A list of outputs. Each element is a numpy array with shape [ batch_size n_classes ]
     @return A numpy array with shape [ batch_size n_classes ] where each element is a numpy array
    """
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
        """
         @brief Initialize class. This is the constructor for the SequenceClassification class. You can call it yourself if you don't know what you are doing.
         @param max_length Maximum length of the sentence to be tokenized
         @param model Name of the model that will be used
         @param top_n Number of top documents to be
         @param keep_retrieval_score
        """
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
        """
         @brief Returns the name of the class. This is used to generate the class name when it is known to be a class
         @param cls The class that we are looking for
         @return The name of the class to be used in the API's class_name attribute or " OptimumRerank
        """
        return "OptimumRerank"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """
         @brief Postprocess nodes to be used by LSTM. This is a helper method to perform postprocessing on a list of nodes and return a list of node with scores
         @param nodes List of nodes to be postprocessed
         @param query_bundle Query bundle containing query string information.
         @return List of nodes with score ( s ) after postprocessing. Note that scores are in descending order of score
        """
        # Raise ValueError if the query bundle is not set.
        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        # Returns a list of nodes in the list.
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
                outputs = outputs["logits"]
                outputs = outputs.numpy()
                outputs = outputs.reshape(1, -1)
                outputs = outputs[0]
                
            scores = sigmoid(outputs)

            assert len(scores) == len(nodes)

            # keep the retrieval score in metadata
            for node, score in zip(nodes, scores):
                # keep the retrieval score in metadata
                if self.keep_retrieval_score:
                    # keep the retrieval score in metadata
                    node.node.metadata["retrieval_score"] = node.score
                node.score = score

            new_nodes = sorted(nodes, key=lambda x: -x.score if x.score else 0)[
                : self.top_n
            ]
            event.on_end(payload={EventPayload.NODES: new_nodes})

        return new_nodes