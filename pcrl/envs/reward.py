import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import evaluate
import numpy as np
from data.instruction_pool import FIXED_TOKENS
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


def extract_values(dictionaries: List[Dict[str, Any]], key: str) -> List[Any]:
    """Safely extracts a list of values by key from a list of dictionaries.

    Avoids overriding the built-in 'dict' name.
    """
    return [d[key] for d in dictionaries if key in d]


class RewardFunction(ABC):
    """Abstract base class for all reward functions."""

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Flexible signature to support both text-based and contextual rewards."""
        raise NotImplementedError


class BleuRewardFunction(RewardFunction):
    """Reward function based on the BLEU metric, tailored specifically for source code."""

    def __init__(self, coef: float = 1.0) -> None:
        super().__init__()
        self._coef = coef
        # method1 smoothing prevents BLEU from dropping to 0 if there are no 4-gram matches
        self._smoothing = SmoothingFunction().method1

    @staticmethod
    def _tokenize_code(code: str) -> List[str]:
        """Lexical tokenizer for code structures.

        Separates identifiers, keywords, operators, and punctuation marks.
        """
        return re.findall(r"\w+|[^\w\s]", code)

    def __call__(
        self, generated_texts: List[str], reference_texts: List[str]
    ) -> np.ndarray:
        assert len(generated_texts) == len(reference_texts), (
            "The lengths of generated_texts and reference_texts must match."
        )

        scores = []
        for gen, ref in zip(generated_texts, reference_texts):
            gen_tokens = self._tokenize_code(gen)
            ref_tokens = self._tokenize_code(ref)

            # NLTK expects a list of reference lists: [[ref_tokens]]
            score = sentence_bleu(
                [ref_tokens], gen_tokens, smoothing_function=self._smoothing
            )
            scores.append(score)

        return self._coef * np.array(scores)


class MeteorRewardFunction(RewardFunction):
    """Reward function based on the METEOR metric."""

    def __init__(self, coef: float) -> None:
        super().__init__()
        self._metric = evaluate.load("meteor", keep_in_memory=True)
        self._coef = coef

    def __call__(
        self, generated_texts: List[str], reference_texts: List[str]
    ) -> np.ndarray:
        assert len(generated_texts) == len(reference_texts)
        scores = []
        for gen, ref in zip(generated_texts, reference_texts):
            score = self._metric.compute(predictions=[gen], references=[ref])[
                "meteor"
            ]
            scores.append(score)
        return self._coef * np.array(scores)


class RougeRewardFunction(RewardFunction):
    """Reward function based on the ROUGE metric (supports rouge1, rouge2, rougeL)."""

    def __init__(self, rouge_type: str, coef: float) -> None:
        super().__init__()
        self._metric = evaluate.load("rouge", keep_in_memory=True)
        self._rouge_type = rouge_type
        self._coef = coef

    def __call__(
        self, generated_texts: List[str], reference_texts: List[str]
    ) -> np.ndarray:
        assert len(generated_texts) == len(reference_texts)
        score = self._metric.compute(
            predictions=generated_texts,
            references=reference_texts,
            use_aggregator=False,
        )
        return self._coef * np.array(score[self._rouge_type])


class CombineRewardFunction(RewardFunction):
    """Complex contextual reward function (SCST) balancing text similarity and compression rate."""

    def __init__(self, lamb: float, threshold: float, rouge_type: str) -> None:
        super().__init__()
        self._metric = evaluate.load("rouge", keep_in_memory=True)
        self._lambda = lamb
        self._threshold = threshold
        self._rouge_type = rouge_type

    def get_compress_ratio(
        self,
        infos: List[Dict[str, Any]],
        compressed_counts: List[int],
        fixed_counts: Dict[str, Any],
    ) -> np.ndarray:
        base_counts = extract_values(infos, "base_token_counts")
        fixed_n_tokens = np.zeros(len(infos), dtype=np.int16)

        for i, info in enumerate(infos):
            for k, v in FIXED_TOKENS.items():
                if v in info["base_prompt"]:
                    fixed_n_tokens[i] += len(fixed_counts[k])

        compress_rate = 1 - (
            np.array(compressed_counts) - fixed_n_tokens
        ) / (np.array(base_counts) - fixed_n_tokens)
        return compress_rate

    def __call__(
        self,
        infos: List[Dict[str, Any]],
        gen_output: Dict[str, Any],
        fixed_tokens_dict: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        similarity = np.array(
            self._metric.compute(
                predictions=extract_values(infos, "base_gen_texts"),
                references=gen_output["gen_texts"],
                use_aggregator=False,
            )[self._rouge_type]
        )

        compress_rate = self.get_compress_ratio(
            infos, gen_output["compressed_token_counts"], fixed_tokens_dict
        )

        rw_info = {"sim": similarity, "comp": compress_rate}
        rewards = np.where(
            rw_info["sim"] > self._threshold, rw_info["comp"], self._lambda
        )

        return rewards, rw_info