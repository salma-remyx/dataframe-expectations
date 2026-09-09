"""Neurosymbolic association rule mining from one-hot encoded tabular data.

Adapted from "Neurosymbolic Association Rule Mining from Tabular Data" (Aerial+,
arXiv:2504.19354). The paper trains an under-complete autoencoder on one-hot
encoded tabular data and extracts rules from the model's reconstruction
mechanism: test vectors with the antecedent categories marked (set to 1) are
passed through the network, and categories whose reconstructed probability
clears a threshold become consequents. The under-complete bottleneck forces the
network to learn associations between features rather than the identity
mapping, which yields concise rule sets without the combinatorial rule
explosion of exhaustive miners such as Apriori or FP-Growth.

Substitutions relative to the paper, made to fit this package's minimal,
torch-free dependency posture:

- The paper's PyTorch autoencoder (pyaerial) is re-implemented in NumPy. The
  architecture and hyperparameters follow the paper: hidden layers halving in
  size, tanh activations, one softmax output group per input feature, Adam with
  learning rate 5e-3, and 1-2 training epochs.
- The paper's BCE reconstruction loss is replaced with per-feature
  cross-entropy, the natural pairing for the per-feature softmax outputs.
- The paper's downstream classifier experiments are out of scope for a
  DataFrame-validation library.
"""

import itertools
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from pandas import DataFrame as PandasDataFrame


@dataclass(frozen=True)
class MinedRule:
    """A rule extracted from the autoencoder's reconstruction mechanism.

    :ivar antecedent: ``(column, value)`` pairs forming the "if" side.
    :ivar consequent: A single ``(column, value)`` pair — the "then" side.
    :ivar support: Fraction of rows where antecedent and consequent jointly hold.
    :ivar confidence: Conditional probability P(consequent | antecedent) on data.
    :ivar reconstruction: Continuous probability in [0, 1] that the trained model
        reconstructs for the consequent when the antecedent categories are
        marked — the neurosymbolic evidence for the rule.
    """

    antecedent: Tuple[Tuple[str, Any], ...]
    consequent: Tuple[str, Any]
    support: float
    confidence: float
    reconstruction: float


def _one_hot_encode(
    data_frame: PandasDataFrame, columns: Sequence[str], max_cardinality: int
) -> Tuple[np.ndarray, List[str], List[List[Any]], List[int]]:
    """One-hot encode ``columns`` of a pandas DataFrame into a 0/1 matrix.

    Columns with no values or more distinct values than ``max_cardinality``
    (e.g. identifiers) are skipped to avoid input explosion.

    :param data_frame: Reference pandas DataFrame.
    :param columns: Columns to encode.
    :param max_cardinality: Skip columns whose distinct-value count exceeds this.
    :return: Tuple of (0/1 matrix, kept column names, per-column category lists,
        per-column offsets into the matrix's column axis).
    """
    kept_columns: List[str] = []
    categories: List[List[Any]] = []
    blocks: List[np.ndarray] = []
    offsets: List[int] = []
    offset = 0
    for column in columns:
        uniques = data_frame[column].dropna().unique().tolist()
        if not uniques or len(uniques) > max_cardinality:
            continue
        uniques = sorted(uniques, key=repr)  # deterministic category order
        block = np.zeros((len(data_frame), len(uniques)))
        values = data_frame[column].to_numpy()
        for category_index, category in enumerate(uniques):
            block[:, category_index] = values == category
        kept_columns.append(column)
        categories.append(uniques)
        blocks.append(block)
        offsets.append(offset)
        offset += len(uniques)
    matrix = np.concatenate(blocks, axis=1) if blocks else np.zeros((len(data_frame), 0))
    return matrix, kept_columns, categories, offsets


def _hidden_layer_sizes(input_dim: int) -> List[int]:
    """Hidden layer sizes halving per layer, ending in an under-complete bottleneck."""
    sizes: List[int] = []
    current = input_dim // 2
    while current >= 2 and len(sizes) < 2:
        sizes.append(current)
        current //= 2
    return sizes or [1]


class UnderCompleteAutoEncoder:
    """NumPy under-complete autoencoder over per-feature softmax groups.

    The architecture follows Aerial+ (arXiv:2504.19354): hidden layers halving
    in size with tanh activations, and one softmax output group per input
    feature. It is trained with Adam (learning rate 5e-3) on per-feature
    cross-entropy, which substitutes the paper's BCE loss as the natural
    pairing for the softmax outputs.
    """

    def __init__(self, feature_sizes: Sequence[int], random_state: Optional[int] = None) -> None:
        """Initialize weights for the given per-feature category counts.

        :param feature_sizes: Number of categories per encoded feature.
        :param random_state: Seed for weight initialization and batch shuffling.
        """
        input_dim = int(sum(feature_sizes))
        if input_dim == 0:
            raise ValueError("AutoEncoder needs at least one input category")
        self._slices: List[slice] = []
        start = 0
        for size in feature_sizes:
            self._slices.append(slice(start, start + size))
            start += size
        hidden = _hidden_layer_sizes(input_dim)
        dims = [input_dim, *hidden, *hidden[:-1][::-1], input_dim]
        self._rng = np.random.default_rng(random_state)
        self._weights = [
            self._rng.normal(0.0, np.sqrt(1.0 / fan_in), size=(fan_in, fan_out))
            for fan_in, fan_out in zip(dims[:-1], dims[1:])
        ]
        self._biases = [np.zeros(dim) for dim in dims[1:]]
        parameters = [*self._weights, *self._biases]
        self._adam_m = [np.zeros_like(parameter) for parameter in parameters]
        self._adam_v = [np.zeros_like(parameter) for parameter in parameters]
        self._adam_step = 0

    def _forward(self, matrix: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """Return per-layer activations and the per-feature softmax probabilities."""
        activations = [matrix]
        output = matrix
        for weight, bias in zip(self._weights[:-1], self._biases[:-1]):
            output = np.tanh(output @ weight + bias)
            activations.append(output)
        logits = output @ self._weights[-1] + self._biases[-1]
        probabilities = np.empty_like(logits)
        for feature_slice in self._slices:
            group = logits[:, feature_slice]
            exp = np.exp(group - group.max(axis=1, keepdims=True))
            probabilities[:, feature_slice] = exp / exp.sum(axis=1, keepdims=True)
        return activations, probabilities

    def reconstruct(self, matrix: np.ndarray) -> np.ndarray:
        """Per-category reconstructed probabilities for the given 0/1 rows.

        :param matrix: Rows of one-hot encoded (possibly antecedent-marked) data.
        :return: Reconstruction probabilities with the same shape as ``matrix``.
        """
        _, probabilities = self._forward(np.asarray(matrix, dtype=float))
        return probabilities

    def fit(
        self,
        matrix: np.ndarray,
        epochs: int = 2,
        learning_rate: float = 5e-3,
        batch_size: int = 32,
    ) -> None:
        """Train the autoencoder on the one-hot encoded rows with mini-batch Adam.

        :param matrix: 0/1 one-hot encoded training data.
        :param epochs: Number of passes over the data (the paper uses 1-2).
        :param learning_rate: Adam learning rate (the paper uses 5e-3).
        :param batch_size: Mini-batch size for each gradient step.
        """
        data = np.asarray(matrix, dtype=float)
        row_count = len(data)
        if row_count == 0:
            return
        batch_size = max(1, min(batch_size, row_count))
        for _ in range(epochs):
            order = self._rng.permutation(row_count)
            for start in range(0, row_count, batch_size):
                self._train_step(data[order[start : start + batch_size]], learning_rate)

    def _train_step(self, batch: np.ndarray, learning_rate: float) -> None:
        activations, probabilities = self._forward(batch)
        batch_len = len(batch)
        # d loss / d logits: (p - y) per softmax group, scaled by feature count.
        delta = (probabilities - batch) / len(self._slices)
        layer_count = len(self._weights)
        grad_weights: List[np.ndarray] = [np.empty(0)] * layer_count
        grad_biases: List[np.ndarray] = [np.empty(0)] * layer_count
        for layer in range(layer_count - 1, -1, -1):
            grad_weights[layer] = activations[layer].T @ delta / batch_len
            grad_biases[layer] = delta.mean(axis=0)
            if layer > 0:
                previous = activations[layer]
                delta = (delta @ self._weights[layer].T) * (1.0 - previous**2)
        self._adam_update(
            [*self._weights, *self._biases], [*grad_weights, *grad_biases], learning_rate
        )

    def _adam_update(
        self, parameters: List[np.ndarray], gradients: List[np.ndarray], learning_rate: float
    ) -> None:
        self._adam_step += 1
        for index, (parameter, gradient) in enumerate(zip(parameters, gradients)):
            self._adam_m[index] = 0.9 * self._adam_m[index] + 0.1 * gradient
            self._adam_v[index] = 0.999 * self._adam_v[index] + 0.001 * gradient**2
            m_hat = self._adam_m[index] / (1.0 - 0.9**self._adam_step)
            v_hat = self._adam_v[index] / (1.0 - 0.999**self._adam_step)
            parameter -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)


def mine_rules_neurosymbolic(
    data_frame: PandasDataFrame,
    columns: Optional[List[str]] = None,
    min_support: float = 0.05,
    min_confidence: float = 0.8,
    max_antecedent_size: int = 2,
    max_cardinality: int = 50,
    max_rules: Optional[int] = None,
    epochs: int = 2,
    random_state: Optional[int] = None,
) -> List[MinedRule]:
    """Mine association rules with Aerial+'s reconstruction-based extraction.

    An :class:`UnderCompleteAutoEncoder` is trained on the one-hot encoded
    frame. Each frequent antecedent itemset (support >= ``min_support``) is
    then marked in an all-zero test vector and forward-passed; consequent
    categories whose continuous reconstructed probability clears
    ``min_confidence`` become rules. Rule support and confidence are computed
    from the data; the reconstruction probability is kept as continuous
    neurosymbolic evidence on each rule.

    :param data_frame: Reference pandas DataFrame to mine rules from.
    :param columns: Columns to consider. Defaults to every column. Columns with
        more distinct values than ``max_cardinality`` are skipped.
    :param min_support: Minimum fraction of rows an antecedent must cover.
    :param min_confidence: Minimum reconstructed probability to emit a rule
        (the paper's consequent threshold, tau_c=0.8).
    :param max_antecedent_size: Largest antecedent itemset size to explore.
    :param max_cardinality: Skip columns whose distinct-value count exceeds this.
    :param max_rules: If set, keep only this many highest-confidence rules.
    :param epochs: Autoencoder training epochs (the paper uses 1-2).
    :param random_state: Seed for the autoencoder; set for deterministic mining.
    :return: Mined rules sorted by confidence (desc) then support (desc).
    """
    if min_support <= 0 or min_support > 1:
        raise ValueError(f"min_support must be in (0, 1], got {min_support}")
    if min_confidence <= 0 or min_confidence > 1:
        raise ValueError(f"min_confidence must be in (0, 1], got {min_confidence}")

    selected_columns = columns if columns is not None else list(data_frame.columns)
    total_rows = len(data_frame)
    if total_rows == 0:
        return []
    matrix, kept_columns, categories, offsets = _one_hot_encode(
        data_frame, selected_columns, max_cardinality
    )
    if matrix.shape[1] == 0:
        return []

    # Frequent items: (column position, category, flat index into the matrix).
    items: List[Tuple[int, Any, int]] = []
    for position, (column_categories, offset) in enumerate(zip(categories, offsets)):
        for category_index, category in enumerate(column_categories):
            flat_index = offset + category_index
            if matrix[:, flat_index].sum() / total_rows >= min_support:
                items.append((position, category, flat_index))
    if not items:
        return []

    model = UnderCompleteAutoEncoder(
        feature_sizes=[len(column_categories) for column_categories in categories],
        random_state=random_state,
    )
    model.fit(matrix, epochs=epochs)

    # Antecedent candidates: (item combo, boolean row mask where the combo holds).
    candidates: List[Tuple[Tuple[Tuple[int, Any, int], ...], Any]] = []
    for size in range(1, max_antecedent_size + 1):
        for combo in itertools.combinations(items, size):
            # Items in one itemset must come from distinct columns.
            if len({position for position, _, _ in combo}) != size:
                continue
            rows = matrix[:, [flat_index for _, _, flat_index in combo]].all(axis=1)
            if rows.sum() / total_rows < min_support:
                continue
            candidates.append((combo, rows))
    if not candidates:
        return []

    # Mark each antecedent in an all-zero test vector and forward-pass once.
    test_vectors = np.zeros((len(candidates), matrix.shape[1]))
    for row_index, (combo, _) in enumerate(candidates):
        test_vectors[row_index, [flat_index for _, _, flat_index in combo]] = 1.0
    reconstructed = model.reconstruct(test_vectors)

    rules: List[MinedRule] = []
    for row_index, (combo, rows) in enumerate(candidates):
        antecedent_columns = {position for position, _, _ in combo}
        antecedent_count = int(rows.sum())
        for position, category, flat_index in items:
            if position in antecedent_columns:
                continue
            reconstruction = float(reconstructed[row_index, flat_index])
            if reconstruction < min_confidence:
                continue
            joint_count = int((rows & (matrix[:, flat_index] == 1.0)).sum())
            if joint_count == 0:
                continue
            rules.append(
                MinedRule(
                    antecedent=tuple(
                        (kept_columns[item_position], value) for item_position, value, _ in combo
                    ),
                    consequent=(kept_columns[position], category),
                    support=joint_count / total_rows,
                    confidence=joint_count / antecedent_count,
                    reconstruction=reconstruction,
                )
            )

    rules.sort(key=lambda rule: (rule.confidence, rule.support), reverse=True)
    if max_rules is not None:
        rules = rules[:max_rules]
    return rules
