import numpy as np
from hmmlearn.hmm import CategoricalHMM

OBS_MAP = {"High": 0, "Low": 1}
INV_OBS_MAP = {v: k for k, v in OBS_MAP.items()}


def build_ev_model() -> CategoricalHMM:
    """Creates an HMM for the word 'EV'.

    Hidden states:
        0 -> e
        1 -> v

    Observations:
        0 -> High
        1 -> Low
    """
    model = CategoricalHMM(n_components=2, init_params="")
    model.startprob_ = np.array([1.0, 0.0])
    model.transmat_ = np.array([
        [0.6, 0.4],
        [0.2, 0.8],
    ])
    model.emissionprob_ = np.array([
        [0.7, 0.3],
        [0.1, 0.9],
    ])
    return model


def build_okul_model() -> CategoricalHMM:
    """Creates a toy HMM for the word 'OKUL'.

    States roughly represent the phoneme flow o-k-u-l.
    Observation symbols are still simplified into two categories:
        0 -> High
        1 -> Low
    """
    model = CategoricalHMM(n_components=4, init_params="")
    model.startprob_ = np.array([1.0, 0.0, 0.0, 0.0])
    model.transmat_ = np.array([
        [0.60, 0.40, 0.00, 0.00],
        [0.00, 0.60, 0.40, 0.00],
        [0.00, 0.00, 0.55, 0.45],
        [0.00, 0.00, 0.00, 1.00],
    ])
    model.emissionprob_ = np.array([
        [0.75, 0.25],  # o
        [0.25, 0.75],  # k
        [0.70, 0.30],  # u
        [0.20, 0.80],  # l
    ])
    return model


def prepare_sequences(list_of_sequences: list[list[int]]) -> tuple[np.ndarray, list[int]]:
    """Converts observation sequences to hmmlearn input format."""
    arrays = [np.array(seq, dtype=int).reshape(-1, 1) for seq in list_of_sequences]
    X = np.vstack(arrays)
    lengths = [len(seq) for seq in list_of_sequences]
    return X, lengths


def evaluate_models(test_sequence: list[int] | list[str]) -> dict[str, float]:
    """Scores a new observation sequence against both word models.

    Accepts either integer-coded observations [0, 1, ...]
    or label form ['High', 'Low', ...].
    """
    if not test_sequence:
        raise ValueError("test_sequence cannot be empty")

    if isinstance(test_sequence[0], str):
        encoded = [OBS_MAP[item] for item in test_sequence]
    else:
        encoded = [int(item) for item in test_sequence]

    X_test = np.array(encoded, dtype=int).reshape(-1, 1)

    model_ev = build_ev_model()
    model_okul = build_okul_model()

    score_ev = model_ev.score(X_test)
    score_okul = model_okul.score(X_test)

    return {
        "EV": score_ev,
        "OKUL": score_okul,
    }


def classify_word(test_sequence: list[int] | list[str]) -> tuple[str, dict[str, float]]:
    scores = evaluate_models(test_sequence)
    best_label = max(scores, key=scores.get)
    return best_label, scores


def demo() -> None:
    # Training data placeholders. In a real project these would be used to estimate parameters.
    ev_train = [
        [0, 1],
        [0, 0, 1],
        [0, 1, 1],
    ]
    okul_train = [
        [0, 1, 0, 1],
        [0, 1, 1, 1],
        [0, 0, 1, 1],
    ]

    X_ev, lengths_ev = prepare_sequences(ev_train)
    X_okul, lengths_okul = prepare_sequences(okul_train)

    print("EV eğitim matrisi şekli:", X_ev.shape, "uzunluklar:", lengths_ev)
    print("OKUL eğitim matrisi şekli:", X_okul.shape, "uzunluklar:", lengths_okul)

    test_sequence = ["High", "Low"]
    prediction, scores = classify_word(test_sequence)

    print("\nTest gözlem dizisi:", test_sequence)
    print("Log-Likelihood skorları:")
    for label, score in scores.items():
        print(f"  {label}: {score:.6f}")
    print("Tahmin edilen kelime:", prediction)


if __name__ == "__main__":
    demo()
