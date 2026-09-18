from fastembed import TextEmbedding


class Embedder:
    """Local embedding model (ONNX via fastembed). No API key required."""

    def __init__(self, model_name: str) -> None:
        self._model = TextEmbedding(model_name=model_name)
        self.dim = len(list(self._model.embed(["dimension probe"]))[0])

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [vector.tolist() for vector in self._model.embed(texts)]
