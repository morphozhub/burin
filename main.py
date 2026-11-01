from sn17gen.pipeline import run


def generate(image: bytes, seed: int, witness, renderer) -> str | None:
    return run(image, seed, witness, renderer)
