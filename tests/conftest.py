import pytest

TINY = "HuggingFaceTB/SmolLM2-135M-Instruct"
SMALL = "Qwen/Qwen2.5-0.5B-Instruct"
HYBRID = "Qwen/Qwen3.5-0.8B"  # Gated DeltaNet + gated attention


def cached(name):
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(name, local_files_only=True, allow_patterns=["*.json", "*.safetensors"])
        return True
    except Exception:  # noqa: BLE001 - any failure means "not cached"
        return False


def require(name):
    if not cached(name):
        pytest.skip(f"{name} is not downloaded")
    return name


@pytest.fixture(scope="session")
def tiny():
    return require(TINY)


@pytest.fixture(scope="session")
def small():
    return require(SMALL)


@pytest.fixture(scope="session")
def hybrid():
    return require(HYBRID)


@pytest.fixture(scope="session")
def device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
