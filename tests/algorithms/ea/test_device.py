import pytest

torch = pytest.importorskip("torch")

from linkingtk.algorithms.ea._device import resolve_device  # noqa: E402
from linkingtk.exceptions import LinkingTKError  # noqa: E402


class TestResolveDevice:
    def test_cpu_returns_cpu_device(self) -> None:
        assert resolve_device("cpu").type == "cpu"

    def test_cuda_returns_cuda_device_when_available(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device available in this environment")
        assert resolve_device("cuda").type == "cuda"

    def test_cuda_raises_when_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(LinkingTKError, match="cuda"):
            resolve_device("cuda")

    def test_invalid_device_string_raises(self) -> None:
        with pytest.raises(LinkingTKError, match="Invalid device"):
            resolve_device("not-a-device")
