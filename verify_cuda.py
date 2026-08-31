"""Fail early with a useful message when the container cannot use its GPU."""

from __future__ import annotations

import sys

import torch


def main() -> int:
    print(
        "CUDA preflight: "
        f"PyTorch {torch.__version__}, CUDA runtime {torch.version.cuda}",
        flush=True,
    )

    if not torch.cuda.is_available():
        print(
            "CUDA preflight failed: PyTorch cannot access an NVIDIA GPU. "
            "Check the NVIDIA driver, Docker Desktop GPU support, and that "
            "the container was started with GPU access (--gpus all).",
            file=sys.stderr,
        )
        return 1

    try:
        device = torch.cuda.current_device()
        name = torch.cuda.get_device_name(device)
        capability = torch.cuda.get_device_capability(device)
        required_arch = f"sm_{capability[0]}{capability[1]}"
        compiled_arches = torch.cuda.get_arch_list()

        print(
            f"CUDA preflight: GPU {device}: {name}, capability {required_arch}; "
            f"compiled architectures: {', '.join(compiled_arches)}",
            flush=True,
        )

        # This catches an old wheel that can enumerate a newer GPU but has no
        # compatible kernel image.  RTX 5060 Ti is Blackwell / sm_120.
        tensor = torch.ones((16, 16), device=f"cuda:{device}", dtype=torch.float16)
        result = tensor @ tensor
        torch.cuda.synchronize(device)
        if not torch.isfinite(result).all().item():
            raise RuntimeError("CUDA smoke-test produced a non-finite result")
    except Exception as error:
        print(
            "CUDA preflight failed while executing a GPU kernel: "
            f"{type(error).__name__}: {error}\n"
            "RTX 50 series GPUs require a PyTorch build with Blackwell "
            "(sm_120) support; this image expects PyTorch 2.7.1 + CUDA 12.8.",
            file=sys.stderr,
        )
        return 1

    print("CUDA preflight: OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
