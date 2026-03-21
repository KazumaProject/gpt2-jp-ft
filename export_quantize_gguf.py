import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from shutil import which
from typing import List


def run_cmd(cmd: List[str], title: str) -> None:
    print(f"[run] {title}")
    print("[cmd] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def resolve_convert_script(llama_cpp_dir: str) -> str:
    root = Path(llama_cpp_dir)
    candidates = [
        root / "convert-hf-to-gguf.py",
        root / "convert_hf_to_gguf.py",
        root / "convert.py",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError(
        "convert-hf-to-gguf.py / convert_hf_to_gguf.py (or convert.py) was not found in llama.cpp. "
        "Please clone/update llama.cpp and pass --llama_cpp_dir."
    )


def resolve_quantize_binary(llama_cpp_dir: str) -> str:
    root = Path(llama_cpp_dir)
    candidates = [
        root / "llama-quantize",
        root / "quantize",
        root / "build" / "bin" / "llama-quantize",
        root / "build" / "bin" / "quantize",
    ]
    for c in candidates:
        if c.exists() and os.access(c, os.X_OK):
            return str(c)

    in_path = which("llama-quantize")
    if in_path:
        return in_path

    raise FileNotFoundError(
        "llama-quantize binary was not found. Build llama.cpp first, for example:\n"
        "  cmake -S <llama.cpp> -B <llama.cpp>/build\n"
        "  cmake --build <llama.cpp>/build -j"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert HF model to GGUF and optionally quantize")
    parser.add_argument("--model_dir", required=True, help="HF model directory")
    parser.add_argument("--llama_cpp_dir", default="./llama.cpp", help="Path to llama.cpp repository")
    parser.add_argument("--gguf_out_dir", default=None, help="Directory to write GGUF files (default: <model_dir>-gguf)")
    parser.add_argument("--gguf_outtype", default="f16", help="GGUF outtype for convert_hf_to_gguf.py (e.g. f16, f32)")
    parser.add_argument(
        "--quantize_types",
        default="Q4_K_M",
        help="Comma-separated quantization types for llama-quantize (e.g. Q4_K_M,Q5_K_M). Empty to skip.",
    )
    parser.add_argument("--skip_quantize", action="store_true", help="Skip quantization")
    parser.add_argument(
        "--disable_config_patch",
        action="store_true",
        help="Disable auto patching config.json for older llama.cpp converters",
    )
    return parser.parse_args()


def patch_config_for_llamacpp(model_dir: Path) -> None:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if "n_ctx" in cfg:
        return

    n_ctx = cfg.get("n_positions") or cfg.get("max_position_embeddings")
    if n_ctx is None:
        return

    cfg["n_ctx"] = int(n_ctx)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"[info] patched config for llama.cpp: set n_ctx={cfg['n_ctx']} in {config_path}")


def main() -> None:
    args = parse_args()

    model_dir = Path(args.model_dir).expanduser()
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    if not args.disable_config_patch:
        patch_config_for_llamacpp(model_dir)

    llama_cpp_dir = str(Path(args.llama_cpp_dir).expanduser())
    convert_script = resolve_convert_script(llama_cpp_dir)

    gguf_out_dir = Path(args.gguf_out_dir).expanduser() if args.gguf_out_dir else Path(str(model_dir) + "-gguf")
    gguf_out_dir.mkdir(parents=True, exist_ok=True)

    base_gguf = gguf_out_dir / f"model-{args.gguf_outtype}.gguf"
    run_cmd(
        [
            sys.executable,
            convert_script,
            str(model_dir),
            "--outfile",
            str(base_gguf),
            "--outtype",
            args.gguf_outtype,
        ],
        "convert HF model to GGUF",
    )

    print(f"[done] base gguf: {base_gguf}")

    quantize_types = [x.strip() for x in args.quantize_types.split(",") if x.strip()]
    if args.skip_quantize or not quantize_types:
        print("[done] quantization skipped")
        return

    quant_bin = resolve_quantize_binary(llama_cpp_dir)
    for qtype in quantize_types:
        out_path = gguf_out_dir / f"model-{qtype}.gguf"
        run_cmd([quant_bin, str(base_gguf), str(out_path), qtype], f"quantize GGUF to {qtype}")
        print(f"[done] quantized gguf ({qtype}): {out_path}")


if __name__ == "__main__":
    main()
