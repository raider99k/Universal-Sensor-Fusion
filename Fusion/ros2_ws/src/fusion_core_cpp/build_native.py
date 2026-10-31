from __future__ import annotations

import os
from pathlib import Path
import sys
from setuptools._distutils._msvccompiler import MSVCCompiler


def build_shared_library() -> Path:
    pkg_dir = Path(__file__).resolve().parent
    src_file = str(pkg_dir / "src" / "fusion_fast_kernels.cpp")
    inc_dir = str(pkg_dir / "include")
    out_dir = pkg_dir / "bin"
    out_dir.mkdir(parents=True, exist_ok=True)
    build_temp = pkg_dir / "build_temp"
    build_temp.mkdir(parents=True, exist_ok=True)

    compiler = MSVCCompiler()
    compiler.initialize()
    objs = compiler.compile(
        [src_file],
        output_dir=str(build_temp),
        include_dirs=[inc_dir],
        extra_postargs=["/O2", "/fp:fast", "/std:c++20", "/EHsc"],
    )
    dll_path = out_dir / "fusion_fast_kernels.dll"
    compiler.link_shared_object(
        objs,
        output_filename=str(dll_path),
        build_temp=str(build_temp),
    )
    print(f"Built native C++20 library: {dll_path} ({dll_path.stat().st_size} bytes)")
    return dll_path


if __name__ == "__main__":
    build_shared_library()
