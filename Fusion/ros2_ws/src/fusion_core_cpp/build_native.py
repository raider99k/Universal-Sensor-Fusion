from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys


def build_shared_library() -> Path:
    pkg_dir = Path(__file__).resolve().parent
    src_file = str(pkg_dir / "src" / "fusion_fast_kernels.cpp")
    inc_dir = str(pkg_dir / "include")
    out_dir = pkg_dir / "bin"
    out_dir.mkdir(parents=True, exist_ok=True)
    build_temp = pkg_dir / "build_temp"
    build_temp.mkdir(parents=True, exist_ok=True)

    if sys.platform == "win32":
        from setuptools._distutils._msvccompiler import MSVCCompiler

        compiler = MSVCCompiler()
        compiler.initialize()
        objs = compiler.compile(
            [src_file],
            output_dir=str(build_temp),
            include_dirs=[inc_dir],
            extra_postargs=["/O2", "/fp:fast", "/std:c++20", "/EHsc"],
        )
        lib_path = out_dir / "fusion_fast_kernels.dll"
        compiler.link_shared_object(
            objs,
            output_filename=str(lib_path),
            build_temp=str(build_temp),
        )
    else:
        cxx = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
        if not cxx:
            raise RuntimeError("No C++20 compiler (g++ or clang++) found on PATH")
        ext = "dylib" if sys.platform == "darwin" else "so"
        shared_flag = "-dynamiclib" if sys.platform == "darwin" else "-shared"
        lib_path = out_dir / f"libfusion_fast_kernels.{ext}"
        cmd = [
            cxx,
            "-O3",
            "-ffast-math",
            "-std=c++20",
            "-fPIC",
            shared_flag,
            f"-I{inc_dir}",
            src_file,
            "-o",
            str(lib_path),
        ]
        subprocess.run(cmd, check=True)

    print(f"Built native C++20 library: {lib_path} ({lib_path.stat().st_size} bytes)")
    return lib_path


if __name__ == "__main__":
    build_shared_library()
