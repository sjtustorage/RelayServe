# ---------------------------------------------------------------------------- #
#  shm_manager setup                                                             #
# ---------------------------------------------------------------------------- #
import io
import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install


ROOT_DIR = os.path.dirname(__file__)


def is_ninja_available() -> bool:
    try:
        subprocess.run(["ninja", "--version"], stdout=subprocess.PIPE)
    except FileNotFoundError:
        return False
    return True


class CustomInstall(install):
    def run(self):
        self.run_command("build_ext")
        super().run()


def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


class CMakeExtension(Extension):
    def __init__(self, name: str, cmake_lists_dir: str = ".", **kwa) -> None:
        super().__init__(name, sources=[], **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class cmake_build_ext(build_ext):
    did_config = {}

    def configure(self, ext: CMakeExtension) -> None:
        if ext.cmake_lists_dir in cmake_build_ext.did_config:
            return

        cmake_build_ext.did_config[ext.cmake_lists_dir] = True

        default_cfg = "Debug" if self.debug else "Release"
        cfg = os.getenv("CMAKE_BUILD_TYPE", default_cfg)

        outdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name))
        )

        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={outdir}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={outdir}",
            f"-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={self.build_temp}",
            f"-DSHM_MANAGER_PYTHON_EXECUTABLE={sys.executable}",
        ]

        # Be verbose for easier troubleshooting
        cmake_args += ["-DCMAKE_VERBOSE_MAKEFILE=ON"]

        if is_ninja_available():
            build_tool = ["-G", "Ninja"]
            cmake_args += [
                "-DCMAKE_JOB_POOL_COMPILE:STRING=compile",
                "-DCMAKE_JOB_POOLS:STRING=compile=8",
            ]
        else:
            build_tool = []

        os.makedirs(self.build_temp, exist_ok=True)
        subprocess.check_call(
            ["cmake", ext.cmake_lists_dir, *build_tool, *cmake_args],
            cwd=self.build_temp,
        )

    def build_extensions(self) -> None:
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError as e:
            raise RuntimeError("Cannot find CMake executable") from e

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        for ext in self.extensions:
            self.configure(ext)
            ext_target_name = remove_prefix(ext.name, "shm_manager.")
            build_args = [
                "--build",
                ".",
                "--target",
                ext_target_name,
            ]
            subprocess.check_call(["cmake", *build_args], cwd=self.build_temp)


def read_readme() -> str:
    p = os.path.join(ROOT_DIR, "README.md")
    if os.path.isfile(p):
        return io.open(p, "r", encoding="utf-8").read()
    return ""


setup(
    name="shm-manager",
    version="0.1.0",
    ext_modules=[
        CMakeExtension(name="shm_manager._C"),
    ],
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    packages=["shm_manager"],
    package_dir={"shm_manager": "."},
    package_data={
        "shm_manager": ["*.so"],
    },
    cmdclass={
        "build_ext": cmake_build_ext,
        "install": CustomInstall,
    },
)


