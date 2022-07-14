from setuptools import setup, find_packages
from pathlib import Path

if __name__ == "__main__":
    with Path(Path(__file__).parent, "README.md").open(encoding="utf-8") as file:
        long_description = file.read()

    setup(
        name="clip_search",
        packages=find_packages(),
        include_package_data=True,
        version="0.0.1",
        license="MIT",
        description="Search local images using clip",
        long_description=long_description,
        long_description_content_type="text/markdown",
        entry_points={"console_scripts": ["clip_search = utils.cli:main"]},
        author="Jonas Konschak",
        author_email="jonaskonschak@protonmail.com",
        url="https://github.com/kanttouchthis/clip-search",
        data_files=[(".", ["README.md"])],
        keywords=["machine learning", "computer vision"],
        install_requires=[
            "Pillow",
            "clip @ git+https://github.com/openai/CLIP.git#egg=clip",
        ],
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3.9",
        ],
    )