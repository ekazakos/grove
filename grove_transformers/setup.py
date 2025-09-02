from setuptools import setup, find_packages

setup(
    name="grove-transformers",
    version="0.1.0",
    python_requires=">=3.12",
    packages=find_packages(),
    install_requires=[
        "ffmpeg-python==0.2.0",
        "transformers==4.46.3",
        "numpy==1.26.4",
        "pillow>=10.0",
        "pyyaml>=6.0",
        "opencv-python-headless==4.10.0.84",
        "bleach",
        "sentencepiece",
        "scikit-learn",
        "flash-attn==2.7.3",
    ],
    extras_require={
        "torch": [
            "torch==2.5.1",
            "torchvision==0.20.1",
            "torchaudio==2.5.1",
            "torchtext==0.18.0",
            "torchdata==0.9.0",
        ]
    },
)