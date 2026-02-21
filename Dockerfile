FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3-venv git curl unzip \
    && rm -rf /var/lib/apt/lists/*


RUN python3 -m pip install --upgrade pip


RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

RUN pip install -r req.txt

WORKDIR /workspace