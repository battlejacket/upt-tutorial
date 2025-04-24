# Use official PyTorch GPU image (2.5.1, CUDA 12.4)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Install essential system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    # --- Core dev tools ---
    build-essential \
    git \
    curl \
    wget \
    \
    # --- Gmsh runtime dependencies (optional) ---
    libgl1 \
    libglu1-mesa \
    libx11-6 \
    libxrender1 \
    libxcursor1 \
    libxft2 \
    libxinerama1 \
    libice6 \
    \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Note: The Gmsh-related libraries can be removed if you're not using gmsh or pygmsh. 
# They are only required because the Gmsh Python SDK is linked against GUI-related X11 and OpenGL libraries even when used headlessly.


# Set workdir
WORKDIR /workspace

# Install torch-scatter and torch-cluster compatible with PyTorch 2.5.1 + CUDA 12.4
RUN pip install --no-cache-dir \
    torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu124.html \
    torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu124.html

# Install torch-geometric
RUN pip install --no-cache-dir torch-geometric

# Copy and install other Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# CMD ["bash"]
