
import json
import base64

# Read base64 content
with open("src.b64", "r") as f:
    b64_content = f.read().replace('\n', '')

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Edge Diffusion TTS Training on Colab GPU\n",
    "\n",
    "This notebook contains everything you need. No file uploads required.\n",
    "\n",
    "## Optimized Architecture\n",
    "- **MLA (Multi-Head Latent Attention)**\n",
    "- **RoPE (Rotary Positional Embeddings)**\n",
    "- **Fused FSQ** (MPS/GPU optimized)\n",
    "- **SwiGLU & RMSNorm**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Setup Environment\n",
    "Switch the runtime to GPU (**Runtime -> Change runtime type -> A100 or L4 GPU**).\n",
    "Then run the cells below in order."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install diffusers transformers accelerate matplotlib fastprogress"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
    "print(f\"Using device: {device}\")\n",
    "if device == \"cpu\":\n",
    "    print(\"WARNING: GPU not detected! Make sure you selected a GPU runtime.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Deploy Code\n",
    "This cell extracts the source code directly onto the Colab machine."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import base64\n",
    "import os\n",
    "\n",
    "# Reduced payload for embedding\n",
    "b64_data = \"" + b64_content + "\"\n",
    "\n",
    "print(\"Writing src.zip...\")\n",
    "with open(\"src.zip\", \"wb\") as f:\n",
    "    f.write(base64.b64decode(b64_data))\n",
    "\n",
    "print(\"Extracting...\")\n",
    "!unzip -o src.zip\n",
    "\n",
    "if not os.path.exists(\"edge_diffusion_tts\"):\n",
    "    print(\"ERROR: Extraction failed!\")\n",
    "else:\n",
    "    print(\"Code deployed successfully!\")\n",
    "    \n",
    "os.makedirs(\"data\", exist_ok=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "if not os.path.exists(\"data/LJSpeech-1.1\"):\n",
    "    print(\"Downloading LJSpeech dataset...\")\n",
    "    !curl -L https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -o data/ljspeech.tar.bz2\n",
    "    print(\"Extracting...\")\n",
    "    !tar -xjf data/ljspeech.tar.bz2 -C data/\n",
    "    print(\"Done!\")\n",
    "else:\n",
    "    print(\"Dataset already present.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Start Training\n",
    "The script will automatically detect the TPU (XLA) device."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!python train_improved.py"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Download Checkpoints\n",
    "After training, zip and download the results."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!zip -r run_output.zip run_edge_diffusion\n",
    "from google.colab import files\n",
    "files.download(\"run_output.zip\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open("colab_training.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)
