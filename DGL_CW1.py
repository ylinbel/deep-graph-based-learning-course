{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "jWAqfqlcc5O4",
   "metadata": {
    "id": "jWAqfqlcc5O4"
   },
   "source": [
    "# DGL CW1\n",
    "Hi everyone, good luck on the first CW of the DGL course. Please do not hesitate to ask your questions on EdStem if you have any.\n",
    "\n",
    "## Questions\n",
    "1. Centrality-based Graph Classification (25 points)\n",
    "2. Implementation of GraphSAGE with Node Sampling (30 points)\n",
    "3. Attention-based aggregation in node classification (30 points)\n",
    "4. Propagation rule integrating edge features/embeddings (15 points)\n",
    "5. Bonus Questions (5 points)\n",
    "\n",
    "## Submission Instructions\n",
    "For submission, you only need to submit your Jupyter Notebook file named \"CID.ipynb\"\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3JNrkaurjfpM",
   "metadata": {
    "id": "3JNrkaurjfpM"
   },
   "source": [
    "# Required Libraries and Imports"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "id": "vZCmHCdxjn4I",
   "metadata": {
    "id": "vZCmHCdxjn4I"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: dgl in c:\\users\\linyi\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (2.0.0)\n",
      "Requirement already satisfied: numpy>=1.14.0 in c:\\users\\linyi\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from dgl) (1.26.1)\n",
      "Requirement already satisfied: scipy>=1.1.0 in c:\\users\\linyi\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from dgl) (1.11.3)\n",
      "Requirement already satisfied: networkx>=2.1 in c:\\users\\linyi\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from dgl) (3.2.1)\n",
      "Requirement already satisfied: requests>=2.19.0 in c:\\users\\linyi\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from dgl) (2.30.0)\n",
      "Requirement already satisfied: tqdm in c:\\users\\linyi\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from dgl) (4.65.0)\n",
      "Requirement already satisfied: psutil>=5.8.0 in c:\\users\\linyi\\appdata\\roaming\\python\\python311\\site-packages (from dgl) (5.9.6)\n",
      "Requirement already satisfied: torchdata>=0.5.0 in c:\\users\\linyi\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from dgl) (0.7.1)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in c:\\users\\linyi\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from requests>=2.19.0->dgl) (3.1.0)\n",
      "Requirement already satisfied: idna<4,>=2.5 in c:\\users\\linyi\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from requests>=2.19.0->dgl) (3.4)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in c:\\users\\linyi\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from requests>=2.19.0->dgl) (2.0.2)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\users\\linyi\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from requests>=2.19.0->dgl) (2023.5.7)\n",
      "Requirement already satisfied: torch>=2 in c:\\users\\linyi\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from torchdata>=0.5.0->dgl) (2.2.0)\n",
      "Requirement already satisfied: colorama in c:\\users\\linyi\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from tqdm->dgl) (0.4.6)\n",
      "Requirement already satisfied: filelock in c:\\users\\linyi\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from torch>=2->torchdata>=0.5.0->dgl) (3.13.1)\n",
      "Requirement already satisfied: typing-extensions>=4.8.0 in c:\\users\\linyi\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from torch>=2->torchdata>=0.5.0->dgl) (4.8.0)\n",
      "Requirement already satisfied: sympy in c:\\users\\linyi\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from torch>=2->torchdata>=0.5.0->dgl) (1.12)\n",
      "Requirement already satisfied: jinja2 in c:\\users\\linyi\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from torch>=2->torchdata>=0.5.0->dgl) (3.1.3)\n",
      "Requirement already satisfied: fsspec in c:\\users\\linyi\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from torch>=2->torchdata>=0.5.0->dgl) (2024.2.0)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in c:\\users\\linyi\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from jinja2->torch>=2->torchdata>=0.5.0->dgl) (2.1.3)\n",
      "Requirement already satisfied: mpmath>=0.19 in c:\\users\\linyi\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from sympy->torch>=2->torchdata>=0.5.0->dgl) (1.3.0)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n",
      "[notice] A new release of pip is available: 23.3.1 -> 24.0\n",
      "[notice] To update, run: python.exe -m pip install --upgrade pip\n"
     ]
    }
   ],
   "source": [
    "%pip install dgl"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "id": "DBqqR7fPjqKq",
   "metadata": {
    "id": "DBqqR7fPjqKq"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CUDA not available. Using CPU.\n"
     ]
    }
   ],
   "source": [
    "# Standard library imports\n",
    "import random\n",
    "import math\n",
    "import copy\n",
    "import time\n",
    "\n",
    "# Data handling, numerical processing, and visualization\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# Machine learning, neural network modules, and metrics\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "from torch import optim\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, euclidean_distances\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.decomposition import PCA\n",
    "\n",
    "# Graph processing\n",
    "import networkx as nx\n",
    "\n",
    "# DGL and datasets\n",
    "from dgl.data import PPIDataset, CoraGraphDataset\n",
    "\n",
    "# Set a fixed random seed for reproducibility across multiple libraries\n",
    "random_seed = 42\n",
    "random.seed(random_seed)\n",
    "np.random.seed(random_seed)\n",
    "torch.manual_seed(random_seed)\n",
    "\n",
    "# Check for CUDA (GPU support) and set device accordingly\n",
    "if torch.cuda.is_available():\n",
    "    device = torch.device(\"cuda\")\n",
    "    print(\"CUDA is available. Using GPU.\")\n",
    "    torch.cuda.manual_seed(random_seed)\n",
    "    torch.cuda.manual_seed_all(random_seed)  # For multi-GPU setups\n",
    "    # Additional settings for ensuring reproducibility on CUDA\n",
    "    torch.backends.cudnn.deterministic = True\n",
    "    torch.backends.cudnn.benchmark = False\n",
    "else:\n",
    "    device = torch.device(\"cpu\")\n",
    "    print(\"CUDA not available. Using CPU.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "id": "077c7d5f",
   "metadata": {},
   "outputs": [],
   "source": [
    "import networkx as nx\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "LQ7SGGtic5O6",
   "metadata": {
    "id": "LQ7SGGtic5O6"
   },
   "source": [
    "# 1) Centrality-based Graph Classification (25 points)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "DeiwfbLOc5O7",
   "metadata": {
    "id": "DeiwfbLOc5O7"
   },
   "source": [
    "## Part 1: Implementing and Analyzing PageRank Centrality\n",
    "\n",
    "### Objective\n",
    "The goal of this task is to enhance your comprehension of the PageRank centrality algorithm through hands-on implementation. You are tasked with coding the algorithm from the ground up and applying it to two specific graphs, named $G1$ and $G2$. Following the application, you will analyze and compare the PageRank distributions between these two graphs.\n",
    "\n",
    "### Total Points: 10\n",
    "\n",
    "---\n",
    "\n",
    "### Instructions\n",
    "\n",
    "#### 1. **Implement the PageRank Algorithm**\n",
    "\n",
    "- **Initialization**: Assign an equal PageRank to every node initially. For a graph containing $N$ nodes, each node's starting PageRank is $1/N$.\n",
    "\n",
    "- **Calculation**: Update each node's PageRank by employing the formula:\n",
    "  $$PR(p_i) = \\frac{1-d}{N} + d \\sum_{p_j \\in M(p_i)} \\frac{PR(p_j)}{L(p_j)}$$\n",
    "  where:\n",
    "  - $PR(p_i)$ denotes the PageRank of page $p_i$,\n",
    "  - $d$ represents the damping factor, usually set at 0.85,\n",
    "  - $N$ is the total count of nodes within the graph,\n",
    "  - $M(p_i)$ encompasses the set of nodes linking to $p_i$,\n",
    "  - $L(p_j)$ signifies the number of outbound links from node $p_j$.\n",
    "\n",
    "- **Convergence**: Continuously recalculate until the PageRank values reach a stable condition, evidenced by negligible changes between successive iterations.\n",
    "\n",
    "#### 2. **Pre-defined Graphs G1 and G2**\n",
    "\n",
    "- You have access to two graphs, $G1$ and $G2$, within your environment. These will serve as your experimental subjects for the PageRank algorithm application.\n",
    "\n",
    "#### 3. **Calculate PageRank Values**\n",
    "\n",
    "- Implement your PageRank algorithm on both $G1$ and $G2$ to determine the PageRank values for their respective nodes.\n",
    "\n",
    "#### 4. **Visualize the Results**\n",
    "\n",
    "- Create plots to showcase the PageRank value distributions for $G1$ and $G2$. Apply a transparent blue overlay for $G1$ and a transparent red for $G2$, facilitating an effortless comparison.\n",
    "\n",
    "#### 5. **Analysis**\n",
    "\n",
    "- **Reflect** upon the PageRank distribution variances between $G1$ and $G2$, pondering over:\n",
    "  - The impact of $G1$ and $G2$ structures on their PageRank distributions.\n",
    "  - Whether the PageRank values are more clustered or dispersed in one graph relative to the other.\n",
    "  - The presence of any nodes that emerge as notably more central within their graphs.\n",
    "\n",
    "### Comment Section\n",
    "\n",
    "- **Discuss** the insights and observations derived from juxtaposing the PageRank distributions of $G1$ and $G2$. Highlight any discerned patterns or anomalies, and deliberate on how each graph's inherent structure could elucidate these findings.\n",
    "\n",
    "---\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "id": "2f0c276d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAkAAAAHHCAYAAABXx+fLAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/SrBM8AAAACXBIWXMAAA9hAAAPYQGoP6dpAABcC0lEQVR4nO3dd1gU1/s28HtpSwcLNRZEUKIiKirBiopiwxpbUEGN3URjNMb8EkVNxBJb7Ca2WGKCNdHYlcSCXVSiohAVjRQrCCr1vH/4Ml/XXfpS5/5c1146Z86cec7M7uzDzJlZhRBCgIiIiEhGdEo6ACIiIqLixgSIiIiIZIcJEBEREckOEyAiIiKSHSZAREREJDtMgIiIiEh2mAARERGR7DABIiIiItlhAkRERESywwSI8kyhUCAwMLCkwyi0TZs2wcXFBfr6+rC0tCzpcMqswMBAKBQKPH78uKRDwd27d6FQKLBhw4aSDiVPAgIC4ODgUCzrcnBwQEBAgDS9YcMGKBQKXLhwoVjW7+XlBS8vr2JZV0EcOHAADRo0gKGhIRQKBZ4/f17SIZVJZe0zCDABypeoqCiMHDkSjo6OMDQ0hLm5OZo3b44lS5bg1atXJR0e5cHNmzcREBCAmjVr4scff8SaNWuyrZv1BZ/1MjY2Rp06dfD1118jMTGxGKPWLCQkRCU+XV1dWFtb48MPP8SNGzdKOrxsdevWDcbGxnjx4kW2dfz8/GBgYIAnT54UY2QFo+l9Uq1aNfj6+mL9+vVISUnRynquX7+OwMBA3L17VyvtaVNpji0nT548Qd++fWFkZITly5dj06ZNMDExKfL13rlzB+PGjUOtWrVgbGwsHVvGjh2Lq1evqtSNiYnBl19+iTZt2sDMzAwKhQIhISFFHmNRi4+Px5dffglXV1eYmprC0NAQTk5OGDJkCE6ePKlS9/z58xg3bhzq1q0LExMTVKtWDX379sWtW7cKFYNeoZaWkX379qFPnz5QKpUYPHgw6tWrh9TUVJw8eRKTJ0/GP//8k+OXaXnw6tUr6OmV7bdMSEgIMjMzsWTJEjg5OeVpmZUrV8LU1BRJSUk4dOgQvvvuOxw7dgynTp2CQqEo4ohz9+mnn6JJkyZIS0vD1atXsWrVKoSEhCA8PBy2trYlHZ4aPz8//PHHH9i1axcGDx6sNv/ly5fYs2cPOnbsiEqVKpVAhAWT9T5JSUnBf//9h4MHD2Lo0KFYvHgx9u7di6pVq0p1f/zxR2RmZuar/evXr2PGjBnw8vLK19mjiIgI6OgU7d+6OcV26NChIl13YZw/fx4vXrzArFmz4O3tXSzr3Lt3L/r16wc9PT34+fnBzc0NOjo6uHnzJnbu3ImVK1fizp07qF69OoA3+2/u3LlwdnaGq6srQkNDiyXOonTu3Dl06dIFL168QP/+/TFq1CgolUrcuXMHu3fvxoYNG/DXX3+hVatWAIC5c+fi1KlT6NOnD+rXr4/Y2FgsW7YMjRo1wpkzZ1CvXr2CBSIoV//++68wNTUVLi4u4uHDh2rzb9++LRYvXlwCkRW9jIwM8erVq5IOQ2tmzJghAIhHjx7lWnf69Oka6/bq1UsAEKdPny6qMPPk+PHjAoAIDg5WKV+5cqUAIObOnVuk689u++Tm5cuXwszMTPj4+Gicv3XrVgFAbNu2Lc9t3rlzRwAQ69evz1cs2pDTdti8ebPQ0dERHh4ehV5PcHCwACCOHz+ea93MzEzx8uVLjfPWr18vAIjz588XOqaCxFaabNy4UevbIikpKdt5kZGRwsTERLz//vsav0vS0tLEkiVLRHR0tFSWmJgonjx5IoQo3ds5r5/Bp0+fCjs7O2Fraytu3LihNj8zM1Ns3bpVnDt3Tio7deqUSElJUal369YtoVQqhZ+fX4Fj5iWwPJg3bx6SkpKwdu1a2NnZqc13cnLC+PHjpen09HTMmjULNWvWhFKphIODA7766iu1U+EODg7o2rUrQkJC0LhxYxgZGcHV1VU6vblz5064urrC0NAQ7u7uuHz5ssryAQEBMDU1xb///gsfHx+YmJjA3t4eM2fOhBBCpe7333+PZs2aoVKlSjAyMoK7uzu2b9+u1heFQoFx48Zhy5YtqFu3LpRKJQ4cOCDNe3sM0IsXLzBhwgQ4ODhAqVTC2toa7du3x6VLl1TaDA4Ohru7O4yMjFC5cmUMHDgQ//33n8a+/Pfff+jRowdMTU1hZWWFSZMmISMjI5s9o2rFihVSzPb29hg7dqzK9XwHBwdMnz4dAGBlZVXgMU1t27YF8OY0dmpqKqZNmwZ3d3dYWFjAxMQELVu2xPHjx9WWe/LkCQYNGgRzc3NYWlrC398fV65c0Xjd/ObNm/jwww9RsWJFGBoaonHjxvj999/zFF/Lli0BvLlk+7b8vgd2796NevXqQalUom7dutL7ICf37t2Dk5MT6tWrh7i4OI11jIyM0KtXLxw9ehTx8fFq87du3QozMzN069YNT58+xaRJk6TT5Obm5ujUqROuXLmSayzZjT3RNP4mMzMTixcvRt26dWFoaAgbGxuMHDkSz549y3U9OfHz88PHH3+Ms2fP4vDhwznGsG3bNri7u8PMzAzm5uZwdXXFkiVLALwZt9OnTx8AQJs2baTLbVnHiqxjycGDB6VjyerVq6V5b48ByvLy5UuMHDkSlSpVgrm5OQYPHqzW3+w+I2+3mVtsmvZDfHw8hg0bBhsbGxgaGsLNzQ0bN25UqZM1puT777/HmjVrpONpkyZNcP78eZW6sbGxGDJkCKpUqQKlUgk7Ozt07949x0tyXl5e8Pf3BwA0adIECoVCZTvl57gVFRWFzp07w8zMDH5+ftmuc968eUhOTsb69es1fpfo6enh008/VTlbaGZmhooVK2bbZm5OnDiBPn36oFq1alAqlahatSo+++wztWEb+TkGP3/+HAEBAbCwsJCOZXkdO7Vq1SrExMRg8eLFcHFxUZuvUCgwYMAANGnSRCpr1qwZDAwMVOo5Ozujbt26hbrczwQoD/744w84OjqiWbNmear/8ccfY9q0aWjUqBEWLVqE1q1bIygoCP3791erGxkZiY8++gi+vr4ICgrCs2fP4Ovriy1btuCzzz7DwIEDMWPGDERFRaFv375qp80zMjLQsWNH2NjYYN68eXB3d8f06dOlL/osS5YsQcOGDTFz5kzMnj0benp66NOnD/bt26cW07Fjx/DZZ5+hX79+WLJkSban20eNGoWVK1eid+/eWLFiBSZNmgQjIyOVN+SGDRvQt29f6OrqIigoCMOHD8fOnTvRokULtQ9MRkYGfHx8UKlSJXz//fdo3bo1FixYkKdLi4GBgRg7dizs7e2xYMEC9O7dG6tXr0aHDh2QlpYGAFi8eDF69uwJ4M3lik2bNqFXr165tv2urMSiUqVKSExMxE8//QQvLy/MnTsXgYGBePToEXx8fBAWFiYtk5mZCV9fX/zyyy/w9/fHd999h5iYGOkA/LZ//vkHH3zwAW7cuIEvv/wSCxYsgImJCXr06IFdu3blGl/WQb9ChQoq5fl5D5w8eRJjxoxB//79MW/ePLx+/Rq9e/fOcUxOVFQUWrVqBTMzM4SEhMDGxibbun5+fkhPT8dvv/2mUv706VMcPHgQPXv2hJGREf7991/s3r0bXbt2xcKFCzF58mRcu3YNrVu3xsOHD3PdFnk1cuRITJ48WRrTN2TIEGzZsgU+Pj7S+6egBg0aBCDnS0GHDx/GgAEDUKFCBcydOxdz5syBl5cXTp06BQBo1aoVPv30UwDAV199hU2bNmHTpk14//33pTYiIiIwYMAAtG/fHkuWLEGDBg1yjGvcuHG4ceMGAgMDMXjwYGzZsgU9evRQ++MpN3mJ7W2vXr2Cl5cXNm3aBD8/P8yfPx8WFhYICAiQEr63bd26FfPnz8fIkSPx7bff4u7du+jVq5fKfunduzd27dqFIUOGYMWKFfj000/x4sULREdHZxv3//3f/2HEiBEAgJkzZ2LTpk0YOXIkgPwdt9LT0+Hj4wNra2t8//336N27d7br3Lt3L5ycnODh4ZFtHW0LDg7Gy5cvMXr0aCxduhQ+Pj5YunSpxsvPeTkGCyHQvXt3bNq0CQMHDsS3336LBw8eaDyWafLHH39IfwQVhhACcXFxqFy5cqEaoRwkJCQIAKJ79+55qh8WFiYAiI8//lilfNKkSQKAOHbsmFRWvXp1tUspBw8eFACEkZGRuHfvnlS+evVqtVOf/v7+AoD45JNPpLLMzEzRpUsXYWBgoHJK/t3T4ampqaJevXqibdu2KuUAhI6Ojvjnn3/U+gZATJ8+XZq2sLAQY8eOzXZbpKamCmtra1GvXj2Vy2h79+4VAMS0adPU+jJz5kyVNho2bCjc3d2zXYcQQsTHxwsDAwPRoUMHkZGRIZUvW7ZMABDr1q2TyvJz2SarbkREhHj06JG4c+eOWL16tVAqlcLGxkYkJyeL9PR0tVOzz549EzY2NmLo0KFS2Y4dOwQAlUulGRkZom3btmqnjdu1aydcXV3F69evpbLMzEzRrFkz4ezsLJVlXQJbt26dePTokXj48KE4cOCAcHJyEgqFQuUUshD5ew8YGBiIyMhIqezKlSsCgFi6dKnGbXnjxg1hb28vmjRpIp4+fZrrtk1PTxd2dnbC09NTpXzVqlUCgDh48KAQQojXr1+r7FMh3pxqVyqVKu8VTaffW7duLVq3bq22bn9/f1G9enVp+sSJEwKA2LJli0q9AwcOaCx/V27vqWfPngkAomfPntnGMH78eGFubi7S09OzXU9Olz+yjiUHDhzQOM/f31+azroE5u7uLlJTU6XyefPmCQBiz549Utm7n/ns2swptnf3w+LFiwUAsXnzZqksNTVVeHp6ClNTU5GYmCiE+N8+rVSpksp7as+ePQKA+OOPP4QQ/9u+8+fPV1t3bjRdDizIcevLL7/MdV1Z3yU9evRQm/fs2TPx6NEj6ZXd5cuCXALT1FZQUJBQKBQq3zF5PQbv3r1bABDz5s2TytLT00XLli3zdAmsQoUKokGDBmrliYmJKtsgp0uJQgixadMmAUCsXbs2x3o54RmgXGTd7WNmZpan+n/++ScAYOLEiSrln3/+OQCo/bVdp04deHp6StNZfxm0bdsW1apVUyv/999/1dY5btw46f9Zly9SU1Nx5MgRqdzIyEj6/7Nnz5CQkICWLVuqXa4CgNatW6NOnTq59BSwtLTE2bNns/1L/MKFC4iPj8eYMWNgaGgolXfp0gUuLi4azzyMGjVKZbply5Ya+/y2I0eOIDU1FRMmTFAZ7Dl8+HCYm5trXE9+1K5dG1ZWVqhRowZGjhwJJycn7Nu3D8bGxtDV1ZVOzWZmZuLp06dIT09H48aNVbbtgQMHoK+vj+HDh0tlOjo6GDt2rMq6nj59imPHjqFv37548eIFHj9+jMePH+PJkyfw8fHB7du31U7DDx06FFZWVrC3t0fHjh2RkJCATZs2qZxCBvL3HvD29kbNmjWl6fr168Pc3FzjvggPD0fr1q3h4OCAI0eOqJ150kRXVxf9+/dHaGioymWKrVu3wsbGBu3atQMAKJVKaZ9mZGTgyZMnMDU1Re3atTXGXRDBwcGwsLBA+/btpe39+PFjuLu7w9TUVOPlzPwwNTUFgBzverO0tERycrLKZbL8qlGjBnx8fPJcf8SIEdDX15emR48eDT09PekYVlT+/PNP2NraYsCAAVKZvr4+Pv30UyQlJeGvv/5Sqd+vXz+V91TWJd6s96KRkREMDAwQEhJS6EuWQMGOW6NHj8613azvkqz3w9u8vLxgZWUlvZYvX16IHqh6+3OfnJyMx48fo1mzZhBCqA2rAHI/Bv/555/Q09NT6bOuri4++eSTPMWTmJiocRsMGjRIZRtMmTIl2zZu3ryJsWPHwtPTM89nnjRhApQLc3NzADkfvN5279496OjoqN1hZGtrC0tLS9y7d0+l/O0kBwAsLCwAQOUa8Nvl737AdXR04OjoqFJWq1YtAFD5Ytm7dy8++OADGBoaomLFirCyssLKlSuRkJCg1ocaNWrk1k0Ab65nh4eHo2rVqmjatCkCAwNVPihZfa1du7basi4uLmrbwtDQEFZWViplFSpUyPWglt16DAwM4OjoqLae/NqxYwcOHz6MkJAQREZGIjw8HO7u7tL8jRs3on79+jA0NESlSpVgZWWFffv2qWzbe/fuwc7ODsbGxiptv/s+iYyMhBAC33zzjcrBwMrKSrqs+e64mWnTpuHw4cPSXVUJCQka7/rJz3vg3fclkP2+8PX1hZmZGQ4ePCh9XvIia6zE1q1bAQAPHjzAiRMn0L9/f+jq6gJ4k1QuWrQIzs7OUCqVqFy5MqysrHD16lWNcRfE7du3kZCQAGtra7VtnpSUpHGcUn4kJSUByPmPqDFjxqBWrVro1KkTqlSpgqFDh+ZpzNXb8vq5zeLs7KwybWpqCjs7uyK/lf3evXtwdnZWe49mXTLL7RiZlQxlvReVSiXmzp2L/fv3w8bGBq1atcK8efMQGxtb4PiAvB+39PT0UKVKlVzbzdr/We+Ht61evRqHDx/G5s2bCxJyjqKjoxEQEICKFStK43pat24NAGqfobwcg7OOZe8mMZq2lyZmZmYat8HMmTNx+PDhXP8IiI2NRZcuXWBhYYHt27dLx4qCKNv3NBcDc3Nz2NvbIzw8PF/L5fX26Ox2XnblIp/X54E3g+C6deuGVq1aYcWKFbCzs4O+vj7Wr18vffm87e2/GHLSt29ftGzZErt27cKhQ4cwf/58zJ07Fzt37kSnTp3yHWdh3shFqVWrVtleZ968eTMCAgLQo0cPTJ48GdbW1tK4gXcHIedF1hivSZMmZfvX/LtJk6urq3QLb48ePfDy5UsMHz4cLVq0kBLp/L4H8vP+6927NzZu3IgtW7ZIYyjywt3dHS4uLvjll1/w1Vdf4ZdffoEQQmUQ6ezZs/HNN99g6NChmDVrFipWrAgdHR1MmDAh19vIFQqFxnjfHdCZmZkJa2trbNmyRWM7734h5FfWsSOnxy5YW1sjLCwMBw8exP79+7F//36sX78egwcPVhscnJ28fm61Ia83JmhDXt6LEyZMgK+vL3bv3o2DBw/im2++QVBQEI4dO4aGDRsWaXxvn6XMiYWFBezs7DR+l2Sd4dd28pmRkYH27dvj6dOnmDJlClxcXGBiYoL//vsPAQEBap+h4jgGu7i44MqVK0hLS1M5A1m/fv1cl01ISECnTp3w/PlznDhxAvb29oWKhWeA8qBr166IiorK0/MXqlevjszMTNy+fVulPC4uDs+fP5ee7aAtmZmZapclsh4OlTV4eceOHTA0NJSeS9KpUyetPfPCzs4OY8aMwe7du3Hnzh1UqlQJ3333HQCoPMfiXREREVrbFtmtJzU1VeV5GkVh+/btcHR0xM6dOzFo0CD4+PjA29sbr1+/VosxJiYGL1++VCmPjIxUmc46m6evrw9vb2+Nr9wux86ZMwevX7+W9gNQtO+B+fPnY9iwYRgzZozGZConfn5+CA8Px9WrV7F161Y4OzurXLrbvn072rRpg7Vr16J///7o0KEDvL2983THSYUKFTTWe/cv+Jo1a+LJkydo3ry5xu3t5uaWrz69a9OmTQCQ6+UpAwMD+Pr6YsWKFdJDV3/++WfpPaLtZ069e4xKSkpCTEyMyk0PmrZhamoqYmJiVMryE1v16tVx+/ZttS/fmzdvSvMLombNmvj8889x6NAhhIeHIzU1FQsWLMh3O0V53OrSpQsiIyNx7ty5AreRH9euXcOtW7ewYMECTJkyBd27d4e3t3ehEoesY9m7Z3E0bS9NunbtilevXuXpho63vX79Gr6+vrh16xb27t2bp2EauWEClAdffPEFTExM8PHHH2u8tTcqKkq6e6Fz584A3txx9LaFCxcCePMB0LZly5ZJ/xdCYNmyZdDX15fGUejq6kKhUKj81Xb37l3s3r27wOvMyMhQO31qbW0Ne3t76Xb/xo0bw9raGqtWrVJ5BMD+/ftx48YNrW0Lb29vGBgY4IcfflD5q3Dt2rVISEgokm2eJesvprfXe/bsWbVkOetuoh9//FEqy8zMVLvWb21tDS8vL6xevVrtSwYAHj16lGtMNWvWRO/evbFhwwbpMkBRvAeyKBQKrFmzBh9++CH8/f3zfLs+8L/LYNOmTUNYWJjaLcS6urpqZ3GCg4PVxkFpUrNmTdy8eVNlm125ckW6sypL3759kZGRgVmzZqm1kZ6eXqifRti6dSt++ukneHp6Sp9HTd69u05HR0f6izjrs5P1hGJt/VTDmjVrVO6kWrlyJdLT01XO3tasWRN///232nLvngHKT2ydO3dGbGwsfv31V6ksPT0dS5cuhampqXR5Jq9evnyp9gdHzZo1YWZmVqCncBflceuLL76AsbExhg4dqvG7pCBn+HOi6fgkhNB4t11ede7cGenp6Vi5cqVUlpGRgaVLl+Zp+dGjR8PGxgafffaZxic5Z3fWtl+/fggNDUVwcLDKuNnC4CWwPKhZsya2bt2Kfv364f3331d5EvTp06cRHBwsPT/Czc0N/v7+WLNmDZ4/f47WrVvj3Llz2LhxI3r06IE2bdpoNTZDQ0McOHAA/v7+8PDwwP79+7Fv3z589dVX0qn7Ll26YOHChejYsSM++ugjxMfHY/ny5XByclJ77HpevXjxAlWqVMGHH34INzc3mJqa4siRIzh//rz0V5e+vj7mzp2LIUOGoHXr1hgwYADi4uKkW+s/++wzrWwDKysrTJ06FTNmzEDHjh3RrVs3REREYMWKFWjSpAkGDhyolfVo0rVrV+zcuRM9e/ZEly5dcOfOHaxatQp16tRR+QupR48eaNq0KT7//HNERkbCxcUFv//+O54+fQpA9S/o5cuXo0WLFnB1dcXw4cPh6OiIuLg4hIaG4sGDB3l6Bs7kyZPx22+/YfHixZgzZ06RvAfepqOjg82bN6NHjx7o27cv/vzzT+l5STmpUaMGmjVrhj179gCAWgLUtWtXzJw5E0OGDEGzZs1w7do1bNmyRW3cmyZDhw7FwoUL4ePjg2HDhiE+Ph6rVq1C3bp1VX7KpHXr1hg5ciSCgoIQFhaGDh06QF9fH7dv30ZwcDCWLFmCDz/8MNf1bd++HaampkhNTZWeBH3q1Cm4ubkhODg4x2U//vhjPH36FG3btkWVKlVw7949LF26FA0aNJDGxjRo0AC6urqYO3cuEhISoFQq0bZtW1hbW+camyapqalo164d+vbtK31eWrRogW7duqnENWrUKPTu3Rvt27fHlStXcPDgQbVLwvmJbcSIEVi9ejUCAgJw8eJFODg4YPv27Th16hQWL16c5xtOsty6dUvqR506daCnp4ddu3YhLi5O46NHclOUxy1nZ2ds3boVAwYMQO3ataUnQQshcOfOHWzduhU6OjpqY4q+/fZbAG8ekQG8OauY9XMRX3/9dbbrc3FxQc2aNTFp0iT8999/MDc3x44dOwo1WNzX1xfNmzfHl19+ibt376JOnTrYuXNnnsfkVaxYEbt27YKvry/c3NzQv39/NGnSBPr6+rh//770WXl77Nfnn3+O33//Hb6+vnj69KnaWKkCH+MLfP+YDN26dUsMHz5cODg4CAMDA2FmZiaaN28uli5dqnLLclpampgxY4aoUaOG0NfXF1WrVhVTp05VqSPEm1tJu3TporYeAGq3l2fdEvr2rZ7+/v7CxMREREVFiQ4dOghjY2NhY2Mjpk+frnbr8Nq1a4Wzs7NQKpXCxcVFrF+/Xrp9N7d1vz0v65bYlJQUMXnyZOHm5ibMzMyEiYmJcHNzEytWrFBb7tdffxUNGzYUSqVSVKxYUfj5+YkHDx6o1Mnqy7s0xZidZcuWCRcXF6Gvry9sbGzE6NGjxbNnzzS2V5gnQb8tMzNTzJ49W1SvXl0olUrRsGFDsXfvXrXbnIUQ4tGjR+Kjjz4SZmZmwsLCQgQEBIhTp05pfOpxVFSUGDx4sLC1tRX6+vrivffeE127dhXbt2+X6mT3JOgsXl5ewtzcXDx//lwIUfj3wLu3PmvaPi9fvhStW7cWpqam4syZM9lut7ctX75cABBNmzZVm/f69Wvx+eefCzs7O2FkZCSaN28uQkND1W6tzu4ptJs3bxaOjo7CwMBANGjQQBw8eFDjvhFCiDVr1gh3d3dhZGQkzMzMhKurq/jiiy80PrH3bVnbIetlaGgoqlSpIrp27SrWrVun9rkXQv02+O3bt4sOHToIa2trYWBgIKpVqyZGjhwpYmJiVJb78ccfhaOjo9DV1VW5HTq7Y0nWPE23wf/1119ixIgRokKFCsLU1FT4+flJTxzOkpGRIaZMmSIqV64sjI2NhY+Pj4iMjFRrM6fYND2OIC4uTgwZMkRUrlxZGBgYCFdXV7V9p+mYl+XtY9Hjx4/F2LFjhYuLizAxMREWFhbCw8ND/Pbbbxq3x9tyeip2YY5buYmMjBSjR48WTk5OwtDQUBgZGQkXFxcxatQoERYWprG/2b1yc/36deHt7S1MTU1F5cqVxfDhw6XHWry9zfNzDH7y5IkYNGiQMDc3FxYWFmLQoEHi8uXLeboNPktMTIyYPHmyqFOnjjAyMhJKpVI4OjqKwYMHi7///lulbuvWrQu1DbKjEELL59yo2AQEBGD79u0aR9RT2bB792707NkTJ0+eRPPmzUs6HCIi2eAYIKJi8u6j57Oum5ubm6NRo0YlFBURkTxxDBBRMfnkk0/w6tUreHp6IiUlBTt37sTp06cxe/bsYr2FmYiImAARFZu2bdtiwYIF2Lt3L16/fg0nJycsXbpU5UneRERUPDgGiIiIiGSHY4CIiIhIdpgAERERkexwDJAGmZmZePjwIczMzLT++HkiIiIqGkIIvHjxAvb29rn+RhsTIA0ePnyo9mvsREREVDbcv39f7Yna72ICpEHWo9jv378Pc3PzEo6GiIiI8iIxMRFVq1bN00+qMAHSIOuyl7m5ORMgIiKiMiYvw1c4CJqIiIhkhwkQERERyQ4TICIiIpIdjgEiIiIq5TIyMpCWllbSYZQ4fX196OrqaqUtJkBERESllBACsbGxeP78eUmHUmpYWlrC1ta20M/pYwJERERUSmUlP9bW1jA2Npb1w3mFEHj58iXi4+MBAHZ2doVqjwkQERFRKZSRkSElP5UqVSrpcEoFIyMjAEB8fDysra0LdTmMg6CJiIhKoawxP8bGxiUcSemStT0KOyaKCRAREVEpJufLXppoa3swASIiIiLZKdEEKCgoCE2aNIGZmRmsra3Ro0cPREREqNR5/fo1xo4di0qVKsHU1BS9e/dGXFxcju0KITBt2jTY2dnByMgI3t7euH37dlF2hYiIiMqQEh0E/ddff2Hs2LFo0qQJ0tPT8dVXX6FDhw64fv06TExMAACfffYZ9u3bh+DgYFhYWGDcuHHo1asXTp06lW278+bNww8//ICNGzeiRo0a+Oabb+Dj44Pr16/D0NCwuLpHRERUJAIDy8b6YmNjERQUhH379uHBgwewsLCAk5MTBg4cCH9/fxgbG2PNmjXYunUrLl26hBcvXuDZs2ewtLTUZvgalWgCdODAAZXpDRs2wNraGhcvXkSrVq2QkJCAtWvXYuvWrWjbti0AYP369Xj//fdx5swZfPDBB2ptCiGwePFifP311+jevTsA4Oeff4aNjQ12796N/v37F33HiIiIZO7ff/9F8+bNYWlpidmzZ8PV1RVKpRLXrl3DmjVr8N5776Fbt254+fIlOnbsiI4dO2Lq1KnFFl+pug0+ISEBAFCxYkUAwMWLF5GWlgZvb2+pjouLC6pVq4bQ0FCNCdCdO3cQGxursoyFhQU8PDwQGhrKBIiIiKgYjBkzBnp6erhw4YJ0VQcAHB0d0b17dwghAAATJkwAAISEhBRrfKUmAcrMzMSECRPQvHlz1KtXD8CbU2cGBgZqp8JsbGwQGxursZ2schsbmzwvk5KSgpSUFGk6MTGxoN0gIiKSvSdPnuDQoUOYPXu2SvLztpK+u63UJEBjx45FeHg4Tp48WezrDgoKwowZM4p9vdpWlNeEi/t6MxERlV2RkZEQQqB27doq5ZUrV8br168BvPnenzt3bkmEB6CU3AY/btw47N27F8ePH0eVKlWkcltbW6Smpqr9BkpcXBxsbW01tpVV/u6dYjktM3XqVCQkJEiv+/fvF6I3REREpMm5c+cQFhaGunXrqlx5KQklmgAJITBu3Djs2rULx44dQ40aNVTmu7u7Q19fH0ePHpXKIiIiEB0dDU9PT41t1qhRA7a2tirLJCYm4uzZs9kuo1QqYW5urvIiIiKignFycoJCoVB7tI2joyOcnJykn7QoSSWaAI0dOxabN2/G1q1bYWZmhtjYWMTGxuLVq1cA3gxeHjZsGCZOnIjjx4/j4sWLGDJkCDw9PVUGQLu4uGDXrl0A3lxTnDBhAr799lv8/vvvuHbtGgYPHgx7e3v06NGjJLpJREQkK5UqVUL79u2xbNkyJCcnl3Q4GpXoGKCVK1cCALy8vFTK169fj4CAAADAokWLoKOjg969eyMlJQU+Pj5YsWKFSv2IiAjpDjIA+OKLL5CcnIwRI0bg+fPnaNGiBQ4cOMBnABERERWTFStWoHnz5mjcuDECAwNRv3596Ojo4Pz587h58ybc3d0BQDr5ERkZCQC4du0azMzMUK1aNemu8KKgEFn3oZEkMTERFhYWSEhIKFOXwzgImoio/Hj9+jXu3LmDGjVqqP0BX1YehBgTE4PZs2dLD0JUKpWoU6cO+vTpgzFjxsDY2BiBgYEab0R6+2TI23LaLvn5/mYCpAEToOJtm4iI1OX0RS9n2kqASsVdYERERETFiQkQERERyQ4TICIiIpIdJkBEREQkO0yAiIiISHaYABEREZHsMAEiIiIi2WECRERERLLDBIiIiIhkhwkQERERyU6J/hgqERERFUAZ+TGw2NhYBAUFSb8FZmFhAScnJwwcOBD+/v54/fo1pk+fjkOHDiE6OhpWVlbo0aMHZs2aBQsLC+324R1MgIiIiEjr/v33XzRv3hyWlpaYPXs2XF1doVQqce3aNaxZswbvvfceHB0d8fDhQ3z//feoU6cO7t27h1GjRuHhw4fYvn17kcbHBIiIiIi0bsyYMdDT08OFCxdgYmIilTs6OqJ79+4QQkChUGDHjh3SvJo1a+K7777DwIEDkZ6eDj29oktTOAaIiIiItOrJkyc4dOgQxo4dq5L8vE2hUGgsz/ol96JMfgAmQERERKRlkZGREEKgdu3aKuWVK1eGqakpTE1NMWXKFLXlHj9+jFmzZmHEiBFFHiMTICIiIioW586dQ1hYGOrWrYuUlBSVeYmJiejSpQvq1KmDwGIY5M0xQERERKRVTk5OUCgUiIiIUCl3dHQEABgZGamUv3jxAh07doSZmRl27doFfX39Io+RZ4CIiIhIqypVqoT27dtj2bJlSE5OzrFuYmIiOnToAAMDA/z+++8wNDQslhiZABEREZHWrVixAunp6WjcuDF+/fVX3LhxAxEREdi8eTNu3rwJXV1dKflJTk7G2rVrkZiYiNjYWMTGxiIjI6NI4+MlMCIiItK6mjVr4vLly5g9ezamTp2KBw8eQKlUok6dOpg0aRLGjBmDc+fO4ezZswDeXDZ72507d+Dg4FBk8TEBIiIiKmuK+0nQBWRnZ4elS5di6dKlGud7eXlBCFHMUb3BS2BEREQkO0yAiIiISHaYABEREZHsMAEiIiIi2WECREREVIqV1CDh0kpb24MJEBERUSmU9TTkly9flnAkpUvW9ijs06J5GzwREVEppKurC0tLS8THxwMAjI2Ns/0FdTkQQuDly5eIj4+HpaUldHV1C9UeEyCCV0hg7pVyqlJGnkdBRFTW2NraAoCUBBFgaWkpbZfCYAJERERUSikUCtjZ2cHa2hppaWklHU6J09fXL/SZnyxMgIiIiEo5XV1drX3x0xscBE1ERESyU6IJ0N9//w1fX1/Y29tDoVBg9+7dKvMVCoXG1/z587NtMzAwUK2+i4tLEfeEiIiIypISTYCSk5Ph5uaG5cuXa5wfExOj8lq3bh0UCgV69+6dY7t169ZVWe7kyZNFET4RERGVUSU6BqhTp07o1KlTtvPfHeW9Z88etGnTBo6Ojjm2q6enp5UR4kRERFQ+lZkxQHFxcdi3bx+GDRuWa93bt2/D3t4ejo6O8PPzQ3R0dDFESERERGVFmbkLbOPGjTAzM0OvXr1yrOfh4YENGzagdu3aiImJwYwZM9CyZUuEh4fDzMxM4zIpKSlISUmRphMTE7UaOxEREZUuZSYBWrduHfz8/GBoaJhjvbcvqdWvXx8eHh6oXr06fvvtt2zPHgUFBWHGjBlajZeIiIhKrzJxCezEiROIiIjAxx9/nO9lLS0tUatWLURGRmZbZ+rUqUhISJBe9+/fL0y4REREVMqViQRo7dq1cHd3h5ubW76XTUpKQlRUFOzs7LKto1QqYW5urvIiIiKi8qtEE6CkpCSEhYUhLCwMAHDnzh2EhYWpDFpOTExEcHBwtmd/2rVrh2XLlknTkyZNwl9//YW7d+/i9OnT6NmzJ3R1dTFgwIAi7QsRERGVHSU6BujChQto06aNND1x4kQAgL+/PzZs2AAA2LZtG4QQ2SYwUVFRePz4sTT94MEDDBgwAE+ePIGVlRVatGiBM2fOwMrKqug6QkRERGVKiSZAXl5eEELkWGfEiBEYMWJEtvPv3r2rMr1t2zZthEZERETlWJkYA0RERESkTUyAiIiISHaYABEREZHsMAEiIiIi2WECRERERLLDBIiIiIhkhwkQERERyQ4TICIiIpIdJkBEREQkO0yAiIiISHaYABEREZHsMAEiIiIi2WECRERERLLDBIiIiIhkhwkQERERyQ4TICIiIpIdJkBEREQkO0yAiIiISHaYABEREZHsMAEiIiIi2WECRERERLLDBIiIiIhkhwkQERERyQ4TICIiIpIdJkBEREQkO0yAiIiISHaYABEREZHsMAEiIiIi2WECRERERLLDBIiIiIhkhwkQERERyQ4TICIiIpIdJkBEREQkOyWaAP3999/w9fWFvb09FAoFdu/erTI/ICAACoVC5dWxY8dc212+fDkcHBxgaGgIDw8PnDt3roh6QERERGVRiSZAycnJcHNzw/Lly7Ot07FjR8TExEivX375Jcc2f/31V0ycOBHTp0/HpUuX4ObmBh8fH8THx2s7fCIiIiqj9Epy5Z06dUKnTp1yrKNUKmFra5vnNhcuXIjhw4djyJAhAIBVq1Zh3759WLduHb788stCxUtERETlQ6kfAxQSEgJra2vUrl0bo0ePxpMnT7Ktm5qaiosXL8Lb21sq09HRgbe3N0JDQ4sjXCIiIioDSvQMUG46duyIXr16oUaNGoiKisJXX32FTp06ITQ0FLq6umr1Hz9+jIyMDNjY2KiU29jY4ObNm9muJyUlBSkpKdJ0YmKi9jpBREREpU6pToD69+8v/d/V1RX169dHzZo1ERISgnbt2mltPUFBQZgxY4bW2iuPQkJymBdY8HYDC7EsERFRQZX6S2Bvc3R0ROXKlREZGalxfuXKlaGrq4u4uDiV8ri4uBzHEU2dOhUJCQnS6/79+1qNm4iIiEqXMpUAPXjwAE+ePIGdnZ3G+QYGBnB3d8fRo0elsszMTBw9ehSenp7ZtqtUKmFubq7yIiIiovKrRBOgpKQkhIWFISwsDABw584dhIWFITo6GklJSZg8eTLOnDmDu3fv4ujRo+jevTucnJzg4+MjtdGuXTssW7ZMmp44cSJ+/PFHbNy4ETdu3MDo0aORnJws3RVGREREVKJjgC5cuIA2bdpI0xMnTgQA+Pv7Y+XKlbh69So2btyI58+fw97eHh06dMCsWbOgVCqlZaKiovD48WNpul+/fnj06BGmTZuG2NhYNGjQAAcOHFAbGE1ERETyVaIJkJeXF4QQ2c4/ePBgrm3cvXtXrWzcuHEYN25cYUIjIiKicqxMjQEiIiIi0gYmQERERCQ7TICIiIhIdpgAERERkewwASIiIiLZYQJEREREssMEiIiIiGSHCRARERHJDhMgIiIikh0mQERERCQ7TICIiIhIdkr0t8CofPAKCSzwsiFeWf8WvI3sYvDyykcDgYVbPxERlS08A0RERESywwSIiIiIZIcJEBEREckOEyAiIiKSHSZAREREJDtMgIiIiEh2mAARERGR7DABIiIiItlhAkRERESywwSIiIiIZIcJEBEREckOEyAiIiKSHSZAREREJDtMgIiIiEh2mAARERGR7DABIiIiItlhAkRERESywwSIiIiIZIcJEBEREckOEyAiIiKSHSZAREREJDslmgD9/fff8PX1hb29PRQKBXbv3i3NS0tLw5QpU+Dq6goTExPY29tj8ODBePjwYY5tBgYGQqFQqLxcXFyKuCdERERUlpRoApScnAw3NzcsX75cbd7Lly9x6dIlfPPNN7h06RJ27tyJiIgIdOvWLdd269ati5iYGOl18uTJogifiIiIyii9klx5p06d0KlTJ43zLCwscPjwYZWyZcuWoWnTpoiOjka1atWybVdPTw+2trZajZWIiIjKjzI1BighIQEKhQKWlpY51rt9+zbs7e3h6OgIPz8/REdHF0+AREREVCaU6Bmg/Hj9+jWmTJmCAQMGwNzcPNt6Hh4e2LBhA2rXro2YmBjMmDEDLVu2RHh4OMzMzDQuk5KSgpSUFGk6MTFR6/ETERFR6VEmEqC0tDT07dsXQgisXLky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xqJChQpwd3fH6dOnUadOneIMmYiIiEqxMpEAdejQAUIIjfNCQkJUphctWoRFixYVQ1RERERUVpWJS2BERERE2sQEiIiIiGSHCRARERHJDhMgIiIikh0mQERERCQ7TICIiIhIdpgAERERkewwASIiIiLZYQJEREREssMEiIiIiGSHCRARERHJDhMgIiIikh0mQERERCQ7TICIiIhIdpgAERERkewwASIiIiLZYQJEREREssMEiIiIiGSHCRARERHJDhMgIiIikh29kg6ACs8rJLCkQyAiIipTeAaIiIiIZIcJEBEREckOEyAiIiKSHSZAREREJDtMgIiIiEh2mAARERGR7DABIiIiItlhAkRERESywwSIiIiIZIcJEBEREckOEyAiIiKSHSZAREREJDulOgEKDAyEQqFQebm4uOS4THBwMFxcXGBoaAhXV1f8+eefxRQtERERlRWlOgECgLp16yImJkZ6nTx5Mtu6p0+fxoABAzBs2DBcvnwZPXr0QI8ePRAeHl6MERMREVFpV+oTID09Pdja2kqvypUrZ1t3yZIl6NixIyZPnoz3338fs2bNQqNGjbBs2bJijJiIiIhKu1KfAN2+fRv29vZwdHSEn58foqOjs60bGhoKb29vlTIfHx+EhoYWdZhERERUhuiVdAA58fDwwIYNG1C7dm3ExMRgxowZaNmyJcLDw2FmZqZWPzY2FjY2NiplNjY2iI2NzXE9KSkpSElJkaYTExO10wEiIiIqlUp1AtSpUyfp//Xr14eHhweqV6+O3377DcOGDdPaeoKCgjBjxgyttZdfXiGBJbZuKl0CA8tWu0REZVWpvwT2NktLS9SqVQuRkZEa59va2iIuLk6lLC4uDra2tjm2O3XqVCQkJEiv+/fvay1mIiIiKn3KVAKUlJSEqKgo2NnZaZzv6emJo0ePqpQdPnwYnp6eObarVCphbm6u8iIiIqLyq1QnQJMmTcJff/2Fu3fv4vTp0+jZsyd0dXUxYMAAAMDgwYMxdepUqf748eNx4MABLFiwADdv3kRgYCAuXLiAcePGlVQXiIiIqBQq1WOAHjx4gAEDBuDJkyewsrJCixYtcObMGVhZWQEAoqOjoaPzvxyuWbNm2Lp1K77++mt89dVXcHZ2xu7du1GvXr2S6gIRERGVQqU6Adq2bVuO80NCQtTK+vTpgz59+hRRRERERFQelOpLYERERERFgQkQERERyQ4TICIiIpIdJkBEREQkO0yAiIiISHaYABEREZHsMAEiIiIi2WECRERERLLDBIiIiIhkhwkQERERyQ4TICIiIpKdUv1bYERERSEwsGy2TUTawzNAREREJDtMgIiIiEh2mAARERGR7DABIiIiItlhAkRERESywwSIiIiIZIcJEBEREckOEyAiIiKSHSZAREREJDtMgIiIiEh2mAARERGR7DABIiIiItlhAkRERESywwSIiIiIZIcJEBEREckOEyAiIiKSHSZAREREJDt6JR0AERW9wMDiWY9XSOFWFOL1v+WLK2YikieeASIiIiLZYQJEREREssMEiIiIiGSnVCdAQUFBaNKkCczMzGBtbY0ePXogIiIix2U2bNgAhUKh8jI0NCymiImIiKgsKNUJ0F9//YWxY8fizJkzOHz4MNLS0tChQwckJyfnuJy5uTliYmKk171794opYiIiIioLSvVdYAcOHFCZ3rBhA6ytrXHx4kW0atUq2+UUCgVsbW2LOjwiIiIqo0r1GaB3JSQkAAAqVqyYY72kpCRUr14dVatWRffu3fHPP/8UR3hERERURpSZBCgzMxMTJkxA8+bNUa9evWzr1a5dG+vWrcOePXuwefNmZGZmolmzZnjw4EG2y6SkpCAxMVHlRUREROVXqb4E9raxY8ciPDwcJ0+ezLGep6cnPD09pelmzZrh/fffx+rVqzFr1iyNywQFBWHGjBlajZeIiIhKrzJxBmjcuHHYu3cvjh8/jipVquRrWX19fTRs2BCRkZHZ1pk6dSoSEhKk1/379wsbMhEREZVipfoMkBACn3zyCXbt2oWQkBDUqFEj321kZGTg2rVr6Ny5c7Z1lEollEplYUIlIiKiMqRUJ0Bjx47F1q1bsWfPHpiZmSE2NhYAYGFhASMjIwDA4MGD8d577yEoKAgAMHPmTHzwwQdwcnLC8+fPMX/+fNy7dw8ff/xxifWDiIiISpdSnQCtXLkSAODl5aVSvn79egQEBAAAoqOjoaPzvyt5z549w/DhwxEbG4sKFSrA3d0dp0+fRp06dYorbCIiIirlSnUCJITItU5ISIjK9KJFi7Bo0aIiioiIiIjKgzIxCJqIiIhIm5gAERERkewwASIiIiLZKdVjgIjkxCsksFDLh3gVbnltxFAeFH4bFHZ5otIjMLBstZsfPANE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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "def initialize_page_rank(graph):\n",
    "    \"\"\"\n",
    "    Initialize the PageRank scores for all nodes in a graph.\n",
    "    \n",
    "    #TODO: Implement this function to initialize PageRank scores for each node.\n",
    "    - Assume each node has an equal initial PageRank value.\n",
    "    - The function takes a graph as input and returns a dictionary with nodes as keys and initial PageRank scores as values.\n",
    "    \"\"\"        \n",
    "    initial_page_rank = 1 / len(graph)\n",
    "    return {node: initial_page_rank for node in graph}\n",
    "        \n",
    "\n",
    "\n",
    "def calculate_page_rank(graph, d=0.85, max_iterations=100, tol=1e-6):\n",
    "    \"\"\"\n",
    "    Calculate the PageRank scores for all nodes in the graph using the iterative PageRank formula.\n",
    "    \n",
    "    #TODO: Implement this function to calculate the PageRank scores.\n",
    "    - Use the iterative PageRank algorithm with the damping factor `d`.\n",
    "    - Continue iterating until the change in PageRank scores is below `tol` or until `max_iterations` is reached.\n",
    "    - The function returns a dictionary with nodes as keys and their PageRank scores as values.\n",
    "    \"\"\"\n",
    "    N = len(graph)\n",
    "    page_rank = initialize_page_rank(graph)\n",
    "    for i in range(max_iterations):\n",
    "        new_page_rank = {}\n",
    "        for node in graph:\n",
    "            incoming_links = [n for n in graph if node in graph[n]]\n",
    "            sum_links = sum((page_rank[n] / len(graph[n])) for n in incoming_links)\n",
    "            new_page_rank[node] = (1 - d) / N + d * sum_links\n",
    "        \n",
    "        if all(abs(new_page_rank[node] - page_rank[node]) < tol for node in graph):\n",
    "            break\n",
    "        page_rank = new_page_rank\n",
    "    \n",
    "    return page_rank\n",
    "\n",
    "def visualize_distributions(page_ranks_g1, page_ranks_g2):\n",
    "    \"\"\"\n",
    "    Visualizes the distributions of PageRank values for two graphs on the same histogram for comparison.\n",
    "    \n",
    "    #TODO: Implement this function to visualize the distributions of PageRank values.\n",
    "    - Use a histogram to compare the PageRank distributions of two different graphs.\n",
    "    - Customize the plot with labels, titles, and a legend.\n",
    "    \"\"\"\n",
    "    plt.hist(list(page_ranks_g1.values()), alpha=0.5, label='G1', color='blue', bins=20)\n",
    "    plt.hist(list(page_ranks_g2.values()), alpha=0.5, label='G2', color='red', bins=20)\n",
    "    plt.xlabel('PageRank value')\n",
    "    plt.ylabel('Frequency')\n",
    "    plt.title('Comparison of PageRank Value Distributions for G1 and G2')\n",
    "    plt.legend(loc='upper right')\n",
    "    plt.show()\n",
    "\n",
    "# Generating larger graphs G1 and G2 with more complexity\n",
    "G1 = nx.fast_gnp_random_graph(100, 0.05, directed=True)\n",
    "G2 = nx.fast_gnp_random_graph(100, 0.05, directed=True)\n",
    "\n",
    "# Convert NetworkX graphs to adjacency lists\n",
    "graph_g1 = {n: list(G1.successors(n)) for n in G1.nodes()}\n",
    "graph_g2 = {n: list(G2.successors(n)) for n in G2.nodes()}\n",
    "\n",
    "#TODO: Use the `calculate_page_rank` function to calculate PageRank values for `graph_g1` and `graph_g2`.\n",
    "page_ranks_g1 = calculate_page_rank(graph_g1)\n",
    "page_ranks_g2 = calculate_page_rank(graph_g2)\n",
    "#TODO: Use the `visualize_distributions` function to compare the PageRank distributions of `graph_g1` and `graph_g2`.\n",
    "visualize_distributions(page_ranks_g1, page_ranks_g2)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "755b005c",
   "metadata": {},
   "source": [
    "\n",
    "## Part 2: Graph Convolutional Networks (GCN) for Graph Classification\n",
    "\n",
    "### Objective\n",
    "\n",
    "Your main objective is to modify the degree-based propagation rule using the following topological measures for normalization:\n",
    "1. **PageRank Centrality**: Utilize the PageRank centrality implementation from Part 1.\n",
    "2. **Betweenness Centrality**: Implement normalization using Betweenness centrality, which can be calculated using NetworkX.\n",
    "3. **Clustering Coefficient**: Implement normalization using the Clustering coefficient, available through NetworkX.\n",
    "\n",
    "For each normalization technique, you will develop a separate GCN model, resulting in four different GNN models including the original degree-based model.\n",
    "\n",
    "### Total Points: 15\n",
    "\n",
    "---\n",
    "\n",
    "### Instructions\n",
    "\n",
    "#### 1. **Implement a 2-Layer GCN**\n",
    "- **Graph Representation**: Begin with the simple node degree normalization for your initial GCN implementation. Specifically, employ the normalization technique $D^{-1}A$ where $D$ is the degree matrix and $A$ is the adjacency matrix of the graph.\n",
    "- **GCN Propagation Rule**: Implement the GCN layer using the updated propagation rule:\n",
    "  $$H^{(l+1)} = \\sigma(D^{-1}AH^{(l)}W^{(l)})$$\n",
    "  Here, $H^{(l)}$ represents the node features at layer $l$, $W^{(l)}$ is the weight matrix at layer $l$, and $\\sigma$ denotes a non-linear activation function, such as ReLU.\n",
    "- **Architecture**: Design your GCN with two convolutional layers following this propagation rule. Conclude with a Mean Pooling layer to aggregate node embeddings for graph-level prediction.\n",
    "- **Prediction Head**: Develop a prediction head that processes the pooled graph representation to classify the graph.\n",
    "\n",
    "#### 2. **Topological Measures for Normalization**\n",
    "Adapt the degree-based propagation rule to incorporate the following topological measures:\n",
    "- **PageRank Centrality**: Leverage the PageRank centrality implementation from Part 1.\n",
    "- **Betweenness Centrality and Clustering Coefficient**: Use NetworkX to compute these centrality measures for each node, applying them as normalization factors in the GCN propagation rule.\n",
    "\n",
    "#### 3. **Data Preparation**\n",
    "- Ensure the train and test sets are fixed to guarantee consistent evaluation across the different normalization techniques.\n",
    "\n",
    "#### 4. **Model Training and Evaluation**\n",
    "- Independently train a GCN model for each normalization technique.\n",
    "- Assess the performance of each model, focusing on accuracy, sensitivity, and specificity.\n",
    "\n",
    "#### 5. **Analysis of Embedding Distributions and Classification Results**\n",
    "- **Embedding Distributions**: Employ PCA to reduce the dimensionality of embeddings obtained from the final layer (Layer 2) of each GCN model, and visualize these distributions.\n",
    "- **Classification Results**: Contrast the accuracy, sensitivity, and specificity results among the four models.\n",
    "\n",
    "### Comment Section\n",
    "\n",
    "Engage in discussion on the following topics based on your analysis:\n",
    "- **Comparative Analysis**: Examine how the embedding distributions and classification outcomes differ with each normalization technique.\n",
    "- **Interpretation of Results**: Reflect on the performances of GCN models under different topological normalizations. Consider how PageRank centrality, betweenness centrality, and clustering coefficient influence the embeddings and classification capabilities of the models.\n",
    "\n",
    "Delve into the reasons behind the performances observed and theorize how the distinct topological characteristics of graphs may affect GCN model efficacy. Ponder the impact of each normalization method on feature propagation within the network and its capacity to harness the structural information of the graph for classification purposes.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "acedd480",
   "metadata": {},
   "source": [
    "## 1.1 Dataset "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "id": "8debb0c3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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2Ps2RXZsBDQ3TsL5uT75U4kUfa+Rw6P3ZCRsmYr40ElwaNQO9rWbAFIZcpJjob52HHYKJ95niKhSqbC/0S5dugQASLl+Brh+ptD2vOKpOHJHVyjf+Q7pf+6G5sZJaG6eBqRS2Ho+BbdXx0HR9EXIZVJ0alD8MthERFT5dWroiY3hd2AwCuU+1sbdB15DlyLtzHZkRYcj8/IRSOW2sPXyh8cbX8CpYRDvM0REVVynhp6Ie3kUUpw9kHn5CDSRZyCv7oEaLw2Hc+vXSz3epoY3vIcuQfq5vdBEnkFW1FlAECCv4QXnlt0w7P2PKuBdEBFVHIkgCOX/ZP4EohIy0GXRCbOd/8i4DvDzdDLb+YmIyLLxPkNERObE+wwRUflU+Kx1/jWd0N7PHTKpxKTnlUklaO/nzr+kiYiqON5niIjInHifISIqH1GWS5jTqxnkJv6LWi6VYE6vZiY9JxERWSfeZ4iIyJx4nyEiKjtRiicfVwfMCA4w6TlnBgfAx9XBpOckIiLrxPsMERGZE+8zRERlJ0rxBAD9WvtiQtcGJjnXp10bom9rX5Oci4iIKgfeZ4iIyJx4nyEiKpsKn1z8v7aei8W0fVehNwrlWoFIJpVALpVgZnAA/5ImIqJi8T5DRETmxPsMEVHJRC+eACAuRYPJuyPwR3QyZFJJiX9h521v7+eOOb2acTgqERGVivcZIiIyJ95niIiKZxHFU56ohAyEhsciLDIRsSoNHg0mAeDr5oBODTwxMMiXqz0QEVG58T5DRETmxPsMEVFhFlU8PUqt1SNGpYZOb4StXIq6bgoo7ORixyIiokqC9xkiIjIn3meIiHJZbPFERERERERERETWTbRV7YiIiIiIiIiIqHJj8URERERERERERGbB4omIiIiIiIiIiMyCxRMREREREREREZkFiyciIiIiIiIiIjILFk9ERERERERERGQWLJ6IiIiIiIiIiMgsWDwREREREREREZFZsHgiIiIiIiIiIiKzYPFERERERERERERmweKJiIiIiIiIiIjMgsUTERERERERERGZBYsnIiIiIiIiIiIyCxZPRERERERERERkFiyeiIiIiIiIiIjILFg8ERERERERERGRWbB4IiIiIiIiIiIis2DxREREREREREREZsHiiYiIiIiIiIiIzILFExERERERERERmQWLJyIiIiIiIiIiMgsWT0REREREREREZBYsnoiIiIiIiIiIyCxYPBERERERERERkVmweCIiIiIiIiIiIrNg8URERERERERERGbB4omIiIiIiIiIiMyCxRMREREREREREZnF/wCKWu/R+2o1xQAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 1500x500 with 3 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "def create_random_molecule_graph(num_atoms=9, num_bonds=12):\n",
    "    \"\"\"\n",
    "    Creates a random molecular graph with a specified number of atoms and bonds.\n",
    "\n",
    "    Args:\n",
    "        num_atoms (int): The number of atoms in the molecule.\n",
    "        num_bonds (int): The number of bonds between atoms in the molecule.\n",
    "\n",
    "    Returns:\n",
    "        (nx.Graph, list): A tuple containing the generated molecular graph and the list of atom labels.\n",
    "    \"\"\"\n",
    "    G = nx.Graph()\n",
    "    atom_labels = [f\"{i}:{random.choice(['H', 'C', 'O'])}\" for i in range(num_atoms)]\n",
    "    G.add_nodes_from(atom_labels)\n",
    "\n",
    "    for _ in range(num_bonds):\n",
    "        atom1, atom2 = random.choice(atom_labels), random.choice(atom_labels)\n",
    "        while atom1 == atom2 or G.has_edge(atom1, atom2):\n",
    "            atom1, atom2 = random.choice(atom_labels), random.choice(atom_labels)\n",
    "        G.add_edge(atom1, atom2)\n",
    "\n",
    "    return G, atom_labels\n",
    "\n",
    "def classify_molecule(G):\n",
    "    \"\"\"\n",
    "    Classifies a molecule into type 1 or type 2 based on more complex criteria (without considering nitrogen).\n",
    "\n",
    "    Args:\n",
    "        G (nx.Graph): The molecular graph.\n",
    "\n",
    "    Returns:\n",
    "        int: The type of molecule (0 or 1).\n",
    "    \"\"\"\n",
    "    atom_labels = list(G.nodes)\n",
    "    num_carbon = sum(1 for label in atom_labels if label.endswith(\":C\"))\n",
    "    num_oxygen = sum(1 for label in atom_labels if label.endswith(\":O\"))\n",
    "    num_hydrogen = sum(1 for label in atom_labels if label.endswith(\":H\"))\n",
    "    num_bonds = len(G.edges)\n",
    "\n",
    "    # Define more complex criteria for classification\n",
    "    if num_oxygen >= 2 and num_hydrogen >= 3 and num_carbon >= 2 and num_bonds >= 4:\n",
    "        return 1\n",
    "    elif num_oxygen >= 1 and num_carbon >= 3 and num_hydrogen >= 2 and num_bonds >= 3:\n",
    "        return 0\n",
    "    else:\n",
    "        return 2\n",
    "\n",
    "\n",
    "def generate_balanced_molecule_dataset(num_samples_per_type):\n",
    "    \"\"\"\n",
    "    Generates a balanced dataset of random molecule graphs.\n",
    "\n",
    "    Args:\n",
    "        num_samples_per_type (int): The number of samples per molecule type.\n",
    "\n",
    "    Returns:\n",
    "        list: A list of tuples, each containing a graph, its atom labels, and its classification.\n",
    "    \"\"\"\n",
    "    dataset = []\n",
    "    while len(dataset) < num_samples_per_type * 2:\n",
    "        G, atom_labels = create_random_molecule_graph()\n",
    "        classification = classify_molecule(G)\n",
    "        if classification in [0, 1]:\n",
    "            dataset.append((G, atom_labels, classification))\n",
    "    return dataset\n",
    "\n",
    "\n",
    "def graph_to_tensors(G, atom_labels, normalization_func=None):\n",
    "    \"\"\"\n",
    "    Converts a networkx graph of a molecule into tensor representations, with optional adjacency matrix normalization.\n",
    "\n",
    "    Args:\n",
    "        G (nx.Graph): The molecular graph.\n",
    "        atom_labels (list): List of atom labels in the molecule.\n",
    "        normalization_func (callable, optional): Function to normalize the adjacency matrix.\n",
    "\n",
    "    Returns:\n",
    "        tuple: A tuple containing the tensor representation of atom types (X) and (optionally normalized) adjacency matrix (A).\n",
    "    \"\"\"\n",
    "    atom_types = {'H': [1, 0, 0], 'C': [0, 1, 0], 'O': [0, 0, 1]}\n",
    "    X = torch.tensor([atom_types[node.split(':')[1]] for node in atom_labels], dtype=torch.float)\n",
    "\n",
    "    N = len(atom_labels)\n",
    "    A = torch.zeros((N, N), dtype=torch.float)\n",
    "    for i, j in G.edges:\n",
    "        idx1 = atom_labels.index(i)\n",
    "        idx2 = atom_labels.index(j)\n",
    "        A[idx1, idx2] = 1\n",
    "        A[idx2, idx1] = 1\n",
    "\n",
    "    if normalization_func:\n",
    "        A = normalization_func(G, A)  # Apply normalization if provided\n",
    "\n",
    "    return X, A\n",
    "\n",
    "from torch.utils.data import Dataset\n",
    "import torch\n",
    "\n",
    "class MoleculeDataset(Dataset):\n",
    "    \"\"\"\n",
    "    A custom dataset class for handling molecular graphs. Each sample in the dataset represents a molecule,\n",
    "    characterized by its graph structure and atom features, along with a classification label (e.g., for binary classification tasks).\n",
    "\n",
    "    The dataset initializes with a list of molecular data and optionally applies a normalization function to the adjacency matrix\n",
    "    of each molecule's graph. This normalization can be crucial for certain graph neural network models, affecting how the model\n",
    "    interprets the connectivity and flow of information through the graph.\n",
    "\n",
    "    Attributes:\n",
    "        dataset (list): A list of tuples, where each tuple corresponds to a molecule and contains:\n",
    "                        - A graph representation of the molecule (e.g., a `networkx` graph).\n",
    "                        - Atom labels or features as a list or array.\n",
    "                        - A classification label for the molecule.\n",
    "        normalization_func (callable, optional): A function that takes an adjacency matrix (and potentially a graph) as input\n",
    "                        and returns a normalized adjacency matrix. This can be any normalization technique, such as degree normalization,\n",
    "                        PageRank-based normalization, etc.\n",
    "\n",
    "    The class provides two main methods as part of the PyTorch `Dataset` interface:\n",
    "    - `__len__` returns the number of items in the dataset.\n",
    "    - `__getitem__` retrieves a single item from the dataset by index, applying the normalization function if provided.\n",
    "\n",
    "    #TODO: Implement the `__init__` method to initialize the dataset with the given list and normalization function.\n",
    "    - Store the provided dataset list and normalization function as instance attributes.\n",
    "\n",
    "    #TODO: Implement the `__len__` method to return the size of the dataset.\n",
    "    - This method should simply return the length of the dataset list.\n",
    "\n",
    "    #TODO: Implement the `__getitem__` method to fetch and preprocess a single graph representation from the dataset.\n",
    "    - Extract the graph `G`, atom labels `atom_labels`, and classification `classification` for the specified index `idx`.\n",
    "    - Convert the graph `G` and `atom_labels` into tensor formats suitable for graph neural networks. This often involves creating a feature matrix `X` for nodes and an adjacency matrix `A`. Use the `graph_to_tensors` function (to be implemented separately by the student) for this conversion. Apply the `normalization_func` to the adjacency matrix if provided.\n",
    "    - Convert the `classification` label into a tensor of type `torch.long`.\n",
    "    - Return the feature matrix `X`, the (optionally normalized) adjacency matrix `A`, and the label tensor.\n",
    "    \"\"\"\n",
    "    def __init__(self, dataset, normalization_func=None):\n",
    "        super(MoleculeDataset, self).__init__()\n",
    "        \n",
    "        self.dataset = dataset\n",
    "        self.normalization_func = normalization_func\n",
    "        \n",
    "\n",
    "    def __len__(self):\n",
    "        return len(self.dataset)\n",
    "\n",
    "    def __getitem__(self, idx):\n",
    "        G, atom_labels, classification = self.dataset[idx]\n",
    "        X, A = graph_to_tensors(G, atom_labels, self.normalization_func)\n",
    "        classification_tensor = torch.tensor([classification], dtype=torch.long)\n",
    "        return X, A, classification_tensor\n",
    "\n",
    "\n",
    "def prepare_data_loaders(num_samples_per_type, normalization_func=None, batch_size=10):\n",
    "    \"\"\"\n",
    "    Prepares DataLoader for training and testing datasets.\n",
    "\n",
    "    Args:\n",
    "        num_samples_per_type (int): Number of samples per class to generate.\n",
    "        normalization_func (callable, optional): Normalization function to apply to adjacency matrices.\n",
    "        batch_size (int): Size of each data batch.\n",
    "\n",
    "    Returns:\n",
    "        Tuple of DataLoader: Training and testing DataLoader objects.\n",
    "    \"\"\"\n",
    "    dataset = generate_balanced_molecule_dataset(num_samples_per_type)\n",
    "    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=random_seed)\n",
    "\n",
    "    train_dataset = MoleculeDataset(train_dataset, normalization_func=normalization_func)\n",
    "    test_dataset = MoleculeDataset(test_dataset, normalization_func=normalization_func)\n",
    "\n",
    "    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)\n",
    "    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)\n",
    "\n",
    "    return train_loader, test_loader\n",
    "\n",
    "# Your GCNLayer and GCN classes remain unchanged.\n",
    "\n",
    "# Generating a few random molecule graphs for visualization\n",
    "graphs = [create_random_molecule_graph() for _ in range(3)]\n",
    "\n",
    "# Plotting the generated graphs\n",
    "fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n",
    "for i, (G, atom_labels) in enumerate(graphs):\n",
    "    pos = nx.spring_layout(G)  # Using spring layout for visual aesthetics\n",
    "    nx.draw(G, pos, with_labels=True, ax=axes[i])\n",
    "    axes[i].set_title(f\"Molecule Graph {i+1}\")\n",
    "\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1e2ba759",
   "metadata": {},
   "source": [
    "## 1.2 Visualization Helper Functions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "id": "e02d5fe4",
   "metadata": {},
   "outputs": [],
   "source": [
    "def visualize_omega_distribution(layers, epochs):\n",
    "    \"\"\"\n",
    "    Visualizes the distribution of weights (Omega) in each GCN layer of the Graph Neural Network across different epochs using smooth KDE plots.\n",
    "\n",
    "    Args:\n",
    "        layers (list of tuples): Each tuple contains a layer's name and a list of arrays representing the layer's weights at different epochs.\n",
    "        epochs (list of int): List of epoch numbers corresponding to the weight arrays.\n",
    "\n",
    "    This function plots the kernel density estimation (KDE) of weight values for each layer across specified epochs, allowing for the observation of how weight distributions evolve during training.\n",
    "    \"\"\"\n",
    "    for layer_name, weight_arrays in layers:\n",
    "        plt.figure(figsize=(12, 6))\n",
    "        for i, weights in enumerate(weight_arrays):\n",
    "            omega_values = weights.flatten()  # Flatten the array to get a distribution of individual weight values\n",
    "            sns.kdeplot(omega_values, fill=True, label=f'Epoch {epochs[i]}')  # Use 'fill' for shaded KDE plots\n",
    "        plt.title(f'{layer_name} Weight Distribution Across Epochs', fontsize=16)\n",
    "        plt.xlabel('Weight Values', fontsize=12)\n",
    "        plt.ylabel('Density', fontsize=12)\n",
    "        plt.legend()\n",
    "        plt.show()\n",
    "\n",
    "\n",
    "def plot_training_losses(losses_list, normalization_names):\n",
    "    \"\"\"\n",
    "    Plots the training loss over epochs for different normalization techniques.\n",
    "\n",
    "    Args:\n",
    "        losses_list (list of lists): Each sublist contains the training losses for one normalization technique over all epochs.\n",
    "        normalization_names (list of str): Names of the normalization techniques used.\n",
    "\n",
    "    This function creates a line plot for each normalization technique's training loss over epochs, facilitating comparison of their performance.\n",
    "    \"\"\"\n",
    "    plt.figure(figsize=(10, 6))\n",
    "    for i, train_losses in enumerate(losses_list):\n",
    "        plt.plot(range(1, len(train_losses) + 1), train_losses, label=f'{normalization_names[i]} Normalization')\n",
    "    plt.title('Training Loss Over Epochs for Different Normalization Techniques', fontsize=16)\n",
    "    plt.xlabel('Epoch', fontsize=12)\n",
    "    plt.ylabel('Loss', fontsize=12)\n",
    "    plt.legend()\n",
    "    plt.grid(True)\n",
    "    plt.show()\n",
    "\n",
    "\n",
    "def plot_metric_bar_charts(names, metric_values, metric_names):\n",
    "    \"\"\"\n",
    "    Creates bar charts for different evaluation metrics across various normalization techniques.\n",
    "\n",
    "    Args:\n",
    "        names (list of str): Names of the normalization techniques.\n",
    "        metric_values (list of lists): Each sublist contains the values of a metric for each normalization technique.\n",
    "        metric_names (list of str): Names of the metrics being plotted.\n",
    "\n",
    "    This function plots a bar chart for each provided metric, comparing the performance of different normalization techniques,\n",
    "    with y-axis limits dynamically adjusted to emphasize differences while capping at 1.\n",
    "    \"\"\"\n",
    "    num_metrics = len(metric_names)\n",
    "    num_rows = num_cols = int(math.ceil(math.sqrt(num_metrics)))\n",
    "\n",
    "    sns.set(style='whitegrid', palette='muted', font_scale=1.2)\n",
    "\n",
    "    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12 * num_cols, 8 * num_rows))\n",
    "    fig.subplots_adjust(hspace=0.4, wspace=0.4)\n",
    "\n",
    "    for i, metric_name in enumerate(metric_names):\n",
    "        ax = axes.flatten()[i] if num_metrics > 1 else axes\n",
    "\n",
    "        # Create DataFrame for seaborn\n",
    "        data = pd.DataFrame({\n",
    "            'Normalization Technique': np.repeat(names, len(metric_values[i])),\n",
    "            metric_name: np.concatenate([metric_values[i] for _ in names])\n",
    "        })\n",
    "\n",
    "        sns.barplot(x='Normalization Technique', y=metric_name, data=data, ax=ax, alpha=0.75)\n",
    "\n",
    "        # Dynamically adjust the y-axis limits\n",
    "        min_val = min(data[metric_name]) * 0.9  # Start slightly below the smallest value for better visibility\n",
    "        max_val = 1  # Ensuring the upper limit is 1\n",
    "        ax.set_ylim([min_val, max_val])\n",
    "\n",
    "        ax.set_xlabel('Normalization Technique', fontsize=14)\n",
    "        ax.set_ylabel(f'{metric_name} Value', fontsize=14)\n",
    "        ax.set_title(f'Comparison of {metric_name}', fontsize=16)\n",
    "        ax.tick_params(axis='x', rotation=45, labelsize=12)\n",
    "        ax.tick_params(axis='y', labelsize=12)\n",
    "\n",
    "        # Add text labels above bars\n",
    "        for p, value in zip(ax.patches, np.concatenate([metric_values[i] for _ in names])):\n",
    "            ax.text(p.get_x() + p.get_width() / 2., p.get_height(), f'{value:.2f}', ha='center', va='bottom', fontsize=10)\n",
    "\n",
    "    # Hide unused subplots if the number of metrics is less than the number of subplot positions\n",
    "    for i in range(num_metrics, num_rows * num_cols):\n",
    "        if num_rows * num_cols == 1:\n",
    "            break\n",
    "        fig.delaxes(axes.flatten()[i])\n",
    "\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "\n",
    "\n",
    "def extract_embeddings(model, loader, device='cpu'):\n",
    "    \"\"\"\n",
    "    Extracts embeddings from a model given a data loader.\n",
    "\n",
    "    #TODO: Implement this function to extract embeddings from the provided model using data from the loader.\n",
    "    - Set the model to evaluation mode.\n",
    "    - Initialize lists or arrays to store embeddings and labels.\n",
    "    - Iterate over batches of data from the loader, ensuring to move the data to the specified device.\n",
    "    - For each batch, use the model to compute embeddings. If the model requires specific inputs (e.g., features and adjacency matrix), ensure they are correctly passed.\n",
    "    - Apply necessary post-processing on embeddings (e.g., mean pooling) and convert them to a suitable format (e.g., numpy array) for further analysis or visualization.\n",
    "    - Collect and store the labels associated with each embedding for potential use in tasks like visualization or analysis.\n",
    "    - Return the embeddings and labels as a tuple. Ensure embeddings are in a continuous array format suitable for analysis.\n",
    "    \n",
    "    Args:\n",
    "        model (torch.nn.Module): The trained model from which to extract embeddings.\n",
    "        loader (DataLoader): DataLoader providing batches of data for embedding extraction.\n",
    "        device (str): Device to run the model on ('cpu' or 'cuda').\n",
    "\n",
    "    Returns:\n",
    "        tuple: A tuple containing two elements. The first is a numpy array of embeddings, and the second is a list of labels associated with each embedding.\n",
    "\n",
    "    Note: This function should handle device placement (CPU or GPU) for both the data and model, and ensure gradients are not computed to optimize memory and compute resources.\n",
    "    \"\"\"\n",
    "    \n",
    "    model.eval()\n",
    "    embeddings = []\n",
    "    labels = []\n",
    "    \n",
    "    for batch in loader:\n",
    "        batch = batch.to(device)\n",
    "        features = batch[0]\n",
    "        adjacency_matrix = batch[1]\n",
    "        label = batch[2]\n",
    "        \n",
    "        embedding = model(features, adjacency_matrix)\n",
    "        mean_embedding = global_mean_pooling(embedding)\n",
    "        \n",
    "        embeddings.append(mean_embedding)\n",
    "        labels.append(label)\n",
    "    \n",
    "    embeddings = np.vstack(embeddings)\n",
    "    labels = np.concatenate(labels)\n",
    "    \n",
    "    return embeddings, labels    \n",
    "\n",
    "def global_mean_pooling(X: torch.Tensor):\n",
    "    \"\"\"\n",
    "    Params:\n",
    "        X (tensor): Node feature matrix of shape (Nxd)\n",
    "    Returns:\n",
    "        Combined embeddings vector of shape (num_classes,)\n",
    "    \"\"\"\n",
    "    return torch.mean(X, dim=0)\n",
    "\n",
    "\n",
    "def apply_pca_and_visualize_all_sns(embeddings_list, titles, labels_list):\n",
    "    \"\"\"\n",
    "    Applies PCA to reduce dimensionality of embeddings and visualizes them using scatter plots.\n",
    "\n",
    "    Args:\n",
    "        embeddings_list (list of np.ndarray): List of embeddings arrays to be visualized.\n",
    "        titles (list of str): Titles for each subplot, typically representing the condition or category of the embeddings.\n",
    "        labels_list (list of np.ndarray): List of label arrays corresponding to the embeddings for coloring the points.\n",
    "\n",
    "    This function reduces embeddings to two principal components using PCA and plots them, coloring points by their labels to distinguish between categories.\n",
    "    \"\"\"\n",
    "    sns.set(style='whitegrid')\n",
    "    fig, axs = plt.subplots(2, 2, figsize=(16, 12))\n",
    "    axs = axs.flatten()\n",
    "\n",
    "    for i, (embeddings, title, labels) in enumerate(zip(embeddings_list, titles, labels_list)):\n",
    "        pca = PCA(n_components=2)\n",
    "        pca_embeddings = pca.fit_transform(embeddings)\n",
    "        df = pd.DataFrame(data=pca_embeddings, columns=['PCA1', 'PCA2'])\n",
    "        df['Label'] = labels  # Add labels for coloring\n",
    "\n",
    "        sns.scatterplot(ax=axs[i], x='PCA1', y='PCA2', hue='Label', data=df, palette='viridis', alpha=0.7).set_title(title)\n",
    "        axs[i].set_xlabel('PCA Component 1')\n",
    "        axs[i].set_ylabel('PCA Component 2')\n",
    "\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a793cd23",
   "metadata": {},
   "source": [
    "## 1.3 Model "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "2ae7ecb0",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'nn' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[1], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m \u001b[38;5;28;01mclass\u001b[39;00m \u001b[38;5;21;01mGCNLayer\u001b[39;00m(\u001b[43mnn\u001b[49m\u001b[38;5;241m.\u001b[39mModule):\n\u001b[0;32m      2\u001b[0m \u001b[38;5;250m    \u001b[39m\u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[0;32m      3\u001b[0m \u001b[38;5;124;03m    Implements a single Graph Convolutional Layer.\u001b[39;00m\n\u001b[0;32m      4\u001b[0m \n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m     14\u001b[0m \u001b[38;5;124;03m        - Accept the number of output features per node.\u001b[39;00m\n\u001b[0;32m     15\u001b[0m \u001b[38;5;124;03m    \"\"\"\u001b[39;00m\n\u001b[0;32m     16\u001b[0m     \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21m__init__\u001b[39m(\u001b[38;5;28mself\u001b[39m, in_features, out_features):\n",
      "\u001b[1;31mNameError\u001b[0m: name 'nn' is not defined"
     ]
    }
   ],
   "source": [
    "class GCNLayer(nn.Module):\n",
    "    \"\"\"\n",
    "    Implements a single Graph Convolutional Layer.\n",
    "\n",
    "    #TODO: Implement a GCN layer that performs graph convolution. This layer should first apply a linear transformation\n",
    "    to the node features and then utilize the adjacency matrix to incorporate neighborhood information. You will need to\n",
    "    define and initialize a weight matrix for the linear transformation of node features.\n",
    "\n",
    "    Attributes:\n",
    "        - Define an attribute for the weight matrix.\n",
    "\n",
    "    Args:\n",
    "        - Accept the number of input features per node.\n",
    "        - Accept the number of output features per node.\n",
    "    \"\"\"\n",
    "    def __init__(self, in_features, out_features):\n",
    "        super(GCNLayer, self).__init__()\n",
    "        #TODO: Initialize the weight matrix as a torch.nn.Parameter.\n",
    "        pass\n",
    "\n",
    "    def init_parameters(self):\n",
    "        #TODO: Implement a method to initialize the weights uniformly with a standard deviation based on layer size.\n",
    "        pass\n",
    "\n",
    "    def forward(self, input, adjacency):\n",
    "        \"\"\"\n",
    "        Forward pass of the GCN layer.\n",
    "\n",
    "        #TODO: Implement the forward pass method. Apply a linear transformation to the input features and then\n",
    "        use the adjacency matrix to incorporate neighborhood information. The method should return the output feature\n",
    "        matrix after graph convolution.\n",
    "\n",
    "        Args:\n",
    "            - input: Input feature matrix where each row represents node features.\n",
    "            - adjacency: Adjacency matrix of the graph.\n",
    "\n",
    "        Returns:\n",
    "            - Output feature matrix after applying the graph convolution.\n",
    "        \"\"\"\n",
    "        pass\n",
    "\n",
    "class GCN(nn.Module):\n",
    "    \"\"\"\n",
    "    Implements a Graph Convolutional Network (GCN) for node classification.\n",
    "\n",
    "    #TODO: Implement a GCN model for node classification. The model should consist of two GCN layers followed by a\n",
    "    global mean pooling and a fully connected layer for classification. You need to define the GCN layers and the fully\n",
    "    connected layer in the constructor.\n",
    "\n",
    "    Args:\n",
    "        - nfeat: Number of features for each input node.\n",
    "        - nhid: Number of hidden units for each GCN layer.\n",
    "        - nclass: Number of classes (output dimension).\n",
    "    \"\"\"\n",
    "    def __init__(self, nfeat, nhid, nclass):\n",
    "        super(GCN, self).__init__()\n",
    "        #TODO: Define the first and second graph convolutional layers and the fully connected layer for classification.\n",
    "\n",
    "    def forward(self, x, adj, return_embedding=False):\n",
    "        \"\"\"\n",
    "        Forward pass of the GCN.\n",
    "\n",
    "        #TODO: Implement the forward pass. Apply two GCN layers with ReLU activation, perform global mean pooling,\n",
    "        and then use a fully connected layer for classification. The output should be appropriate for the loss function\n",
    "        you plan to use. For instance, if using CrossEntropyLoss, ensure the output is the raw logits. If `return_embedding`\n",
    "        is True, return the embedding from the second GCN layer before classification.\n",
    "\n",
    "        Args:\n",
    "            - x: Input feature matrix where each row is the feature vector of a node.\n",
    "            - adj: Adjacency matrix of the graph.\n",
    "            - return_embedding: If True, returns the embedding from the second GCN layer before classification.\n",
    "\n",
    "        Returns:\n",
    "            - The output appropriate for your chosen loss function if return_embedding is False, otherwise\n",
    "              returns the embeddings from the second GCN layer.\n",
    "        \"\"\"\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "33a96b83",
   "metadata": {},
   "source": [
    "## 1.4 Normalization Methods"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "id": "1dec72e2",
   "metadata": {},
   "outputs": [],
   "source": [
    "def compute_degree_matrix_normalization(G, adjacency):\n",
    "    \"\"\"\n",
    "    Computes the degree matrix normalization D^-1 * A for the given graph.\n",
    "\n",
    "    #TODO: Implement this function to normalize the adjacency matrix by the inverse degree of each node.\n",
    "    - Calculate the degree for each node.\n",
    "    - Compute the inverse degree matrix D^-1.\n",
    "    - Normalize the adjacency matrix using D^-1 * A.\n",
    "    - Ensure to handle cases with isolated nodes by adding a small epsilon to the degrees to prevent division by zero.\n",
    "    - Convert and return the normalized adjacency matrix as a PyTorch tensor.\n",
    "    \"\"\"\n",
    "    # degrees = np.sum(adjacency, axis=1)\n",
    "    # epsilon = 1e-6\n",
    "    # adjacency = torch.tensor(adjacency, dtype=torch.float)\n",
    "    # degrees[degrees == 0] += epsilon\n",
    "    # D_inv = torch.diag(1.0 / degrees)\n",
    "    # normalized_adjacency = torch.mm(D_inv, adjacency)\n",
    "    \n",
    "    # return normalized_adjacency\n",
    "\n",
    "    adjacency = torch.tensor(adjacency, dtype=torch.float) if not isinstance(adjacency, torch.Tensor) else adjacency\n",
    "    degrees = np.array([degree for node, degree in G.degree()], dtype=np.float32)\n",
    "    degrees += 1e-5\n",
    "    D_inv = torch.diag(1.0 / torch.tensor(degrees, dtype=torch.float))\n",
    "    \n",
    "    normalized_adjacency = torch.mm(D_inv, adjacency)\n",
    "    \n",
    "    return normalized_adjacency\n",
    "\n",
    "\n",
    "def compute_pagerank_normalization(G, adjacency):\n",
    "    \"\"\"\n",
    "    Normalizes the adjacency matrix using PageRank centrality values.\n",
    "\n",
    "    #TODO: Implement this function to apply PageRank normalization on the adjacency matrix.\n",
    "    - Compute PageRank values for each node in the graph.\n",
    "    - Create a diagonal matrix with PageRank values.\n",
    "    - Normalize the adjacency matrix using the PageRank diagonal matrix.\n",
    "    - Convert and return the normalized adjacency matrix as a PyTorch tensor.\n",
    "    \"\"\"\n",
    "    adjacency = torch.tensor(adjacency, dtype=torch.float)\n",
    "    pagerank_values = nx.pagerank(G)\n",
    "    diagonal = [pagerank_values[i] for i in range(len(G.nodes()))]\n",
    "    diagonal = torch.diag(torch.tensor(diagonal, dtype=torch.float))\n",
    "    normalized_adjacency = torch.mm(diagonal, adjacency)\n",
    "    \n",
    "    return normalized_adjacency    \n",
    "\n",
    "def compute_betweenness_normalization(G, adjacency):\n",
    "    \"\"\"\n",
    "    Normalizes the adjacency matrix using Betweenness centrality values.\n",
    "\n",
    "    #TODO: Implement this function to utilize Betweenness centrality for adjacency matrix normalization.\n",
    "    - Calculate Betweenness centrality for each node.\n",
    "    - Construct a diagonal matrix using the centrality values.\n",
    "    - Apply this matrix to normalize the adjacency matrix.\n",
    "    - Convert and return the normalized adjacency matrix as a PyTorch tensor.x\n",
    "    \"\"\"\n",
    "    adjacency = torch.tensor(adjacency, dtype=torch.float)\n",
    "    betweenness_centrality = nx.betweenness_centrality(G)\n",
    "    diagonal = [betweenness_centrality[i] for i in range(len(G.nodes()))]\n",
    "    diagonal = torch.diag(torch.tensor(diagonal, dtype=torch.float))\n",
    "    normalized_adjacency = torch.mm(diagonal, adjacency)\n",
    "    \n",
    "    return normalized_adjacency\n",
    "\n",
    "def compute_clustering_coefficient_normalization(G, adjacency):\n",
    "    \"\"\"\n",
    "    Normalizes the adjacency matrix using Clustering coefficient values.\n",
    "\n",
    "    #TODO: Implement this function to leverage Clustering coefficients for adjacency matrix normalization.\n",
    "    - Compute the Clustering coefficient for each node.\n",
    "    - Form a diagonal matrix with these coefficients.\n",
    "    - Normalize the adjacency matrix using this coefficient matrix.\n",
    "    - Convert and return the normalized adjacency matrix as a PyTorch tensor.\n",
    "    \"\"\"\n",
    "    adjacency = torch.tensor(adjacency, dtype=torch.float)\n",
    "    clustering_coefficients = nx.clustering(G)\n",
    "    diagonal = [clustering_coefficients[i] for i in range(len(G.nodes()))]\n",
    "    diagonal = torch.diag(torch.tensor(diagonal, dtype=torch.float))\n",
    "    normalized_adjacency = torch.mm(diagonal, adjacency)\n",
    "    \n",
    "    return normalized_adjacency"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "id": "fddfa2d9",
   "metadata": {},
   "outputs": [],
   "source": [
    "def compute_degree_matrix_normalization(G, adjacency):\n",
    "    degrees = np.array([val for (node, val) in G.degree()])\n",
    "    # Adding a small epsilon to avoid division by zero for isolated nodes\n",
    "    inv_degrees = 1.0 / (degrees + 1e-5)\n",
    "    D_inv = np.diag(inv_degrees)\n",
    "    normalized_adjacency = np.dot(D_inv, adjacency)\n",
    "    return torch.tensor(normalized_adjacency, dtype=torch.float)\n",
    "\n",
    "def compute_pagerank_normalization(G, adjacency):\n",
    "    pagerank = nx.pagerank(G)\n",
    "    pagerank_values = np.array([pagerank[node] for node in G.nodes()])\n",
    "    D_pr = np.diag(pagerank_values)\n",
    "    normalized_adjacency = np.dot(D_pr, adjacency)\n",
    "    return torch.tensor(normalized_adjacency, dtype=torch.float)\n",
    "\n",
    "def compute_betweenness_normalization(G, adjacency):\n",
    "    betweenness = nx.betweenness_centrality(G)\n",
    "    betweenness_values = np.array([betweenness[node] for node in G.nodes()])\n",
    "    D_betweenness = np.diag(betweenness_values)\n",
    "    normalized_adjacency = np.dot(D_betweenness, adjacency)\n",
    "    return torch.tensor(normalized_adjacency, dtype=torch.float)\n",
    "\n",
    "def compute_clustering_coefficient_normalization(G, adjacency):\n",
    "    clustering_coefficients = nx.clustering(G)\n",
    "    clustering_values = np.array([clustering_coefficients[node] for node in G.nodes()])\n",
    "    D_clustering = np.diag(clustering_values)\n",
    "    normalized_adjacency = np.dot(D_clustering, adjacency)\n",
    "    return torch.tensor(normalized_adjacency, dtype=torch.float)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "951e56d0",
   "metadata": {},
   "source": [
    "## 1.5 Training & Evaluation "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "id": "0336e8bc",
   "metadata": {},
   "outputs": [],
   "source": [
    "def train_model(model, train_loader, optimizer, criterion, epochs=100, device='cpu'):\n",
    "    \"\"\"\n",
    "    Trains the model over a specified number of epochs.\n",
    "\n",
    "    Args:\n",
    "        model (torch.nn.Module): The neural network model to be trained.\n",
    "        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.\n",
    "        optimizer (torch.optim.Optimizer): Optimizer used for model parameter updates.\n",
    "        criterion (torch.nn.Module): Loss function used for training.\n",
    "        epochs (int, optional): Number of epochs to train the model. Defaults to 100.\n",
    "        device (str, optional): The device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.\n",
    "\n",
    "    Returns:\n",
    "        list: A list containing the average loss value for each epoch.\n",
    "\n",
    "    This function iterates over the training dataset for a given number of epochs, performing\n",
    "    forward and backward passes, and updates the model parameters. The average loss per epoch is recorded and returned.\n",
    "    \"\"\"\n",
    "    model.train()  # Set the model to training mode\n",
    "    loss_values = []  # Initialize a list to store the average loss per epoch\n",
    "\n",
    "    for epoch in range(epochs):\n",
    "        total_loss = 0  # Track total loss for each epoch\n",
    "\n",
    "        for X, A, labels in train_loader:\n",
    "            # Move data to the specified device\n",
    "            X, A, labels = X.to(device), A.to(device), labels.to(device)\n",
    "\n",
    "            optimizer.zero_grad()  # Clear gradients for the next train step\n",
    "            output = model(X, A)  # Forward pass\n",
    "\n",
    "            loss = criterion(output, labels)  # Compute the loss\n",
    "            loss.backward()  # Backward pass to compute gradients\n",
    "            optimizer.step()  # Update model parameters\n",
    "\n",
    "            total_loss += loss.item()  # Accumulate the loss\n",
    "\n",
    "        avg_loss = total_loss / len(train_loader)  # Calculate average loss\n",
    "        loss_values.append(avg_loss)  # Append average loss to list\n",
    "\n",
    "        # Print the average loss for the current epoch\n",
    "        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')\n",
    "\n",
    "    return loss_values\n",
    "\n",
    "\n",
    "def evaluate_model(model, test_loader, device='cpu'):\n",
    "    \"\"\"\n",
    "    Evaluates the model on a test dataset.\n",
    "\n",
    "    Args:\n",
    "        model (torch.nn.Module): The neural network model to be evaluated.\n",
    "        test_loader (torch.utils.data.DataLoader): DataLoader for the test data.\n",
    "        device (str, optional): The device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.\n",
    "\n",
    "    Returns:\n",
    "        tuple: A tuple containing the accuracy, precision, recall, and F1 score of the model on the test dataset.\n",
    "\n",
    "    This function performs a forward pass on the test dataset to obtain the model's predictions,\n",
    "    then calculates and returns various evaluation metrics including accuracy, precision, recall, and F1 score.\n",
    "    \"\"\"\n",
    "    model.eval()  # Set the model to evaluation mode\n",
    "    true_labels = []  # List to store actual labels\n",
    "    predictions = []  # List to store model predictions\n",
    "\n",
    "    with torch.no_grad():  # Disable gradient computation\n",
    "        for X, A, labels in test_loader:\n",
    "            # Move data to the specified device\n",
    "            X, A, labels = X.to(device), A.to(device), labels.to(device)\n",
    "\n",
    "            output = model(X, A)  # Forward pass\n",
    "            _, predicted = torch.max(output.data, 1)  # Get the index of the max log-probability\n",
    "\n",
    "            true_labels += labels.tolist()  # Append actual labels\n",
    "            predictions += predicted.tolist()  # Append predicted labels\n",
    "\n",
    "    # Calculate evaluation metrics\n",
    "    accuracy = accuracy_score(true_labels, predictions)\n",
    "    precision = precision_score(true_labels, predictions, average='weighted')\n",
    "    recall = recall_score(true_labels, predictions, average='weighted')\n",
    "    f1 = f1_score(true_labels, predictions, average='weighted')\n",
    "\n",
    "    return accuracy, precision, recall, f1\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "484968ac",
   "metadata": {},
   "source": [
    "## 1.6 Main Script"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "id": "7cde7dfa",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Training model with degree normalization...\n",
      "DataLoader batch size: 50\n",
      "input shape: torch.Size([50, 9, 3])\n",
      "weight shape: torch.Size([3, 16])\n",
      "input shape: torch.Size([50, 9, 16])\n",
      "weight shape: torch.Size([16, 16])\n"
     ]
    },
    {
     "ename": "ValueError",
     "evalue": "Expected input batch_size (9) to match target batch_size (50).",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[93], line 83\u001b[0m\n\u001b[0;32m     80\u001b[0m     apply_pca_and_visualize_all_sns(embeddings_list, titles, labels_list)\n\u001b[0;32m     82\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;18m__name__\u001b[39m \u001b[38;5;241m==\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m__main__\u001b[39m\u001b[38;5;124m\"\u001b[39m:\n\u001b[1;32m---> 83\u001b[0m     \u001b[43mmain\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\n",
      "Cell \u001b[1;32mIn[93], line 43\u001b[0m, in \u001b[0;36mmain\u001b[1;34m()\u001b[0m\n\u001b[0;32m     40\u001b[0m criterion \u001b[38;5;241m=\u001b[39m nn\u001b[38;5;241m.\u001b[39mCrossEntropyLoss()\n\u001b[0;32m     42\u001b[0m \u001b[38;5;66;03m# Train the model\u001b[39;00m\n\u001b[1;32m---> 43\u001b[0m train_losses \u001b[38;5;241m=\u001b[39m \u001b[43mtrain_model\u001b[49m\u001b[43m(\u001b[49m\u001b[43mmodel\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mtrain_loader\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43moptimizer\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mcriterion\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mepochs\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mnum_epochs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mdevice\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mdevice\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m     44\u001b[0m train_losses_list\u001b[38;5;241m.\u001b[39mappend(train_losses)\n\u001b[0;32m     46\u001b[0m \u001b[38;5;66;03m# Store the trained model\u001b[39;00m\n",
      "Cell \u001b[1;32mIn[86], line 32\u001b[0m, in \u001b[0;36mtrain_model\u001b[1;34m(model, train_loader, optimizer, criterion, epochs, device)\u001b[0m\n\u001b[0;32m     29\u001b[0m optimizer\u001b[38;5;241m.\u001b[39mzero_grad()  \u001b[38;5;66;03m# Clear gradients for the next train step\u001b[39;00m\n\u001b[0;32m     30\u001b[0m output \u001b[38;5;241m=\u001b[39m model(X, A)  \u001b[38;5;66;03m# Forward pass\u001b[39;00m\n\u001b[1;32m---> 32\u001b[0m loss \u001b[38;5;241m=\u001b[39m \u001b[43mcriterion\u001b[49m\u001b[43m(\u001b[49m\u001b[43moutput\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mlabels\u001b[49m\u001b[43m)\u001b[49m  \u001b[38;5;66;03m# Compute the loss\u001b[39;00m\n\u001b[0;32m     33\u001b[0m loss\u001b[38;5;241m.\u001b[39mbackward()  \u001b[38;5;66;03m# Backward pass to compute gradients\u001b[39;00m\n\u001b[0;32m     34\u001b[0m optimizer\u001b[38;5;241m.\u001b[39mstep()  \u001b[38;5;66;03m# Update model parameters\u001b[39;00m\n",
      "File \u001b[1;32mc:\\Users\\linyi\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\torch\\nn\\modules\\module.py:1511\u001b[0m, in \u001b[0;36mModule._wrapped_call_impl\u001b[1;34m(self, *args, **kwargs)\u001b[0m\n\u001b[0;32m   1509\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_compiled_call_impl(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)  \u001b[38;5;66;03m# type: ignore[misc]\u001b[39;00m\n\u001b[0;32m   1510\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m-> 1511\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_call_impl\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43margs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mkwargs\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[1;32mc:\\Users\\linyi\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\torch\\nn\\modules\\module.py:1520\u001b[0m, in \u001b[0;36mModule._call_impl\u001b[1;34m(self, *args, **kwargs)\u001b[0m\n\u001b[0;32m   1515\u001b[0m \u001b[38;5;66;03m# If we don't have any hooks, we want to skip the rest of the logic in\u001b[39;00m\n\u001b[0;32m   1516\u001b[0m \u001b[38;5;66;03m# this function, and just call forward.\u001b[39;00m\n\u001b[0;32m   1517\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m (\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_backward_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_backward_pre_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_forward_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_forward_pre_hooks\n\u001b[0;32m   1518\u001b[0m         \u001b[38;5;129;01mor\u001b[39;00m _global_backward_pre_hooks \u001b[38;5;129;01mor\u001b[39;00m _global_backward_hooks\n\u001b[0;32m   1519\u001b[0m         \u001b[38;5;129;01mor\u001b[39;00m _global_forward_hooks \u001b[38;5;129;01mor\u001b[39;00m _global_forward_pre_hooks):\n\u001b[1;32m-> 1520\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mforward_call\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43margs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mkwargs\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m   1522\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[0;32m   1523\u001b[0m     result \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mNone\u001b[39;00m\n",
      "File \u001b[1;32mc:\\Users\\linyi\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\torch\\nn\\modules\\loss.py:1179\u001b[0m, in \u001b[0;36mCrossEntropyLoss.forward\u001b[1;34m(self, input, target)\u001b[0m\n\u001b[0;32m   1178\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mforward\u001b[39m(\u001b[38;5;28mself\u001b[39m, \u001b[38;5;28minput\u001b[39m: Tensor, target: Tensor) \u001b[38;5;241m-\u001b[39m\u001b[38;5;241m>\u001b[39m Tensor:\n\u001b[1;32m-> 1179\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mF\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mcross_entropy\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;28;43minput\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mtarget\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mweight\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mweight\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1180\u001b[0m \u001b[43m                           \u001b[49m\u001b[43mignore_index\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mignore_index\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mreduction\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mreduction\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1181\u001b[0m \u001b[43m                           \u001b[49m\u001b[43mlabel_smoothing\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mlabel_smoothing\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[1;32mc:\\Users\\linyi\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\torch\\nn\\functional.py:3059\u001b[0m, in \u001b[0;36mcross_entropy\u001b[1;34m(input, target, weight, size_average, ignore_index, reduce, reduction, label_smoothing)\u001b[0m\n\u001b[0;32m   3057\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m size_average \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m \u001b[38;5;129;01mor\u001b[39;00m reduce \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m:\n\u001b[0;32m   3058\u001b[0m     reduction \u001b[38;5;241m=\u001b[39m _Reduction\u001b[38;5;241m.\u001b[39mlegacy_get_string(size_average, reduce)\n\u001b[1;32m-> 3059\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mtorch\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_C\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_nn\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mcross_entropy_loss\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;28;43minput\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mtarget\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mweight\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43m_Reduction\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mget_enum\u001b[49m\u001b[43m(\u001b[49m\u001b[43mreduction\u001b[49m\u001b[43m)\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mignore_index\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mlabel_smoothing\u001b[49m\u001b[43m)\u001b[49m\n",
      "\u001b[1;31mValueError\u001b[0m: Expected input batch_size (9) to match target batch_size (50)."
     ]
    }
   ],
   "source": [
    "def main():\n",
    "    \"\"\"\n",
    "    Main execution function to train and evaluate Graph Convolutional Network (GCN) models\n",
    "    with different graph normalization techniques, visualize training metrics, and perform\n",
    "    embedding analysis through PCA.\n",
    "\n",
    "    Assumes the presence of a GCN model class, data loader preparation functions, and\n",
    "    various normalization technique functions defined outside this script.\n",
    "    \"\"\"\n",
    "    # Configuration parameters\n",
    "    num_samples_per_type = 1000  # Number of samples per class/type\n",
    "    num_epochs = 200  # Number of training epochs\n",
    "    # Dictionary mapping normalization technique names to their corresponding functions\n",
    "    normalization_techniques = {\n",
    "        'degree': compute_degree_matrix_normalization,\n",
    "        'pagerank': compute_pagerank_normalization,\n",
    "        'betweenness': compute_betweenness_normalization,\n",
    "        'clustering': compute_clustering_coefficient_normalization,\n",
    "    }\n",
    "\n",
    "    # Lists for storing evaluation metrics and model information\n",
    "    metric_values = [[] for _ in range(4)]  # Lists to store Accuracy, Precision, Recall, F1 Score\n",
    "    normalization_names = []  # Names of the normalization techniques\n",
    "    train_losses_list = []  # Training loss values for each normalization technique\n",
    "    models = []  # Trained models\n",
    "\n",
    "    # Set the computation device (GPU or CPU)\n",
    "    device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "\n",
    "    # Loop over each normalization technique to train and evaluate a model\n",
    "    for name, norm_func in normalization_techniques.items():\n",
    "        print(f\"\\nTraining model with {name} normalization...\")\n",
    "        # Prepare data loaders\n",
    "        train_loader, test_loader = prepare_data_loaders(num_samples_per_type, normalization_func=norm_func, batch_size=50)\n",
    "        print(f\"DataLoader batch size: {train_loader.batch_size}\")\n",
    "\n",
    "        # Initialize the GCN model, optimizer, and loss criterion\n",
    "        model = GCN(nfeat=3, nhid=16, nclass=2).to(device)\n",
    "        optimizer = optim.Adam(model.parameters(), lr=0.001)\n",
    "        criterion = nn.CrossEntropyLoss()\n",
    "\n",
    "        # Train the model\n",
    "        train_losses = train_model(model, train_loader, optimizer, criterion, epochs=num_epochs, device=device)\n",
    "        train_losses_list.append(train_losses)\n",
    "\n",
    "        # Store the trained model\n",
    "        models.append(model)\n",
    "\n",
    "        # Evaluate the model's performance\n",
    "        accuracy, precision, recall, f1 = evaluate_model(model, test_loader, device=device)\n",
    "        # Store the evaluation metrics\n",
    "        metric_values[0].append(accuracy)\n",
    "        metric_values[1].append(precision)\n",
    "        metric_values[2].append(recall)\n",
    "        metric_values[3].append(f1)\n",
    "        normalization_names.append(name)\n",
    "\n",
    "        # Output the evaluation results\n",
    "        print(f\"Results with {name} normalization - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}\")\n",
    "\n",
    "    # Visualization of training losses and evaluation metrics for each normalization technique\n",
    "    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']\n",
    "    plot_training_losses(train_losses_list, normalization_names)\n",
    "    plot_metric_bar_charts(normalization_names, metric_values, metric_names)\n",
    "\n",
    "    # Embedding extraction and PCA visualization\n",
    "    embeddings_list = []\n",
    "    labels_list = []  # Labels for each set of embeddings\n",
    "    titles = []  # Titles for the PCA plots\n",
    "\n",
    "    # Extract embeddings and labels for each model\n",
    "    for name, model in zip(normalization_names, models):\n",
    "        print(f\"\\nExtracting embeddings for model trained with {name} normalization...\")\n",
    "        embeddings, labels = extract_embeddings(model, test_loader, device=device)\n",
    "        embeddings_list.append(embeddings)\n",
    "        labels_list.append(labels)  # Append corresponding labels\n",
    "        titles.append(f\"Embedding Distributions with {name} normalization\")\n",
    "\n",
    "    # Apply PCA and visualize the embeddings\n",
    "    apply_pca_and_visualize_all_sns(embeddings_list, titles, labels_list)\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    main()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "Xg8ZtRfK463d",
   "metadata": {
    "id": "Xg8ZtRfK463d"
   },
   "source": [
    "# 2) Implementation of GraphSAGE with Node Sampling (30 points)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cn37bADeLk0h",
   "metadata": {
    "id": "cn37bADeLk0h"
   },
   "source": [
       
       
    "## 2.1) Dataset\n",
    "\n",
    "In this question, we are going to train and test GraphSAGE on a **node classification** task using a toy Protein-Protein Interaction (PPI) dataset.\n",
    "\n",
    "The dataset contains 24 graphs. The average number of nodes per graph is 2372. Each node has 50 features and 121 labels.\n",
    "\n",
    "Since we will work on a node classification task, we will select only one of the graphs.\n",
    "\n",
    "Below, we load the dataset and split into train/validation/test splits."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "id": "pKOlr5XwLoaw",
   "metadata": {
    "id": "pKOlr5XwLoaw"
   },
   "outputs": [],
   "source": [
    "def load_ppi_data():\n",
    "    # Load the dataset\n",
    "    dataset = PPIDataset()\n",
    "\n",
    "    # Select one graph from the PPI dataset\n",
    "    g = dataset[0]\n",
    "\n",
    "    # Extract features, labels\n",
    "    features = g.ndata['feat']\n",
    "    labels = g.ndata['label']\n",
    "\n",
    "    num_nodes = g.number_of_nodes()\n",
    "    num_train = int(0.6 * num_nodes)  # 60% for training\n",
    "    num_val = int(0.2 * num_nodes)    # 20% for validation\n",
    "\n",
    "    # Create a random permutation of node indices\n",
    "    indices = torch.randperm(num_nodes)\n",
    "\n",
    "    # Assign the first num_train nodes to the training set\n",
    "    # Assign the next num_val nodes to the validation set\n",
    "    # Assign the remaining nodes to the test set\n",
    "    train_mask = torch.zeros(num_nodes, dtype=torch.bool)\n",
    "    val_mask = torch.zeros(num_nodes, dtype=torch.bool)\n",
    "    test_mask = torch.zeros(num_nodes, dtype=torch.bool)\n",
    "\n",
    "    train_mask[indices[:num_train]] = True\n",
    "    val_mask[indices[num_train:num_train+num_val]] = True\n",
    "    test_mask[indices[num_train+num_val:]] = True\n",
    "\n",
    "    adj = g.adjacency_matrix().to_dense()\n",
    "\n",
    "    return features, labels, adj, train_mask, val_mask, test_mask"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 98,
   "id": "Rdp7_8kzLpZJ",
   "metadata": {
    "id": "Rdp7_8kzLpZJ"
   },
   "outputs": [
    {
     "ename": "FileNotFoundError",
     "evalue": "Cannot find DGL C++ sparse library at c:\\Users\\linyi\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\dgl\\dgl_sparse\\dgl_sparse_pytorch_2.2.0.dll",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mFileNotFoundError\u001b[0m                         Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[98], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m features, labels, adj, train_mask, val_mask, test_mask \u001b[38;5;241m=\u001b[39m \u001b[43mload_ppi_data\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m      3\u001b[0m features \u001b[38;5;241m=\u001b[39m features\u001b[38;5;241m.\u001b[39mto(device)\n\u001b[0;32m      4\u001b[0m labels \u001b[38;5;241m=\u001b[39m labels\u001b[38;5;241m.\u001b[39mto(device)\n",
      "Cell \u001b[1;32mIn[94], line 30\u001b[0m, in \u001b[0;36mload_ppi_data\u001b[1;34m()\u001b[0m\n\u001b[0;32m     27\u001b[0m val_mask[indices[num_train:num_train\u001b[38;5;241m+\u001b[39mnum_val]] \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mTrue\u001b[39;00m\n\u001b[0;32m     28\u001b[0m test_mask[indices[num_train\u001b[38;5;241m+\u001b[39mnum_val:]] \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mTrue\u001b[39;00m\n\u001b[1;32m---> 30\u001b[0m adj \u001b[38;5;241m=\u001b[39m \u001b[43mg\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43madjacency_matrix\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\u001b[38;5;241m.\u001b[39mto_dense()\n\u001b[0;32m     32\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m features, labels, adj, train_mask, val_mask, test_mask\n",
      "File \u001b[1;32mc:\\Users\\linyi\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\dgl\\heterograph.py:3761\u001b[0m, in \u001b[0;36mDGLGraph.adjacency_matrix\u001b[1;34m(self, etype)\u001b[0m\n\u001b[0;32m   3759\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21madjacency_matrix\u001b[39m(\u001b[38;5;28mself\u001b[39m, etype\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mNone\u001b[39;00m):\n\u001b[0;32m   3760\u001b[0m \u001b[38;5;250m    \u001b[39m\u001b[38;5;124;03m\"\"\"Alias of :meth:`adj`\"\"\"\u001b[39;00m\n\u001b[1;32m-> 3761\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43madj\u001b[49m\u001b[43m(\u001b[49m\u001b[43metype\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[1;32mc:\\Users\\linyi\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\dgl\\heterograph.py:3823\u001b[0m, in \u001b[0;36mDGLGraph.adj\u001b[1;34m(self, etype, eweight_name)\u001b[0m\n\u001b[0;32m   3820\u001b[0m \u001b[38;5;66;03m# Temporal fix to introduce a dependency on torch\u001b[39;00m\n\u001b[0;32m   3821\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mtorch\u001b[39;00m\n\u001b[1;32m-> 3823\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01msparse\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m spmatrix\n\u001b[0;32m   3825\u001b[0m etype \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mto_canonical_etype(etype)\n\u001b[0;32m   3826\u001b[0m indices \u001b[38;5;241m=\u001b[39m torch\u001b[38;5;241m.\u001b[39mstack(\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mall_edges(etype\u001b[38;5;241m=\u001b[39metype))\n",
      "File \u001b[1;32mc:\\Users\\linyi\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\dgl\\sparse\\__init__.py:43\u001b[0m\n\u001b[0;32m     39\u001b[0m     \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mException\u001b[39;00m:  \u001b[38;5;66;03m# pylint: disable=W0703\u001b[39;00m\n\u001b[0;32m     40\u001b[0m         \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mImportError\u001b[39;00m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mCannot load DGL C++ sparse library\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[1;32m---> 43\u001b[0m \u001b[43mload_dgl_sparse\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[1;32mc:\\Users\\linyi\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\dgl\\sparse\\__init__.py:35\u001b[0m, in \u001b[0;36mload_dgl_sparse\u001b[1;34m()\u001b[0m\n\u001b[0;32m     33\u001b[0m path \u001b[38;5;241m=\u001b[39m os\u001b[38;5;241m.\u001b[39mpath\u001b[38;5;241m.\u001b[39mjoin(dirname, \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mdgl_sparse\u001b[39m\u001b[38;5;124m\"\u001b[39m, basename)\n\u001b[0;32m     34\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m os\u001b[38;5;241m.\u001b[39mpath\u001b[38;5;241m.\u001b[39mexists(path):\n\u001b[1;32m---> 35\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mFileNotFoundError\u001b[39;00m(\u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mCannot find DGL C++ sparse library at \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mpath\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m\"\u001b[39m)\n\u001b[0;32m     37\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[0;32m     38\u001b[0m     torch\u001b[38;5;241m.\u001b[39mclasses\u001b[38;5;241m.\u001b[39mload_library(path)\n",
      "\u001b[1;31mFileNotFoundError\u001b[0m: Cannot find DGL C++ sparse library at c:\\Users\\linyi\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\dgl\\dgl_sparse\\dgl_sparse_pytorch_2.2.0.dll"
     ]
    }
   ],
   "source": [
    "features, labels, adj, train_mask, val_mask, test_mask = load_ppi_data()\n",
    "\n",
    "features = features.to(device)\n",
    "labels = labels.to(device)\n",
    "adj = adj.to(device)\n",
    "\n",
    "num_feats = features.shape[1]\n",
    "num_classes = labels.shape[1]\n",
    "\n",
    "# Convert one-hot encoding to class indices format\n",
    "# e.g.,\n",
    "# one-hot encoding vector [0, 0, 1, 0, 0, ....] is converted to class index 2.\n",
    "# To use torch.nn.CrossEntropyLoss, we need to have this class indices format. \n",
    "# For more information, check https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html\n",
    "labels = torch.argmax(labels, dim=1)\n",
    "\n",
    "train_features = features[train_mask]\n",
    "val_features = features[val_mask]\n",
    "test_features = features[test_mask]\n",
    "\n",
    "train_labels = labels[train_mask]\n",
    "val_labels = labels[val_mask]\n",
    "test_labels = labels[test_mask]\n",
    "\n",
    "train_adj = adj[train_mask][:, train_mask]\n",
    "val_adj = adj[val_mask][:, val_mask]\n",
    "test_adj = adj[test_mask][:, test_mask]\n",
    "\n",
    "print(f\"Number of train nodes: {train_adj.shape[0]}\")\n",
    "print(f\"Number of val nodes: {val_adj.shape[0]}\")\n",
    "print(f\"Number of test nodes: {test_adj.shape[0]}\")\n",
    "print(f\"Number of features: {num_feats}\")\n",
    "print(f\"Number of classes: {num_classes}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "FCGjLT2KfEPF",
   "metadata": {
    "id": "FCGjLT2KfEPF"
   },
   "outputs": [],
   "source": [
    "def compute_average_degree(A):\n",
    "    degrees = A.sum(dim=1)  # Sum along rows to get degrees\n",
    "    average_degree = degrees.mean().item()  # Compute the mean degree\n",
    "\n",
    "    return average_degree\n",
    "\n",
    "average_degree = compute_average_degree(train_adj)\n",
    "print(\"Average Degree:\", average_degree)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "MiyIOQSXMgqB",
   "metadata": {
    "id": "MiyIOQSXMgqB"
   },
   "source": [
    "## 2.2) Node-wise Sampling (15 points)\n",
    "Node-wise sampling involves aggregating a subset of neighbors for each node in the graph, as opposed to considering all neighbors for aggregation (see the Figure 1). \n",
    "\n",
    "In the GraphSAGE, they sample a fixed number of neighbors in each layer. More specifically, they use $K=2$ number of layers and for the first layer and second layer, they sample $S_1=25$ and $S_2=10$ neighbors, respectively. \n",
    "\n",
    "**Here, for simplicity, we will sample $S=S_1=S_2=5$ neighbors for both layers**.\n",
    "\n",
    "<img src=\"figures/nodewise_sampling.jpg\" alt=\"Node-wise sampling\" width=\"200\" />\n",
    "\n",
    "(Figure 1: Node-wise sampling<sup>1</sup>)\n",
    "\n",
    "<sup>1</sup>_Hamilton, W., Ying, Z., & Leskovec, J. (2017). Inductive representation learning on large graphs. Advances in neural information processing systems, 30._\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "Zdm91LJaOMp3",
   "metadata": {
    "id": "Zdm91LJaOMp3"
   },
   "source": [
    "Below, you need to implement the sampler function. It takes the adj. matrix A and number of neighbors to sample, returns a list of lists including indices to sampled neighbors for each node in A.\n",
    "\n",
    "You can use [torch.randperm](https://pytorch.org/docs/stable/generated/torch.randperm.html) or feel free to use any other function that does the same job."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "OA2mQO3GOliw",
   "metadata": {
    "id": "OA2mQO3GOliw"
   },
   "outputs": [],
   "source": [
    "def sampler(A, num_samples):\n",
    "    \"\"\"\n",
    "    Samples \"num_samples\" amount of neighbors for each node in adj. matrix A\n",
    "    You can use uniform random sampling. No need for any importance sampling strategy.\n",
    "\n",
    "    Params:\n",
    "        A (Tensor): Adj. matrix of shape (N x N)\n",
    "        num_samples (int): Number of neighbors to sample for each node\n",
    "        where N is the number of nodes.\n",
    "\n",
    "    Returns:\n",
    "        A list of lists including indices to sampled neighbors for each node in A.\n",
    "    \"\"\"\n",
    "\n",
    "    N = A.shape[0]  # Number of nodes\n",
    "    sampled_neighbors = []\n",
    "\n",
    "    ########## YOUR CODE HERE ##########\n",
    "    # For each node, populate the sampled_neighors list\n",
    "    # Here is a dummy example with num_samples = 3\n",
    "    # sampled_neighbors = [[3, 41, 2], [53, 234], ...]\n",
    "    # for the first node, the indices for sampled neighbors are 3, 41 and 2.\n",
    "    # the second node has only 2 neighbors (smaller than the num_samples), thus we sampled all its neighbors.\n",
    "    ####################################\n",
    "    \n",
    "    for i in range(N):\n",
    "        # Find indices of all neighbors (non-zero entries in adjacency matrix row)\n",
    "        neighbors = torch.nonzero(A[i], as_tuple=False).squeeze().tolist()\n",
    "        # If neighbors is a scalar (only one neighbor), wrap it in a list\n",
    "        if isinstance(neighbors, int):\n",
    "            neighbors = [neighbors]\n",
    "        # Sample min(num_samples, len(neighbors)) neighbors uniformly without replacement\n",
    "        if len(neighbors) > num_samples:\n",
    "            sampled = random.sample(neighbors, num_samples)\n",
    "        else:\n",
    "            sampled = neighbors\n",
    "        sampled_neighbors.append(sampled)\n",
    "\n",
    "    return sampled_neighbors"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "HjtPeehpQIVP",
   "metadata": {
    "id": "HjtPeehpQIVP"
   },
   "source": [
    "Let's see the sampled neighbors for the first 10 nodes."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "YfGkkONoPFo3",
   "metadata": {
    "id": "YfGkkONoPFo3"
   },
   "outputs": [],
   "source": [
    "# set num_samples to 2 just for now\n",
    "num_samples = 2\n",
    "\n",
    "sampled_neighbors = sampler(adj[:10], num_samples)\n",
    "\n",
    "# Print the sampled neighbors for each node\n",
    "for node, neighbors in enumerate(sampled_neighbors):\n",
    "    print(f\"Node {node}: Sampled Neighbors {neighbors}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "It_a42RolM_h",
   "metadata": {
    "id": "It_a42RolM_h"
   },
   "source": [
    "## 2.3) Implementation of GraphSAGE (15 points)\n",
    "\n",
    "In Figure 2, you can find the pseudo-code for the forward propagation of GraphSAGE. Basically, in each layer, GraphSAGE iterates over all the nodes in the graph and aggregates the neighborhood information from a set of sampled neighbors. Then, different from the original GCN model<sup>2</sup>, the embedding of the current node is concatenated with the aggregated embedding, doubling the size of the embedding vector before applying linear transformation via the learnable parameter $W^k$. After that, the embedding of the current node is updated following the application of a non-linearity.\n",
    "\n",
    "They use different AGGREGATE functions such as mean and max-pooling aggregation. In this question, you need to use mean aggregation.\n",
    "\n",
    "We choose the number of layers as $K=2$ and non-linearity as $ReLU$.\n",
    "\n",
    "It is OK to skip the normalization in the line 7.\n",
    "\n",
    "<img src=\"figures/graphsage_algo.jpg\" alt=\"GraphSAGE Algorithm\" width=\"700\" />\n",
    "\n",
    "(Figure 2: Forward propagation of GraphSAGE<sup>3</sup>)\n",
    "\n",
    "<sup>2</sup>*Kipf, T. N., & Welling, M. (2016). Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907.*\n",
    "\n",
    "<sup>3</sup>*Liu, X., Yan, M., Deng, L., Li, G., Ye, X., & Fan, D. (2021). Sampling methods for efficient training of graph convolutional networks: A survey. IEEE/CAA Journal of Automatica Sinica, 9(2), 205-234.*\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "kcyM2ioCR939",
   "metadata": {
    "id": "kcyM2ioCR939"
   },
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'nn' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[2], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m \u001b[38;5;28;01mclass\u001b[39;00m \u001b[38;5;21;01mGraphSAGEConvLayer\u001b[39;00m(\u001b[43mnn\u001b[49m\u001b[38;5;241m.\u001b[39mModule):\n\u001b[0;32m      2\u001b[0m     \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21m__init__\u001b[39m(\u001b[38;5;28mself\u001b[39m, input_dim, output_dim):\n\u001b[0;32m      3\u001b[0m         \u001b[38;5;28msuper\u001b[39m(GraphSAGEConvLayer, \u001b[38;5;28mself\u001b[39m)\u001b[38;5;241m.\u001b[39m\u001b[38;5;21m__init__\u001b[39m()\n",
      "\u001b[1;31mNameError\u001b[0m: name 'nn' is not defined"
     ]
    }
   ],
   "source": [
    "class GraphSAGEConvLayer(nn.Module):\n",
    "    def __init__(self, input_dim, output_dim):\n",
    "        super(GraphSAGEConvLayer, self).__init__()\n",
    "        ########## YOUR CODE HERE ##########\n",
    "        # Define the learnable parameter W to be used in linear transformation\n",
    "        # Take into account the dimension increase resulting from the concatenation\n",
    "        # of the aggregated neighbor embeddings with the current node embedding.\n",
    "        #\n",
    "        # self.W = ...\n",
    "        ####################################\n",
    "        self.W = nn.Parameter(torch.FloatTensor(2 *input_dim, output_dim))\n",
    "        nn.init.xavier_uniform_(self.W.data)\n",
    "\n",
    "    def forward(self, curr_node_emb, neighbor_embs):\n",
    "        \"\"\"\n",
    "        Forward pass of a single GraphSAGE convolution layer\n",
    "\n",
    "        Params:\n",
    "        curr_node_emb (Tensor): Embedding vector of the current node.\n",
    "        neighbor_embs (Tensor): Embedding vectors of sampled neighbors of the current node\n",
    "\n",
    "        Returns:\n",
    "        Tensor: New embedding of the current node\n",
    "        \"\"\"\n",
    "\n",
    "        ########## YOUR CODE HERE ##########\n",
    "        # 1. Aggregate neighbor embeddings using mean aggregation\n",
    "        # 2. Concatenate the aggregated embeddings with the embedding of the current node\n",
    "        # 3. Apply linear transformation using self.W\n",
    "        # 4. Apply ReLU non-linearity\n",
    "        # 5. Return the new_embedding\n",
    "        ####################################\n",
    "        if neighbor_embs.dim() == 1:\n",
    "            neighbor_embs = neighbor_embs.unsqueeze(0)\n",
    "        \n",
    "        neighbor_embs = torch.mean(neighbor_embs, dim=0, keepdim=True)\n",
    "        concat_embs = torch.cat([curr_node_emb, neighbor_embs], dim=1)\n",
    "        linear = torch.matmul(concat_embs, self.W)\n",
    "        embeddings = F.relu(linear)\n",
    "    \n",
    "        return embeddings\n",
    "        \n",
    "\n",
    "\n",
    "class GraphSAGE(nn.Module):\n",
    "    def __init__(self, input_dim, hidden_dim, num_classes, num_samples):\n",
    "        super(GraphSAGE, self).__init__()\n",
    "        self.layers = nn.ModuleList([\n",
    "            GraphSAGEConvLayer(input_dim, hidden_dim),\n",
    "            GraphSAGEConvLayer(hidden_dim, num_classes)\n",
    "        ])\n",
    "\n",
    "    def forward(self, X, A):\n",
    "        \"\"\"\n",
    "        Forward pass of GraphSAGE\n",
    "\n",
    "        Params:\n",
    "        X (Tensor): Node feature matrix of shape (N x d)\n",
    "        A (Tensor): Adj. matrix of shape (N x N)\n",
    "        where N is the number of nodes and d is the embedding size.\n",
    "\n",
    "        Returns:\n",
    "        Tensor: The output matrix of the last layer with shape (N x num_classes)\n",
    "        \"\"\"\n",
    "\n",
    "        ########## YOUR CODE HERE ##########\n",
    "        # 1. For each layer:\n",
    "        #   1.1. Sample neighbors using the sampler function and adj matrix A\n",
    "        #   1.2. Update the embedding for each node\n",
    "        #     1.2.1 Forward pass through the GraphSAGE convolution layer\n",
    "        #     1.2.2 Store the new embeddings for each node\n",
    "        #   1.3. Update the node feature matrix for the next layer\n",
    "        # 2. Return the final node feature matrix with shape (N x num_classes)\n",
    "        ####################################\n",
    "        node_embs = X\n",
    "        for layer in self.layers:\n",
    "            # Sample neighbors using the sampler function and adj matrix A\n",
    "            # (Note: The sampler function should be defined as per your previous instructions)\n",
    "            sampled_neighbors_indices = sampler(A, num_samples=5)\n",
    "            # Create a list to store the new embeddings for each node\n",
    "            new_embeddings = []\n",
    "            for i, neighbors in enumerate(sampled_neighbors_indices):\n",
    "                # Get the embeddings for the current node and its sampled neighbors\n",
    "                curr_node_emb = node_embs[i].unsqueeze(0)  # Add singleton dimension for concatenation\n",
    "                neighbor_embs = node_embs[neighbors]\n",
    "                # Forward pass through the GraphSAGE convolution layer\n",
    "                new_embedding = layer(curr_node_emb, neighbor_embs)\n",
    "                new_embeddings.append(new_embedding)\n",
    "            # Update the node feature matrix for the next layer\n",
    "            node_embs = torch.cat(new_embeddings, dim=0)\n",
    "        # The final node feature matrix has shape (N x num_classes)\n",
    "        return node_embs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "upLxt7pwWgXn",
   "metadata": {
    "id": "upLxt7pwWgXn"
   },
   "outputs": [],
   "source": [
    "def trainStepGraphSAGE(model, features, adj, labels, loss_fn, optimizer):\n",
    "    model.train()\n",
    "\n",
    "    # Forward pass\n",
    "    logits = model(features, adj)\n",
    "    train_loss = loss_fn(logits, labels)\n",
    "\n",
    "    # Backward pass and optimization\n",
    "    optimizer.zero_grad()\n",
    "    train_loss.backward()\n",
    "    optimizer.step()\n",
    "    \n",
    "    return train_loss.item()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "Fjxyx9j6Wl8m",
   "metadata": {
    "id": "Fjxyx9j6Wl8m"
   },
   "outputs": [],
   "source": [
    "@torch.no_grad()\n",
    "def testGraphSAGE(model, features, adj, labels):\n",
    "    model.eval()\n",
    "    logits = model(features, adj)\n",
    "\n",
    "    _, predicted = torch.max(logits, 1)\n",
    "\n",
    "    correct = (predicted == labels).sum().item()\n",
    "    total = labels.size(0)\n",
    "    accuracy = correct / total\n",
    "\n",
    "    return accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "woYruDm-Wbrn",
   "metadata": {
    "id": "woYruDm-Wbrn"
   },
   "outputs": [],
   "source": [
    "def trainGraphSAGE(model, train_features, val_features, train_adj, val_adj, train_labels, val_labels, num_epochs=50):\n",
    "    t = time.time()\n",
    "    loss_fn = nn.CrossEntropyLoss()\n",
    "    optimizer = optim.AdamW(model.parameters(), lr=0.01)\n",
    "\n",
    "    best_model = None\n",
    "    best_valid_acc = 0\n",
    "\n",
    "    for epoch in range(1, 1 + num_epochs):\n",
    "        train_loss = trainStepGraphSAGE(model, train_features, train_adj, train_labels, loss_fn, optimizer)\n",
    "        val_acc = testGraphSAGE(model, val_features, val_adj, val_labels)\n",
    "        if val_acc > best_valid_acc:\n",
    "            best_valid_acc = val_acc\n",
    "            best_model = copy.deepcopy(model)\n",
    "        \n",
    "        print(f'Epoch: {epoch:02d}, '\n",
    "            f'Train Loss: {train_loss:.4f}, ',\n",
    "            f'Validation Accuracy: {100*val_acc:.4f}%, ',\n",
    "            f'time: {time.time() - t:.4f}s.')\n",
    "    \n",
    "    print(f'best acc_valid: {100*best_valid_acc:.4f}%')\n",
    "\n",
    "    return best_model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "gxNlHZg0XAEE",
   "metadata": {
    "id": "gxNlHZg0XAEE"
   },
   "outputs": [],
   "source": [
    "# feel free to play with hidden_dim :)\n",
    "hidden_dim = 64\n",
    "model = GraphSAGE(num_feats, hidden_dim, num_classes, num_samples=5)\n",
    "\n",
    "model.to(device)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "nwJWPpEVaJo8",
   "metadata": {
    "id": "nwJWPpEVaJo8"
   },
   "outputs": [],
   "source": [
    "best_model = trainGraphSAGE(model, train_features, val_features, train_adj, val_adj, train_labels, val_labels, num_epochs=50)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d97776f1",
   "metadata": {},
   "outputs": [],
   "source": [
    "# You should get around 70-75% test accuracy.\n",
    "test_acc = testGraphSAGE(best_model, test_features, test_adj, test_labels)\n",
    "print(f\"Test Accuracy: {100*test_acc:.4f}%\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6t4V24ZpjXJx",
   "metadata": {
    "id": "6t4V24ZpjXJx"
   },
   "source": [
    "# 3) Attention-based aggregation in node classification (30 points)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0dSBj13VcNy0",
   "metadata": {
    "id": "0dSBj13VcNy0"
   },
   "source": [
    "The objective is to develop two types of aggregation methods: mean aggregation and aggregation by attention.\n",
    "\n",
    "For those who require additional guidance or clarification, it is recommended to revisit the relevant [lecture](https://www.youtube.com/watch?v=zRmzVkidkqA&list=PLug43ldmRSo14Y_vt7S6vanPGh-JpHR7T&index=13) or consult the course [notes](https://drive.google.com/file/d/1p7U1xyW4-5W4ge8gRstUkBGQSY3fKHX6/view).\n",
    "\n",
    "\n",
    "Important Reminder: Please ensure that you execute all the cells in each section in sequence to maintain the integrity of intermediate variables and package imports.\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "g_df0zRghF1b",
   "metadata": {
    "id": "g_df0zRghF1b"
   },
   "source": [
    "## Constructing Layers for Graph Neural Networks (17 points)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "neppPZh5g_47",
   "metadata": {
    "id": "neppPZh5g_47"
   },
   "source": [
    "Let's begin by creating a dummy dataset that will aid in the development and testing of our Graph Neural Networks (GNNs). This dataset will include a simple graph structure with defined nodes, edges, and node features."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "jju0azWFhevg",
   "metadata": {
    "id": "jju0azWFhevg"
   },
   "outputs": [],
   "source": [
    "def create_dummy_data(n_nodes, n_features):\n",
    "  # Create a random adjacency matrix for an undirected graph\n",
    "  # Use a random integer matrix and make it symmetric\n",
    "  adj = torch.triu(torch.randint(0, 2, (n_nodes, n_nodes)), diagonal=1)\n",
    "  adj = adj + adj.T\n",
    "\n",
    "  # Create random features for each node\n",
    "  adj = adj.type(torch.float)\n",
    "  x = torch.rand(n_nodes, n_features)\n",
    "\n",
    "  return x, adj"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "7qDSC9VthjlE",
   "metadata": {
    "id": "7qDSC9VthjlE"
   },
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'torch' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[4], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m x, adj \u001b[38;5;241m=\u001b[39m \u001b[43mcreate_dummy_data\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m5\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m3\u001b[39;49m\u001b[43m)\u001b[49m\n\u001b[0;32m      2\u001b[0m \u001b[38;5;28mprint\u001b[39m(x\u001b[38;5;241m.\u001b[39mshape)\n\u001b[0;32m      3\u001b[0m \u001b[38;5;28mprint\u001b[39m(adj\u001b[38;5;241m.\u001b[39mshape)\n",
      "Cell \u001b[1;32mIn[3], line 4\u001b[0m, in \u001b[0;36mcreate_dummy_data\u001b[1;34m(n_nodes, n_features)\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mcreate_dummy_data\u001b[39m(n_nodes, n_features):\n\u001b[0;32m      2\u001b[0m   \u001b[38;5;66;03m# Create a random adjacency matrix for an undirected graph\u001b[39;00m\n\u001b[0;32m      3\u001b[0m   \u001b[38;5;66;03m# Use a random integer matrix and make it symmetric\u001b[39;00m\n\u001b[1;32m----> 4\u001b[0m   adj \u001b[38;5;241m=\u001b[39m \u001b[43mtorch\u001b[49m\u001b[38;5;241m.\u001b[39mtriu(torch\u001b[38;5;241m.\u001b[39mrandint(\u001b[38;5;241m0\u001b[39m, \u001b[38;5;241m2\u001b[39m, (n_nodes, n_nodes)), diagonal\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m1\u001b[39m)\n\u001b[0;32m      5\u001b[0m   adj \u001b[38;5;241m=\u001b[39m adj \u001b[38;5;241m+\u001b[39m adj\u001b[38;5;241m.\u001b[39mT\n\u001b[0;32m      7\u001b[0m   \u001b[38;5;66;03m# Create random features for each node\u001b[39;00m\n",
      "\u001b[1;31mNameError\u001b[0m: name 'torch' is not defined"
     ]
    }
   ],
   "source": [
    "x, adj = create_dummy_data(5, 3)\n",
    "print(x.shape)\n",
    "print(adj.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "djXNL49_hqyt",
   "metadata": {
    "id": "djXNL49_hqyt"
   },
   "source": [
    "### Building a Graph Neural Network with Mean Aggregation (7 points)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e_NRLPPhhs_v",
   "metadata": {
    "id": "e_NRLPPhhs_v"
   },
   "source": [
    "Observe that the GNN utilizing a mean aggregator is significantly influenced by how the adjacency matrix is normalized.\n",
    "\n",
    "The formula for the next layer's node representations is given by:\n",
    "\n",
    "$H_{k+1} = a[\\beta_k\\mathbf{1}^T + \\Omega_kH_k(AD^{-1}+I)]$\n",
    "\n",
    "We will proceed to implement the mean normalization of the adjacency matrix using the following expression:\n",
    "\n",
    "$\\widetilde{A}=AD^{-1}+I$\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "DqOma83HjBth",
   "metadata": {
    "id": "DqOma83HjBth"
   },
   "outputs": [],
   "source": [
    "def mean_normalization(A):\n",
    "  ############# Your code here ############\n",
    "  ## Note:\n",
    "  ## 1. Calculate the degree matrix\n",
    "  ## 2. Create the inverse of the degree matrix\n",
    "  ## 3. Compute the mean normalization of the adjacency matrix\n",
    "  ## (~3 lines of code)\n",
    "\n",
    "  #########################################\n",
    "  degree_matrix = torch.sum(A, dim=1)\n",
    "  degree_matrix_inv = 1. / degree_matrix\n",
    "  degree_matrix_inv[torch.isinf(degree_matrix_inv)] = 0\n",
    "  normalized = A * degree_matrix_inv\n",
    "  \n",
    "  return normalized"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "wSLQfIfjjChw",
   "metadata": {
    "id": "wSLQfIfjjChw"
   },
   "outputs": [],
   "source": [
    "## Test your implementation to observe the behavior of the\n",
    "## mean normalization adjacency matrix.\n",
    "## Note:\n",
    "## It should reflect the values normalized over the number of neighbors,\n",
    "## with the inclusion of a self-loop for each node\n",
    "mean_normalization(adj)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "gpq54RcAlyHm",
   "metadata": {
    "id": "gpq54RcAlyHm"
   },
   "source": [
    "Now, let's build the GCN layer with a mean aggregator. Remeber, node features are computed as follows:\n",
    "\n",
    "$H_{k+1} = a[\\beta_k\\mathbf{1}^T + \\Omega_kH_k(AD^{-1}+I)]$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "D2gWjKtDmCUW",
   "metadata": {
    "id": "D2gWjKtDmCUW"
   },
   "outputs": [],
   "source": [
    "class GCN(nn.Module):\n",
    "    \"\"\"\n",
    "    A basic implementation of GCN layer.\n",
    "    It aggregates information from a node's neighbors\n",
    "    using mean aggregation.\n",
    "    \"\"\"\n",
    "    def __init__(self, in_features, out_features, activation=None):\n",
    "        super(GCN, self).__init__()\n",
    "        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))\n",
    "        self.bias = nn.Parameter(torch.zeros(out_features))\n",
    "        self.activation = activation\n",
    "        self.reset_parameters()\n",
    "\n",
    "    def reset_parameters(self):\n",
    "        stdv = 1. / np.sqrt(self.weight.size(1))\n",
    "        self.weight.data.uniform_(-stdv, stdv)\n",
    "\n",
    "    def forward(self, x, adj):\n",
    "        \"\"\"\n",
    "        Forward pass of the GCN layer.\n",
    "\n",
    "        Parameters:\n",
    "        input (Tensor): The input features of the nodes.\n",
    "        adj (Tensor): The adjacency matrix of the graph.\n",
    "\n",
    "        Returns:\n",
    "        Tensor: The output features of the nodes after applying the GCN layer.\n",
    "        \"\"\"\n",
    "        adj_norm = mean_normalization(adj)\n",
    "        ############# Your code here ############\n",
    "        ## Note:\n",
    "        ## 1. Apply the linear transformation\n",
    "        ## 2. Perform the graph convolution operation\n",
    "        ## Note: rename the last line as `output`\n",
    "        ## (2 lines of code)\n",
    "\n",
    "        #########################################\n",
    "        h = self.activation(output) if self.activation else output\n",
    "        return h"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "YzuoVNnenV93",
   "metadata": {
    "id": "YzuoVNnenV93"
   },
   "outputs": [],
   "source": [
    "## Ensure that your implementation is flexible enough to accommodate changes\n",
    "## in the number of hidden features. Verify that the dimensions of all\n",
    "## matrices are correctly aligned, allowing for a successful forward\n",
    "## pass through the network.\n",
    "GCN(x.size(1), 3)(x, adj)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0g-hfZN1pHCo",
   "metadata": {
    "id": "0g-hfZN1pHCo"
   },
   "source": [
    "### Developing a Graph Neural Network with Attention-Based Aggregation (10 points)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "uEVWsPIYruLD",
   "metadata": {
    "id": "uEVWsPIYruLD"
   },
   "source": [
    "The transformed node embeddings $H_k^{'}$ are calculated using the formula:\n",
    "\n",
    "\\begin{equation}\n",
    "H_k^{'} = \\beta_k\\mathbf{1}^T + \\Omega_kH_k\n",
    "\\end{equation}\n",
    "\n",
    "In this equation, $\\beta_k$ and $\\Omega_k$ are parameters, and $H_k$ represents the node embeddings at layer $k$.\n",
    "\n",
    "To compute the similarity $s_{mn}$ between any two transformed node embeddings $h^{'}_m$ and $h^{'}_n$, we concatenate these embeddings and then take a dot product with a learned parameter vector $\\phi_k$. An activation function is then applied to this dot product:\n",
    "\n",
    "\\begin{equation}\n",
    "s_{mn} = a\\left[\\phi_k^T \\begin{bmatrix} h^{'}_m\\\\ h^{'}_n \\end{bmatrix}\\right]\n",
    "\\end{equation}\n",
    "\n",
    "These similarity values are organized into an $N \\times N$ matrix $S$, where each element represents the similarity between every pair of nodes.\n",
    "\n",
    "The attention weights that contribute to each output embedding are normalized to ensure they are positive and sum to one.\n",
    "\n",
    "This normalization is achieved using the softmax operation. However, it's important to note that only the values corresponding to a node and its neighbors are considered in this computation. The attention weights are then applied to the transformed embeddings as follows:\n",
    "\n",
    "\\begin{equation}\n",
    "a[H_k^{'} * \\text{Softmax}(S, A+I)]\n",
    "\\end{equation}\n",
    "\n",
    "Here, $A+I$ represents the adjacency matrix with added self-loops, ensuring that each node also considers itself when computing attention weights.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4nXvTZUvvN62",
   "metadata": {
    "id": "4nXvTZUvvN62"
   },
   "source": [
    "To get started, it's important to grasp how to calculate the similarity matrix.\n",
    "\n",
    "A straightforward approach would be to iterate through all the nodes and compute the similarity scores for each pair."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "wfJx30JZpGXp",
   "metadata": {
    "id": "wfJx30JZpGXp"
   },
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'x' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[5], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m N \u001b[38;5;241m=\u001b[39m \u001b[43mx\u001b[49m\u001b[38;5;241m.\u001b[39msize(\u001b[38;5;241m0\u001b[39m)\n\u001b[0;32m      2\u001b[0m D \u001b[38;5;241m=\u001b[39m \u001b[38;5;241m3\u001b[39m\n\u001b[0;32m      4\u001b[0m \u001b[38;5;66;03m# Initialize H' and phi with random values\u001b[39;00m\n",
      "\u001b[1;31mNameError\u001b[0m: name 'x' is not defined"
     ]
    }
   ],
   "source": [
    "N = x.size(0)\n",
    "D = 3\n",
    "\n",
    "# Initialize H' and phi with random values\n",
    "H = torch.rand(size=(N, D))\n",
    "phi = torch.rand(size=(2 * D,))\n",
    "\n",
    "# Initialize the similarity matrix S\n",
    "S = torch.zeros((N, N))\n",
    "\n",
    "# Loop over all nodes to compute the similarity scores\n",
    "for i in range(N):\n",
    "    for j in range(N):\n",
    "        ############# Your code here ############\n",
    "        ## 1. Concatenate the features of nodes i and j\n",
    "        ## 2. Compute the dot product of concatenated features with phi\n",
    "        ## (2 lines of code)\n",
    "\n",
    "        #########################################\n",
    "        features = torch.cat([H[i], H[j]], dim=0)\n",
    "        S[i, j] = torch.dot(features, phi)\n",
    "        \n",
    "print(S)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "Td_GQQX4wiUl",
   "metadata": {
    "id": "Td_GQQX4wiUl"
   },
   "source": [
    "It's crucial to apply a mask to the pre-attention scores before they are processed through the softmax function. This ensures that the normalization is applied exclusively to the existing edges in the graph, maintaining the integrity of the graph structure.\n",
    "\n",
    "\n",
    "Construct the mask using the following equation:\n",
    "$mask = A+I$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "_Tr83bLQwz6L",
   "metadata": {
    "id": "_Tr83bLQwz6L"
   },
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'S' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[6], line 7\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[38;5;66;03m############# Your code here ############\u001b[39;00m\n\u001b[0;32m      2\u001b[0m \u001b[38;5;66;03m## contruct the mask, name it `mask`\u001b[39;00m\n\u001b[0;32m      3\u001b[0m \u001b[38;5;66;03m## return a boolean matrix NxN\u001b[39;00m\n\u001b[0;32m      4\u001b[0m \u001b[38;5;66;03m## (1 line of code)\u001b[39;00m\n\u001b[0;32m      5\u001b[0m \n\u001b[0;32m      6\u001b[0m \u001b[38;5;66;03m#########################################\u001b[39;00m\n\u001b[1;32m----> 7\u001b[0m mask \u001b[38;5;241m=\u001b[39m \u001b[43mS\u001b[49m \u001b[38;5;241m+\u001b[39m torch\u001b[38;5;241m.\u001b[39meye(N)\n\u001b[0;32m      8\u001b[0m \u001b[38;5;28mprint\u001b[39m(mask)\n",
      "\u001b[1;31mNameError\u001b[0m: name 'S' is not defined"
     ]
    }
   ],
   "source": [
    "############# Your code here ############\n",
    "## contruct the mask, name it `mask`\n",
    "## return a boolean matrix NxN\n",
    "## (1 line of code)\n",
    "\n",
    "#########################################\n",
    "mask = S + torch.eye(N)\n",
    "print(mask)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4PCNdExKxVb2",
   "metadata": {
    "id": "4PCNdExKxVb2"
   },
   "source": [
    "Apply the mask to the pre-attention\n",
    "$S[mask]$.\n",
    "\n",
    "Set $S$ to very large negative values. This is effectively to represent negative infinity in the context of the softmax operation that follows."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "Gzfru1H4w3jX",
   "metadata": {
    "id": "Gzfru1H4w3jX"
   },
   "outputs": [],
   "source": [
    "############# Your code here ############\n",
    "## get masked values for S, `S_masked`\n",
    "## hint: see torch.where\n",
    "## Note: The values masked should effectively be zero,\n",
    "## considering the limits of numerical precision.\n",
    "## (1 line of code)\n",
    "\n",
    "#########################################\n",
    "print(S_masked.exp())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fcNf5fYB1E21",
   "metadata": {
    "id": "fcNf5fYB1E21"
   },
   "source": [
    "\n",
    "Now, let's proceed to implement the Graph Attention Network (GAT). The preparatory work we've done should have provided you with all the necessary tools and understanding to successfully implement the attention based aggregation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "hMECcwy7pV6B",
   "metadata": {
    "id": "hMECcwy7pV6B"
   },
   "outputs": [],
   "source": [
    "class GAT(nn.Module):\n",
    "    \"\"\"\n",
    "    A basic implementation of the GAT layer.\n",
    "\n",
    "    This layer applies an attention mechanism in the graph convolution process,\n",
    "    allowing the model to focus on different parts of the neighborhood\n",
    "    of each node.\n",
    "    \"\"\"\n",
    "    def __init__(self, in_features, out_features, activation):\n",
    "        super(GAT, self).__init__()\n",
    "        # Initialize the weights, bias, and attention parameters as\n",
    "        # trainable parameters\n",
    "        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))\n",
    "        self.bias = nn.Parameter(torch.zeros(out_features))\n",
    "        self.phi = nn.Parameter(torch.FloatTensor(2 * out_features, 1))\n",
    "        self.activation = activation\n",
    "        self.reset_parameters()\n",
    "\n",
    "    def reset_parameters(self):\n",
    "        stdv = 1. / np.sqrt(self.weight.size(1))\n",
    "        self.weight.data.uniform_(-stdv, stdv)\n",
    "\n",
    "        stdv = 1. / np.sqrt(self.phi.size(1))\n",
    "        self.phi.data.uniform_(-stdv, stdv)\n",
    "\n",
    "    def forward(self, input, adj):\n",
    "        \"\"\"\n",
    "        Forward pass of the GAT layer.\n",
    "\n",
    "        Parameters:\n",
    "        input (Tensor): The input features of the nodes.\n",
    "        adj (Tensor): The adjacency matrix of the graph.\n",
    "\n",
    "        Returns:\n",
    "        Tensor: The output features of the nodes after applying the GAT layer.\n",
    "        \"\"\"\n",
    "        ############# Your code here ############\n",
    "        ## 1. Apply linear transformation and add bias\n",
    "        ## 2. Compute the attention scores utilizing the previously\n",
    "        ## established mechanism.\n",
    "        ## Note: Keep in mind that employing matrix notation can\n",
    "        ## optimize this process.\n",
    "        ## 3. Compute mask based on adjacency matrix\n",
    "        ## 4. Apply mask to the pre-attention matrix\n",
    "        ## 5. Compute attention weights using softmax\n",
    "        ## 6. Aggregate features based on attention weights\n",
    "        ## Note: name the last line as `h`\n",
    "        ## (9-10 lines of code)\n",
    "\n",
    "        #########################################\n",
    "        linear = torch.matmul(input, self.weight) + self.bias\n",
    "        N = linear.size(0)\n",
    "        attention = torch.cat([linear.repeat(1, N).view(N * N, -1),\n",
    "                            linear.repeat(N, 1)], dim=1).view(N, -1, 2 * self.weight.size(1))\n",
    "        attention_scores = torch.matmul(attention, self.phi).squeeze(2)\n",
    "        masked = adj + torch.eye(N, device=adj.device)\n",
    "        attention_scores = torch.where(masked == 0, torch.tensor(float('-inf'), device=adj.device), attention_scores)\n",
    "\n",
    "        # 5. Compute attention weights using softmax\n",
    "        attention_weights = F.softmax(attention_scores, dim=1)\n",
    "        \n",
    "        # 6. Aggregate features based on attention weights\n",
    "        h = torch.matmul(attention_weights, linear)        \n",
    "        return self.activation(h) if self.activation else h\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "OXVpydRKpV3m",
   "metadata": {
    "id": "OXVpydRKpV3m"
   },
   "outputs": [],
   "source": [
    "## Ensure that your implementation is flexible enough to accommodate changes\n",
    "## in the number of hidden features. Verify that the dimensions of all\n",
    "## matrices are correctly aligned, allowing for a successful forward\n",
    "## pass through the network.\n",
    "GCN(x.size(1), 3)(x, adj)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "rtH8rckF2hLR",
   "metadata": {
    "id": "rtH8rckF2hLR"
   },
   "source": [
    "## Node Classification (6 points)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1a31ef92",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "B3IdKBlv2ulI",
   "metadata": {
    "id": "B3IdKBlv2ulI"
   },
   "source": [
    "Here is a GNN for node classification"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "wSemM4ZJ2tkB",
   "metadata": {
    "id": "wSemM4ZJ2tkB"
   },
   "outputs": [],
   "source": [
    "class NodeClassifier(nn.Module):\n",
    "    def __init__(self, nfeat, nhid, nclass, dropout, gnn_layer):\n",
    "        super(NodeClassifier, self).__init__()\n",
    "\n",
    "        self.gc1 = gnn_layer(nfeat, nhid, F.relu)\n",
    "        self.gc2 = gnn_layer(nhid, nclass, None)\n",
    "        self.dropout = dropout\n",
    "\n",
    "    def forward(self, x, adj):\n",
    "        x = self.gc1(x, adj)\n",
    "        x = F.dropout(x, self.dropout, training=self.training)\n",
    "        x = self.gc2(x, adj)\n",
    "        return F.log_softmax(x, dim=1)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "IPhzOUiNazmc",
   "metadata": {
    "id": "IPhzOUiNazmc"
   },
   "source": [
    "Let's get hands-on experience by loading a benchmark dataset - the Cora dataset.\n",
    "\n",
    "Dataset: Cora is a widely-used benchmark in graph ML. It consists of a citation network where nodes represent scientific papers, and edges correspond to citations between these papers. Each paper (node) is described by a textual abstract and is categorized into one of several classes based on its content.\n",
    "\n",
    "Node Classification: we'll dive into  node classification where the goal is to predict the category of each paper in the Cora citation network."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "E7x9tcgEZe3A",
   "metadata": {
    "id": "E7x9tcgEZe3A"
   },
   "outputs": [],
   "source": [
    "def load_cora_data(subset_size=100):\n",
    "    # Load the dataset\n",
    "    dataset = CoraGraphDataset()\n",
    "    g = dataset[0]\n",
    "    if subset_size > 0:\n",
    "      # Ensure subset_size is smaller than the total number of nodes\n",
    "      total_nodes = g.num_nodes()\n",
    "      subset_size = min(subset_size, total_nodes)\n",
    "\n",
    "      # Select a subset of nodes\n",
    "      subset_nodes = torch.randperm(total_nodes)[:subset_size]\n",
    "\n",
    "      # Create a subgraph with the selected nodes\n",
    "      subg = g.subgraph(subset_nodes)\n",
    "    else:\n",
    "      subg = g\n",
    "\n",
    "    # Extract features, labels, and masks for the graph\n",
    "    features = subg.ndata['feat']\n",
    "    labels = subg.ndata['label']\n",
    "    train_mask = subg.ndata['train_mask']\n",
    "    val_mask = subg.ndata['val_mask']\n",
    "    test_mask = subg.ndata['test_mask']\n",
    "\n",
    "    adj = subg.adjacency_matrix().to_dense()\n",
    "\n",
    "    return features, labels, adj, train_mask, val_mask, test_mask\n",
    "features, labels, adj, train_mask, val_mask, test_mask = load_cora_data(-1)\n",
    "\n",
    "features = features.to(device)\n",
    "labels = labels.to(device)\n",
    "adj = adj.to(device)\n",
    "train_mask = train_mask.to(device)\n",
    "val_mask = val_mask.to(device)\n",
    "test_mask = test_mask.to(device)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "za5azmEYvHMZ",
   "metadata": {
    "id": "za5azmEYvHMZ"
   },
   "source": [
    "Feel free to analyze and print the dimensions of *features, adj*, and *labels*. This will help you debug in case of errors.\n",
    "\n",
    "Note: you can take a subset of the full graph `load_cora_data(subset_size=1000)` to spead up development. However, we expect the final submission on the full graph"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3g_YeKQUTA0y",
   "metadata": {
    "id": "3g_YeKQUTA0y"
   },
   "source": [
    "Let's define metrics to track"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "wvR11ZbbTEu4",
   "metadata": {
    "id": "wvR11ZbbTEu4"
   },
   "outputs": [],
   "source": [
    "def calculate_specificity(y_true, y_pred, labels):\n",
    "    specificity_scores = np.zeros(len(labels))\n",
    "    for i, label in enumerate(labels):\n",
    "        binary_true = (y_true == label).int()\n",
    "        binary_pred = (y_pred == label).int()\n",
    "\n",
    "        tn, fp, fn, tp = confusion_matrix(binary_true, binary_pred).ravel()\n",
    "\n",
    "        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0\n",
    "        specificity_scores[i] = specificity\n",
    "\n",
    "    return np.mean(specificity_scores)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "piwFKneJ4BNi",
   "metadata": {
    "id": "piwFKneJ4BNi"
   },
   "outputs": [],
   "source": [
    "def compute_accuracy(target, prediction):\n",
    "  ############# Your code here ############\n",
    "  ## 1. Count the number of correct predictions\n",
    "  ## 2. Get the total number of predictions\n",
    "  ## 3. Calculate the accuracy\n",
    "  ## (~3 lines of code)\n",
    "\n",
    "  #########################################\n",
    "  \n",
    "  correct_count = (target == prediction)\n",
    "  accuracy = correct_count / len(target)\n",
    "  \n",
    "  return accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "AFZVCe7DUNBV",
   "metadata": {
    "id": "AFZVCe7DUNBV"
   },
   "outputs": [],
   "source": [
    "sensitivity = lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro')\n",
    "specificity = lambda y_true, y_pred: calculate_specificity(y_true, y_pred, labels.unique())\n",
    "accuracy = lambda y_true, y_pred: compute_accuracy(y_true, y_pred)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "HHzjkKv4vwA2",
   "metadata": {
    "id": "HHzjkKv4vwA2"
   },
   "source": [
    "Almost there, write the main training function to train our GNNs."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "BDvI2ZD4gQOj",
   "metadata": {
    "id": "BDvI2ZD4gQOj"
   },
   "outputs": [],
   "source": [
    "def train_step(model, X, A, y ,train_mask, optimizer, loss_fn):\n",
    "  model.train()\n",
    "  loss = 0\n",
    "  ############# Your code here ############\n",
    "  ## 1. Zero grad the optimizer\n",
    "  ## 2. Feed the data into the model\n",
    "  ## 3. Slice the model output and label by train_mask\n",
    "  ## 4. Feed the sliced output and label to loss_fn\n",
    "  ## (~4 lines of code)\n",
    "\n",
    "  #########################################\n",
    "  model.train()\n",
    "\n",
    "  loss.zero_grad() \n",
    "  output = model(X, A)\n",
    "  \n",
    "  out = output[train_mask]\n",
    "  loss = loss_fn(out, y[train_mask])  \n",
    "\n",
    "  loss.backward()\n",
    "  optimizer.step()\n",
    "\n",
    "  return loss.item()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "zDBF9eMe31sg",
   "metadata": {
    "id": "zDBF9eMe31sg"
   },
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'torch' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[8], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m \u001b[38;5;129m@torch\u001b[39m\u001b[38;5;241m.\u001b[39mno_grad()\n\u001b[0;32m      2\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mtest\u001b[39m(model, X, A, y, train_mask, val_mask, test_mask\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mNone\u001b[39;00m, metrics\u001b[38;5;241m=\u001b[39m{}):\n\u001b[0;32m      3\u001b[0m   model\u001b[38;5;241m.\u001b[39meval()\n\u001b[0;32m      5\u001b[0m   \u001b[38;5;66;03m# The output of model on all data\u001b[39;00m\n",
      "\u001b[1;31mNameError\u001b[0m: name 'torch' is not defined"
     ]
    }
   ],
   "source": [
    "@torch.no_grad()\n",
    "def test(model, X, A, y, train_mask, val_mask, test_mask=None, metrics={}):\n",
    "  model.eval()\n",
    "\n",
    "  # The output of model on all data\n",
    "  out = None\n",
    "  logits = model(X, A)\n",
    "  preds = logits.argmax(dim=1)\n",
    "\n",
    "  result = {\n",
    "        'train': {},\n",
    "        'val': {},\n",
    "      }\n",
    "  if test_mask is not None:\n",
    "    result['test'] = {}\n",
    "\n",
    "  for name, metric in metrics.items():\n",
    "    result['train'][name] = metric(y[train_mask], preds[train_mask])\n",
    "    result['val'][name] = metric(y[val_mask], preds[val_mask])\n",
    "\n",
    "    if test_mask is not None:\n",
    "      result['test'][name] = metric(y[test_mask], preds[test_mask])\n",
    "\n",
    "  return result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "RJNEQbtB4xQY",
   "metadata": {
    "id": "RJNEQbtB4xQY"
   },
   "outputs": [],
   "source": [
    "def train(model, epochs, lr):\n",
    "  t = time.time()\n",
    "  optimizer = torch.optim.Adam(model.parameters(),\n",
    "                            lr=lr,\n",
    "                            weight_decay=5e-4)\n",
    "  loss_fn = F.nll_loss\n",
    "  metrics = {\"acc\": accuracy}\n",
    "\n",
    "  best_model = None\n",
    "  best_valid_acc = 0\n",
    "\n",
    "  for epoch in range(1, 1 + epochs):\n",
    "    loss = train_step(model, features, adj, labels, train_mask,\n",
    "                      optimizer, loss_fn)\n",
    "\n",
    "\n",
    "    result = test(model, features, adj, labels,\n",
    "                train_mask, val_mask, None, metrics)\n",
    "    train_acc = result['train']['acc']\n",
    "    valid_acc = result['val']['acc']\n",
    "    if valid_acc > best_valid_acc:\n",
    "        best_valid_acc = valid_acc\n",
    "        best_model = copy.deepcopy(model)\n",
    "    print(f'Epoch: {epoch:02d}, '\n",
    "          f'Loss: {loss:.4f}, ',\n",
    "          f'acc_train: {100*train_acc:.4f}%, ',\n",
    "          f'acc_valid: {100*valid_acc:.4f}%, ',\n",
    "          f'time: {time.time() - t:.4f}s.')\n",
    "  print(f'best acc_valid: {100*best_valid_acc:.4f}%')\n",
    "  return best_model"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7pGPt9EB9kD9",
   "metadata": {
    "id": "7pGPt9EB9kD9"
   },
   "source": [
    "Execute node classification on the Cora dataset using mean aggregation.\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bcgU70zI8KXD",
   "metadata": {
    "id": "bcgU70zI8KXD"
   },
   "outputs": [],
   "source": [
    "## Note:\n",
    "## feel free to play with the parameters to improve validation accuracy\n",
    "## you should get a validationaccuracy of at least 77%-80%\n",
    "model_GCN = NodeClassifier(nfeat=features.shape[1],\n",
    "                  nhid=16,\n",
    "                  nclass=labels.max().item() + 1,\n",
    "                  dropout=0.5,\n",
    "                  gnn_layer=GCN).to(device)\n",
    "model_GCN = train(model_GCN, 100, 0.01)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8M_Shomo95Xq",
   "metadata": {
    "id": "8M_Shomo95Xq"
   },
   "source": [
    "Execute node classification on the Cora dataset using attention based aggregation.\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "SaZjU5tn4yGk",
   "metadata": {
    "id": "SaZjU5tn4yGk"
   },
   "outputs": [],
   "source": [
    "## Note:\n",
    "## feel free to play with the parameters to improve validation accuracy\n",
    "## you should get a validationaccuracy of at least 77%-80%\n",
    "model_GAT = NodeClassifier(nfeat=features.shape[1],\n",
    "              nhid=16,\n",
    "              nclass=labels.max().item() + 1,\n",
    "              dropout=0.5,\n",
    "              gnn_layer=GAT).to(device)\n",
    "model_GAT = train(model_GAT, 100, 0.01)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "jAqFqoZPUTHN",
   "metadata": {
    "id": "jAqFqoZPUTHN"
   },
   "source": [
    "## Visualize and analize results (7 points)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "-mAEk0w8ZiMy",
   "metadata": {
    "id": "-mAEk0w8ZiMy"
   },
   "source": [
    "\n",
    "Familiarize yourself with the following utility functions, which are designed for visual analysis and will assist us in evaluating our GNNs."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "_4lCscL3U1GG",
   "metadata": {
    "id": "_4lCscL3U1GG"
   },
   "outputs": [],
   "source": [
    "def plot_embeddings(embeddings1, embeddings2, labels):\n",
    "    # Convert multi-labels to unique integers for color-coding\n",
    "    unique_labels, label_indices = np.unique(labels, axis=0, return_inverse=True)\n",
    "\n",
    "    # Set up the matplotlib figure and axes\n",
    "    fig, axs = plt.subplots(1, 2, figsize=(12, 6))\n",
    "\n",
    "    # Perform PCA for both sets of embeddings\n",
    "    pca = PCA(n_components=2)\n",
    "    pca_result1 = pca.fit_transform(embeddings1)\n",
    "    pca_result2 = pca.fit_transform(embeddings2)\n",
    "\n",
    "    # Plotting GCN\n",
    "    scatter1 = axs[0].scatter(pca_result1[:, 0], pca_result1[:, 1], c=label_indices, cmap='viridis')\n",
    "    axs[0].set_title('GCN')\n",
    "    axs[0].set_xlabel('PCA Component 1')\n",
    "    axs[0].set_ylabel('PCA Component 2')\n",
    "\n",
    "    # Plotting GAT\n",
    "    scatter2 = axs[1].scatter(pca_result2[:, 0], pca_result2[:, 1], c=label_indices, cmap='viridis')\n",
    "    axs[1].set_title('GAT')\n",
    "    axs[1].set_xlabel('PCA Component 1')\n",
    "    axs[1].set_ylabel('PCA Component 2')\n",
    "\n",
    "    # Create a color bar with label information\n",
    "    cbar = plt.colorbar(scatter1, ax=axs, orientation='vertical', fraction=0.046, pad=0.04)\n",
    "    cbar.set_label('Label Combinations')\n",
    "    cbar.set_ticks(np.arange(len(unique_labels)))\n",
    "    cbar.set_ticklabels([' + '.join(str(comb)) for comb in unique_labels])\n",
    "\n",
    "    # Show the plot\n",
    "    plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4XlA1WI7YsnC",
   "metadata": {
    "id": "4XlA1WI7YsnC"
   },
   "outputs": [],
   "source": [
    "def plot_weight_distributions(weights1, weights2):\n",
    "    # Flatten the weight matrices\n",
    "    flattened_weights1 = weights1.flatten()\n",
    "    flattened_weights2 = weights2.flatten()\n",
    "\n",
    "    # Create KDE plots\n",
    "    plt.figure(figsize=(12, 6))\n",
    "    sns.kdeplot(flattened_weights1, fill=True, color=\"r\", label=\"GCN\")\n",
    "    sns.kdeplot(flattened_weights2, fill=True, color=\"b\", label=\"GAT\")\n",
    "\n",
    "    plt.title(\"Distribution of GNN Weights\")\n",
    "    plt.xlabel(\"Weight Value\")\n",
    "    plt.ylabel(\"Density\")\n",
    "    plt.legend()\n",
    "    plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "uC6QMjPadcfk",
   "metadata": {
    "id": "uC6QMjPadcfk"
   },
   "outputs": [],
   "source": [
    "def plot_metrics(metrics, title, phases = ['train', 'val', 'test']):\n",
    "    n_phases = len(phases)\n",
    "\n",
    "    # Setting up the subplot grid\n",
    "    fig, axs = plt.subplots(n_phases, 1, figsize=(15, 6 * n_phases), sharey=True)\n",
    "\n",
    "    for i, phase in enumerate(phases):\n",
    "        # Extracting metric names\n",
    "        metric_names = list(metrics[0][phase].keys())\n",
    "\n",
    "        # Number of metrics\n",
    "        n_metrics = len(metric_names)\n",
    "\n",
    "        # Data for plotting\n",
    "        scores_modelA = [metrics[0][phase][metric] for metric in metric_names]\n",
    "        scores_modelB = [metrics[1][phase][metric] for metric in metric_names]\n",
    "\n",
    "        # Setting the positions and width for the bars\n",
    "        pos = np.arange(n_metrics)\n",
    "        bar_width = 0.35\n",
    "\n",
    "        # Plotting in the respective subplot\n",
    "        axs[i].bar(pos - bar_width/2, scores_modelA, bar_width, label='GCN')\n",
    "        axs[i].bar(pos + bar_width/2, scores_modelB, bar_width, label='GAT')\n",
    "\n",
    "        # Adding labels and titles\n",
    "        axs[i].set_xlabel('Metrics')\n",
    "        axs[i].set_title(f'{phase.capitalize()} Metrics')\n",
    "        axs[i].set_xticks(pos)\n",
    "        axs[i].set_xticklabels(metric_names)\n",
    "\n",
    "        # Adding a legend to the first subplot\n",
    "        if i == 0:\n",
    "            axs[i].legend()\n",
    "\n",
    "    # Setting the main title and showing the plot\n",
    "    plt.suptitle(title)\n",
    "    plt.tight_layout(rect=[0, 0.03, 1, 0.95])\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "di6Zhiq0ds3p",
   "metadata": {
    "id": "di6Zhiq0ds3p"
   },
   "source": [
    "### Node Embeddings"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "hnPNAFYLFORu",
   "metadata": {
    "id": "hnPNAFYLFORu"
   },
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'GCN' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[2], line 11\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[38;5;66;03m############# Your code here ############\u001b[39;00m\n\u001b[0;32m      2\u001b[0m \u001b[38;5;66;03m## 1. Generate embeddings using the GCN model\u001b[39;00m\n\u001b[0;32m      3\u001b[0m \u001b[38;5;66;03m## 2. Generate embeddings using the GAT model\u001b[39;00m\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m      8\u001b[0m \n\u001b[0;32m      9\u001b[0m \u001b[38;5;66;03m#########################################\u001b[39;00m\n\u001b[1;32m---> 11\u001b[0m H_GCN \u001b[38;5;241m=\u001b[39m \u001b[43mGCN\u001b[49m(features, adj)\u001b[38;5;241m.\u001b[39mdetach()\u001b[38;5;241m.\u001b[39mnumpy()\n\u001b[0;32m     12\u001b[0m H_GAT \u001b[38;5;241m=\u001b[39m GAT(features, adj)\u001b[38;5;241m.\u001b[39mdetach()\u001b[38;5;241m.\u001b[39mnumpy()\n\u001b[0;32m     14\u001b[0m plot_embeddings(H_GCN,H_GAT, labels)\n",
      "\u001b[1;31mNameError\u001b[0m: name 'GCN' is not defined"
     ]
    }
   ],
   "source": [
    "############# Your code here ############\n",
    "## 1. Generate embeddings using the GCN model\n",
    "## 2. Generate embeddings using the GAT model\n",
    "## Note: use detach to avoid using the computation graph\n",
    "## convert the tensor to a numpy array\n",
    "## use H_GCN and H_GAT as variable names\n",
    "## (~2 lines of code)\n",
    "\n",
    "#########################################\n",
    "\n",
    "H_GCN = GCN(x, adj).detach().numpy()\n",
    "H_GAT = GAT(features, adj).detach().numpy()\n",
    "\n",
    "plot_embeddings(H_GCN, H_GAT, labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "c25a18d8",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'gcn_model' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[3], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m Omega_GCN_gc2 \u001b[38;5;241m=\u001b[39m \u001b[43mgcn_model\u001b[49m\u001b[38;5;241m.\u001b[39mgc2\u001b[38;5;241m.\u001b[39mweight\u001b[38;5;241m.\u001b[39mdetach()\u001b[38;5;241m.\u001b[39mcpu()\u001b[38;5;241m.\u001b[39mnumpy()\n\u001b[0;32m      2\u001b[0m Omega_GAT_gc2 \u001b[38;5;241m=\u001b[39m gat_model\u001b[38;5;241m.\u001b[39mgc2\u001b[38;5;241m.\u001b[39mweight\u001b[38;5;241m.\u001b[39mdetach()\u001b[38;5;241m.\u001b[39mcpu()\u001b[38;5;241m.\u001b[39mnumpy()\n",
      "\u001b[1;31mNameError\u001b[0m: name 'gcn_model' is not defined"
     ]
    }
   ],
   "source": [
    "Omega_GCN_gc2 = gcn_model.gc2.weight.detach().cpu().numpy()\n",
    "Omega_GAT_gc2 = gat_model.gc2.weight.detach().cpu().numpy()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "qgz-LV0laj5U",
   "metadata": {
    "id": "qgz-LV0laj5U"
   },
   "source": [
    "Based on the visualizations in the plots, what insights can you gather about the embeddings generated by each model?"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "PSbW4t-Ed6kr",
   "metadata": {
    "id": "PSbW4t-Ed6kr"
   },
   "source": [
    "######## Your response here (double-click) ########\n",
    "\n",
    "\n",
    "\n",
    "#########################################"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b5NkRAVHefKI",
   "metadata": {
    "id": "b5NkRAVHefKI"
   },
   "source": [
    "### Distribution of learned weights"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bcXWvuy0Yc7n",
   "metadata": {
    "id": "bcXWvuy0Yc7n"
   },
   "outputs": [],
   "source": [
    "############# Your code here ############\n",
    "## 1. Extract the weights from the second graph\n",
    "## convolutional layer (gc2) of the GCN model\n",
    "## 2. Extract the weights from the second graph\n",
    "## convolutional layer (gc2) of the GAT model\n",
    "## Note: use detach to avoid using the computation graph\n",
    "## convert the tensor to a numpy array\n",
    "## use Omega_GCN_gc2 and Omega_GAT_gc2 as variable names\n",
    "## (~2 lines of code)\n",
    "\n",
    "#########################################\n",
    "plot_weight_distributions(Omega_GCN_gc2, Omega_GAT_gc2)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "uffhAF6Ie9iD",
   "metadata": {
    "id": "uffhAF6Ie9iD"
   },
   "source": [
    "Reflect on the observed distributions of the weights from the models. What conclusions or understandings can be drawn about the behavior and characteristics of each model based on these distributions?\n",
    "\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "Poj0JkdBe9vV",
   "metadata": {
    "id": "Poj0JkdBe9vV"
   },
   "source": [
    "######## Your response here (double-click) ########\n",
    "\n",
    "\n",
    "\n",
    "#########################################"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "WwhgePeTe4tK",
   "metadata": {
    "id": "WwhgePeTe4tK"
   },
   "source": [
    "### Metrics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "TjhJAM3GbkWW",
   "metadata": {
    "id": "TjhJAM3GbkWW"
   },
   "outputs": [],
   "source": [
    "metrics = {\"acc\": accuracy, \"sensitivity\": sensitivity, \"specifity\": specificity}\n",
    "results_GCN = test(model_GCN, features, adj,\n",
    "                   labels, train_mask, val_mask, test_mask, metrics)\n",
    "results_GAT = test(model_GAT, features, adj,\n",
    "                   labels, train_mask, val_mask, test_mask, metrics)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "26gMjOincTMm",
   "metadata": {
    "id": "26gMjOincTMm"
   },
   "outputs": [],
   "source": [
    "plot_metrics([results_GCN, results_GAT], 'Metric Comparison')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ArHRsbJ7gv9t",
   "metadata": {
    "id": "ArHRsbJ7gv9t"
   },
   "source": [
    "How might you interpret the outcomes of these results?\n",
    "\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "hWjw5eD7gmx7",
   "metadata": {
    "id": "hWjw5eD7gmx7"
   },
   "source": [
    "######## Your response here (double-click) ########\n",
    "\n",
    "\n",
    "\n",
    "#########################################"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "qIc4lj0whLbW",
   "metadata": {
    "id": "qIc4lj0whLbW"
   },
   "source": [
    "## Bonus question 1\n",
    "\n",
    "What changes in the GCN and GAT layers would result in a more effective implementation?\n",
    "\n",
    "Hint: See how [message passing](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html) is implemented in PyTorch Geometric.\n",
    "\n",
    "Furthermore, explore the specific implementations for [GCN](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html) and [GAT](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATConv.html).\n",
    "\n",
    "*Note: Bonus questions are ungraded yet provide an opportunity for those who wish to delve deeper into their graph based learning journey.*\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "Q4uW3ekljtlQ",
   "metadata": {
    "id": "Q4uW3ekljtlQ"
   },
   "source": [
    "######## Your response here (double-click) ########\n",
    "\n",
    "\n",
    "\n",
    "#########################################"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ahE7TBzSwKCt",
   "metadata": {
    "id": "ahE7TBzSwKCt"
   },
   "source": [
    "## Bonus question 2\n",
    "\n",
    "Write the kipf normalization\n",
    "\n",
    "$H_{k+1} = a[\\beta_k\\mathbf{1}^T + \\Omega_kH_k(D^{-1/2}AD^{-1/2}+I)]$\n",
    "\n",
    "*Note: Bonus questions are ungraded yet provide an opportunity for those who wish to delve deeper into their graph based learning journey.*\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "IQ_VTX4Awo7n",
   "metadata": {
    "id": "IQ_VTX4Awo7n"
   },
   "outputs": [],
   "source": [
    "def kipf_normalization(A):\n",
    "  ############# Your code here ############\n",
    "\n",
    "  #########################################\n",
    "  return A"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "Gaf3LOV7c5O9",
   "metadata": {
    "id": "Gaf3LOV7c5O9"
   },
   "source": [
    "# 4) Propagation rule integrating edge features/embeddings (15 points)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "aCStFPrA8rAD",
   "metadata": {
    "id": "aCStFPrA8rAD"
   },
   "outputs": [],
   "source": [
    "\n",
    "seed = 42\n",
    "np.random.seed(seed)  # Setting a fixed seed for numpy random number generator\n",
    "\n",
    "\n",
    "# we will create a stochastic block model graph with two communities and 20 nodes\n",
    "G = nx.stochastic_block_model(sizes=[10, 10], p=[[0.8, 0.1], [0.1, 0.8]], seed=seed)\n",
    "\n",
    "# We will create a node feature\n",
    "communities = nx.community.louvain_communities(G)\n",
    "X = np.zeros((len(G), 1))\n",
    "X[list(communities[0])] = 1\n",
    "\n",
    "A = nx.adjacency_matrix(G).todense()\n",
    "\n",
    "# we will use powers of the adjacency matrix as edge features (considering edge embeddings only on edges already existing)\n",
    "E1 = A\n",
    "E2 = (A @ A)*A # we zero the embedding on edges which aren't connected\n",
    "E3 = (A @ A @ A)*A\n",
    "E4 = (A @ A @ A @ A)*A\n",
    "\n",
    "E = np.stack((E1, E2, E3, E4), axis = 2)\n",
    "\n",
    "print(E)\n",
    "print(X)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3EPth4u885N-",
   "metadata": {
    "id": "3EPth4u885N-"
   },
   "source": [
    "### 4.1) Graph visualization <b>(5 points)</b>\n",
    "\n",
    "#### 4.1.1 Use networkx to visualize the structure and the node features of the graph. (2 points)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "svQEBbzg8xX_",
   "metadata": {
    "id": "svQEBbzg8xX_"
   },
   "outputs": [],
   "source": [
    "def VisualizeGraphNetworkx(G):\n",
    "    '''\n",
    "    input: G\n",
    "    A networkx graph instance\n",
    "    '''\n",
    "    #Complete code here:\n",
    "    ####################\n",
    "\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2BRqCmSV9CxU",
   "metadata": {
    "id": "2BRqCmSV9CxU"
   },
   "source": [
    "#### 4.1.2 Display the edge features for edges which are connected to node 0 (first node). (3 points)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "GAa3JxT79E_7",
   "metadata": {
    "id": "GAa3JxT79E_7"
   },
   "outputs": [],
   "source": [
    "''' code to display the edge features for all edges connected to node 0. List the output in the format\n",
    "[node 0, node x, edge_features]\n",
    "[node 0, node y, edge_features] ...\n",
    " Please change = None to your implementation.'''\n",
    "edge_features = None\n",
    "print(edge_features)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "oqL_AF0m9JCD",
   "metadata": {
    "id": "oqL_AF0m9JCD"
   },
   "source": [
    "### 4.2 Mathematical Formulation <b>(5 points)</b>\n",
    "\n",
    "Consider a graph with node embedding $\\mathbf{h}_n$ based on its neighboring node embeddings $\\{\\mathbf{h}_m\\}_{m \\in ne[n]}$ and the neighboring edge embeddings $\\{\\mathbf{e}_m\\}_{m \\in nee[n]}$\n",
    "$ne[n]$ denotes the neighbors of node $n$ and $nee[n]$ denotes the edges connected to node $n$.\n",
    "  \n",
    "Let the update rule without accounting for edge embeddings is the following\n",
    "\n",
    "$\\mathbf{H}_{k+1} = a[ \\mathbf{\\beta}_k \\mathbf{1}^T + \\mathbf{\\Omega}_k \\mathbf{H}_k (\\mathbf{A} + \\mathbf{I}) ]$"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0wTK6G6X9au5",
   "metadata": {
    "id": "0wTK6G6X9au5"
   },
   "source": [
    "#### 4.2.1 fill in the [...] in the equation below so that we account for edge embeddings (3 points)\n",
    "\n",
    "$\\mathbf{H}_{k+1} = a[ \\mathbf{\\beta}_k \\mathbf{1}^T + \\mathbf{\\Omega}_k \\mathbf{H}_k (\\mathbf{A} + \\mathbf{I}) + [...]]$  (1)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "-o0WJog-9cYa",
   "metadata": {
    "id": "-o0WJog-9cYa"
   },
   "source": [
    "#### 4.2.2 Define any new variables you introduce in the matrix form of (1) and specify their size. (2 points)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "FO84ej4M9e9S",
   "metadata": {
    "id": "FO84ej4M9e9S"
   },
   "source": [
    "### 4.3) Implementation <b>(5 points)</b>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "qtFGBFbG9h5B",
   "metadata": {
    "id": "qtFGBFbG9h5B"
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import networkx as nx\n",
    "import matplotlib.pyplot as plt\n",
    "import torch\n",
    "np.random.seed(42)  # Setting a fixed seed for numpy random number generator\n",
    "torch.manual_seed(42)\n",
    "\n",
    "\n",
    "def sigmoid(x):\n",
    "    return 1 / (1 + np.exp(-x))\n",
    "\n",
    "\n",
    "def add_self_connections(A):\n",
    "    \"\"\"\n",
    "    Add self-connections to the adjacency matrix.\n",
    "\n",
    "    This function adds an identity matrix to the adjacency matrix `A`, which\n",
    "    effectively adds a self-loop to each node in the graph. This is a common\n",
    "    preprocessing step in graph neural network implementations.\n",
    "\n",
    "    Parameters:\n",
    "    A (np.ndarray): The adjacency matrix to modify.\n",
    "\n",
    "    Returns:\n",
    "    np.ndarray: The adjacency matrix with self-connections added.\n",
    "    \"\"\"\n",
    "    I = np.eye(A.shape[0])\n",
    "    return A + I\n",
    "\n",
    "\n",
    "def normalize_adjacency(A_hat):\n",
    "    \"\"\"\n",
    "    Normalize the adjacency matrix.\n",
    "\n",
    "    This function applies symmetric normalization to the adjacency matrix\n",
    "    `A_hat`. The normalization is done using the inverse square root of the\n",
    "    degree matrix. This step is important for many graph-based learning\n",
    "    algorithms to ensure that the scale of the feature representations is not\n",
    "    skewed by node degree.\n",
    "\n",
    "    Parameters:\n",
    "    A_hat (np.ndarray): The adjacency matrix with self-connections.\n",
    "\n",
    "    Returns:\n",
    "    np.ndarray: The normalized adjacency matrix.\n",
    "    \"\"\"\n",
    "    D_hat_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(A_hat, axis=0)))\n",
    "    return D_hat_inv_sqrt.dot(A_hat).dot(D_hat_inv_sqrt)\n",
    "\n",
    "\n",
    "\n",
    "# GCN update steps\n",
    "A_hat = add_self_connections(A) # A is from question 1.2\n",
    "\n",
    "# define input features\n",
    "H_k = X\n",
    "\n",
    "# Define the dimensions of the weight matrix\n",
    "input_features = H_k.shape[1]  # Number of input features (columns of H_k)\n",
    "output_features = 2  # Number of output features (can be chosen based on your model's design)\n",
    "\n",
    "# Initialize the weight matrix Omega_k with random values\n",
    "Omega_k = np.random.rand(input_features, output_features)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "QkMfH2569lAo",
   "metadata": {
    "id": "QkMfH2569lAo"
   },
   "source": [
    "#### 4.1 Implement equation (1) and print out an update using this equation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "oJasGBjZ9nX5",
   "metadata": {
    "id": "oJasGBjZ9nX5"
   },
   "outputs": [],
   "source": [
    "''' Please implement your update which includes edge embeddings below (change = None)\n",
    "You should use A, X and E defined at the beginning of 4)\n",
    "\n",
    "'''\n",
    "H_k_edge_embeddings = None\n",
    "\n",
    "print(\"\\nWith edge-embeddings:\\n\", H_k_edge_embeddings)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "eqEsEFp8c5O-",
   "metadata": {
    "id": "eqEsEFp8c5O-"
   },
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "Yh_VUCCsc5O-",
   "metadata": {
    "id": "Yh_VUCCsc5O-"
   },
   "source": [
    "# 5) Bonus (5 points)\n",
    "\n",
    "#### Graph Fusion"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "S0Fzftx0_kNV",
   "metadata": {
    "id": "S0Fzftx0_kNV"
   },
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "# here we have two datasets of the same digits. One using pixels and the other in the Fourier Domain.\n",
    "\n",
    "# load digits in fourier domain\n",
    "df1 = pd.read_csv('sample_data/fourier.csv', header=None)\n",
    "X1 = df1.to_numpy()\n",
    "print(len(X1))\n",
    "\n",
    "# load digits in pixel domain\n",
    "df2 = pd.read_csv('sample_data/pixel.csv', header=None)\n",
    "X2 = df2.to_numpy()\n",
    "print(len(X2))\n",
    "\n",
    "\n",
    "# generate distances between all pairs of points to create distance matrix\n",
    "A1 = euclidean_distances(X1,X1)\n",
    "A2 = euclidean_distances(X2,X2)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2VRqe8vn_dPR",
   "metadata": {
    "id": "2VRqe8vn_dPR"
   },
   "source": [
    "Here we have two graphs $G =(A1,X1)$ and $G' = (A2,X2)$ representating the same dataset in different domains.\n",
    "\n",
    "#### 5.1) Use the SNF graph cross-diffusion rule (see equations (4) and (5) in the SNF paper) to define a function $G_f = fuse(G,G')$ that fuses the node and edge features of these two graphs. Formalize the solution mathematically in the matrix form (2 points)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "_GCv6arK_dnL",
   "metadata": {
    "id": "_GCv6arK_dnL"
   },
   "source": [
    "answer here"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "Bj2SPVR9_dq0",
   "metadata": {
    "id": "Bj2SPVR9_dq0"
   },
   "source": [
    "#### 5.2) Cross-diffuse graphs G and G' above using your new formula. (3 points)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2UREw95M_dy7",
   "metadata": {
    "id": "2UREw95M_dy7"
   },
   "source": [
    "answer here"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f3YOZHLJc5O-",
   "metadata": {
    "id": "f3YOZHLJc5O-"
   },
   "source": [
    "#### References:\n",
    "1. SNF paper: Wang, Bo, et al. \"Similarity network fusion for aggregating data types on a genomic scale.\" Nature methods 11.3 (2014): 333-337.\n",
    "2. Feel free to check  <A Href=\"https://www.youtube.com/watch?v=Oqrjkm6TIy8&list=PLug43ldmRSo3MV-Jgjr30E5SpwNKkjTvJ&index=32\">video</A> from minute 21 to 30 (9 minutes to cover)."
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "collapsed_sections": [
    "LQ7SGGtic5O6"
   ],
   "gpuType": "T4",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
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
   "version": "3.11.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
