# clip-search
(WIP)
A simple tool for searching images inside a local folder with text/image input
using [CLIP](https://github.com/openai/CLIP)
  
![git](https://user-images.githubusercontent.com/90077736/136833546-b153204c-a37a-440f-bfc3-35532007c554.png)
10 results for "a blonde woman" in a folder of 30k randomly generated faces
## Installation
Install [PyTorch](https://pytorch.org/get-started/locally/) with GPU support if you want to use GPU  
Initially encoding images might run faster with GPU, 
but from limited testing it seems like searching with a saved dictionary might be faster on cpu.
```
git clone https://github.com/kanttouchthis/clip-search.git
cd clip-search
pip install git+https://github.com/openai/CLIP.git
```

## Usage
```python
from searcher import CLIPSearcher

cs = CLIPSearcher(device="cuda")
# Initialize Folder (this may take a while)
cs.load_dir("path/to/image/folder", recursive=True)
# Search for "car" and "house", show 5 results
cs.search(texts=["car", "house"], results=5)
# Search for "car" and house.png, copy results to the folders ./results/car and ./results/house.png
cs.search(texts=["car"], images=["house.png"], outdir="./results")
```
## CLI Usage
TODO
