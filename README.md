﻿# clip-search
(WIP)
A simple tool for searching images inside a local folder with text/image input
using [CLIP](https://github.com/openai/CLIP)
  
![git](https://user-images.githubusercontent.com/90077736/136833546-b153204c-a37a-440f-bfc3-35532007c554.png)
10 results for "a blonde woman" in a folder of 30k randomly generated faces
## Installation
Install [PyTorch](https://pytorch.org/get-started/locally/) with GPU support if you want to use GPU  
It works okay on CPU, but is a little slower  
Note: Dicts created with GPU currently do not seem to be compatible with running on CPU and vice versa
```
git clone https://github.com/kanttouchthis/clip-search.git
cd clip-search
git clone https://github.com/openai/CLIP.git
pip install -r CLIP/requirements.txt
```

## Usage
Basic Usage:
```
#Copy 5 results for the text search "a car" to the results folder (and initiate dictionary if it doesn't exist)
python clip_search.py -f path/to/image/folder -r 5 -c -cr -t "a car"

#Return 10 results for the image search image.png using dictionary dict.pt, without copying the images (only return names)
python clip_search.py -f path/to/image/folder -r 10 -d dict.pt -i image.png
```
```
usage: clip_search.py [-h] [-t TEXTS [TEXTS ...]] [-i IMAGES [IMAGES ...]] [-r RESULTS] [-se SAVE_EVERY] [-f FOLDER] [-d DICT] [-de DEVICE] [-fo FORMAT] [-cf COPY_FOLDER] [-in] [-c] [-cr]

(WIP) A simple tool for searching images inside a local folder with text/image input using CLIP

optional arguments:
  -h, --help            show this help message and exit
  -t TEXTS [TEXTS ...], --texts TEXTS [TEXTS ...]
                        Texts to search for
  -i IMAGES [IMAGES ...], --images IMAGES [IMAGES ...]
                        Images to search for
  -r RESULTS, --results RESULTS
                        Number of search results to return
  -se SAVE_EVERY, --save_every SAVE_EVERY
                        Dictionary save frequency
  -f FOLDER, --folder FOLDER
                        Folder to scan
  -d DICT, --dict DICT  Stored dictionary file
  -de DEVICE, --device DEVICE
                        Device to use ("cuda" or "cpu")
  -fo FORMAT, --format FORMAT
                        Text search formatting
  -cf COPY_FOLDER, --copy_folder COPY_FOLDER
                        Results folder
  -in, --initiate       Initiate new dictionary (overwrite)
  -c, --copy            Copy images to results folder
  -cr, --copy_remove    Remove old results from folder

```
