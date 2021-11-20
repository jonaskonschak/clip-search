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
git clone https://github.com/openai/CLIP.git
pip install -r CLIP/requirements.txt
```

## Usage
Basic Usage:
```
#Any amount/combination of text/image prompts is possible
#Copy 5 results for the text search "a car" to the results folder (and initiate dictionary if it doesn't exist)
python clip_search.py -f path/to/image/folder -r 5 -c -cr -t "a car"

#Return 10 results for image.png and "a car" each using dictionary dict.pt, without copying the images (only return names)
python clip_search.py -f path/to/image/folder -r 10 -d dict.pt -t "a car" -i image.png
```
```
usage: clip_search.py [-h] [-t TEXTS [TEXTS ...]] [-i IMAGES [IMAGES ...]] [-r RESULTS] [-se SAVE_EVERY] [-f FOLDER] [-d DICT] [-de DEVICE] [-fo FORMAT] [-cf COPY_FOLDER] [-in] [-c] [-cr] [-rc]

(WIP) A simple tool for searching images inside a local folder with text/image input using CLIP

optional arguments:
  -h, --help            show this help message and exit
  -t TEXTS [TEXTS ...], --texts TEXTS [TEXTS ...]
                        Texts to search for: "a house" "an old man"
  -i IMAGES [IMAGES ...], --images IMAGES [IMAGES ...]
                        Images to search for: "/path/to/img1.jpg" "path/to/img2.jpg"
  -r RESULTS, --results RESULTS
                        Number of search results to return. Default: 5
  -se SAVE_EVERY, --save_every SAVE_EVERY
                        Dictionary save frequency. Default: 1000
  -f FOLDER, --folder FOLDER
                        Folder to scan. Default: images
  -d DICT, --dict DICT  Stored dictionary file
  -de DEVICE, --device DEVICE
                        Device to use ("cuda" or "cpu"). Default: cuda if available
  -fo FORMAT, --format FORMAT
                        Text search formatting. Default="a picture of {text prompt}"
  -cf COPY_FOLDER, --copy_folder COPY_FOLDER
                        Results folder. Default: results
  -in, --initiate       Initiate new dictionary (overwrite)
  -c, --copy            Copy images to results folder
  -cr, --copy_remove    Remove old results from folder
  -rc, --recursive      Recursive search
```
