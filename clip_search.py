import argparse
import heapq
import os
import shutil
from timeit import default_timer as timer

import torch
from PIL import Image
from torchvision.datasets import CIFAR100

from CLIP import clip

start = timer()

parser = argparse.ArgumentParser(description="")
parser.add_argument("-t",       "--text",           type=str,           default="")
parser.add_argument("-i",       "--image",          type=str)
parser.add_argument("-r",       "--results",        type=int,           default=5)
parser.add_argument("-se",      "--save_every",     type=int,           default=1000)
parser.add_argument("-f",       "--folder",         type=str,           default="images")
parser.add_argument("-d",       "--dict",           type=str,           default=None)
parser.add_argument("-fo",      "--format",         type=str,           default="a picture of ")
parser.add_argument("-in",      "--initiate",       action="store_true")
parser.add_argument("-c",       "--copy",           action="store_true")
parser.add_argument("-cr",      "--copy_remove",    action="store_true")
parser.add_argument("-cf",      "--copy_folder",    type=str,           default="results")
args = parser.parse_args()
assert args.text or args.image


exts = ("jpg", "jpeg", "png", "jfif")
probs_dict = dict()
images = []

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print(f"Using Device: {device}")

if args.dict is None:
    dict_filename = f"{args.folder}_features.pt"
else:
    dict_filename = args.dict

with torch.no_grad():
    cifar100 = CIFAR100(root=os.getcwd() + "/cifar", download=True, train=False)
    texts = [f"{args.format}{args.text}"] + [f"{args.format}{c}" for c in cifar100.classes]
    text = clip.tokenize(texts).to(device)
    target_features = model.encode_text(text)
    target_features /= target_features.norm(dim=-1, keepdim=True)
    if args.image:
        target_image = preprocess(Image.open(args.image)).unsqueeze(0).to(device)
        target_image_features = model.encode_image(target_image)
        target_image_features /= target_image_features.norm(dim=-1, keepdim=True)
        target_features[0] = target_image_features


def image_copy(a_list):
    print("copying")
    if args.copy_remove:
        try:
            shutil.rmtree(args.copy_folder)
        except FileNotFoundError:
            pass
    os.makedirs(args.copy_folder, exist_ok=True)
    for result in range(args.results):
        shutil.copy(f"{args.folder}/{a_list[result][0]}", args.copy_folder + "/")


def save_dict(dict_to_save, filename):
    torch.save(dict_to_save, filename)


def load_dict(init, filename):
    if init:
        print("Initiating new dict")
        loaded_dict = dict()
    else:
        try:
            loaded_dict = torch.load(filename)
        except FileNotFoundError:
            print("No dict found")
            loaded_dict = dict()
            pass
    
    return loaded_dict


image_dict = load_dict(args.initiate, dict_filename)

new_counter = 0
with torch.no_grad():
    for idx, file in enumerate(os.listdir(args.folder)):
        if file.lower().endswith(exts):
            short_filename = file[:min(len(file), 20)]
            if file in image_dict:
                print(f"{short_filename:20s} saved {idx:06d}", end="")
                image_features = image_dict[file].to(device)
                
            else:
                print(f"{short_filename:26s} {idx:06d}", end="")
                image = preprocess(Image.open(f"{args.folder}/{file}")).unsqueeze(0).to(device)
                image_features = model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_dict[file] = image_features
                new_counter += 1
                
            if new_counter % args.save_every == 0 and new_counter != 0:
                save_dict(image_dict, dict_filename)
                
            similarity = (image_features @ target_features.T).softmax(dim=-1)
            print(f" | similarity {similarity[0][0]:.6f}")
            probs_dict[file] = similarity[0][0]

if new_counter != 0:
    save_dict(image_dict, dict_filename)

heap = heapq.nlargest(args.results, probs_dict, key=probs_dict.get)
a = sorted({image: probs_dict[image] for image in heap}.items(), key=lambda x: x[1], reverse=True)

print("-"*55)
print("Results:")
print("-"*55)
for i in range(args.results):
    print(f"{a[i][0][:min(len(file), 27)]:29s} {i:03d} | similarity {a[i][1]:.6f}")
print("-"*55)
   
if args.copy:
    image_copy(a)

end = timer()
print(f"Processing time:{end-start:.3f}")