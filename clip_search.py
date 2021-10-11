import argparse
import heapq
import os
import shutil
from timeit import default_timer as timer

import torch
from PIL import Image

from CLIP import clip


def image_copy(a_list):
    print("copying")
    if args.copy_remove:
        shutil.rmtree(args.copy_folder, ignore_errors=True)
    os.makedirs(args.copy_folder, exist_ok=True)
    for result in range(args.results):
        shutil.copy(f"{args.folder}/{a_list[result][0]}", args.copy_folder + "/")


def get_sim(features):
    return sim_func(features, target_features).item()


def save_dict(dict_to_save, filename):
    torch.save(dict_to_save, filename)


def load_dict(init, filename):
    if init:
        print("Initiating new dict")
        loaded_dict = dict()
    else:
        try:
            loaded_dict = torch.load(filename, map_location=device)
        except FileNotFoundError:
            print("No dict found")
            loaded_dict = dict()
            pass

    return loaded_dict


if __name__ == "__main__":
    start = timer()
    parser = argparse.ArgumentParser(description="(WIP) A simple tool for searching images "
                                                 "inside a local folder with text/image input using CLIP")
    parser.add_argument("-t",  "--text",        type=str, default=None,   help="Text search")
    parser.add_argument("-i",  "--image",       type=str, help="Image search")
    parser.add_argument("-r",  "--results",     type=int, default=5,    help="Number of search results to return")
    parser.add_argument("-se", "--save_every",  type=int, default=1000, help="Dictionary save frequency")
    parser.add_argument("-f",  "--folder",      type=str, default="images", help="Folder to scan")
    parser.add_argument("-d",  "--dict",        type=str, default=None, help="Stored dictionary file")
    parser.add_argument("-de", "--device",      type=str, default=None, help="Device to use (\"cuda\" or \"cpu\")")
    parser.add_argument("-fo", "--format",      type=str, default="a picture of ",  help="Text search formatting")
    parser.add_argument("-cf", "--copy_folder", type=str, default="results",  help="Results folder")
    parser.add_argument("-in", "--initiate",    action="store_true",  help="Initiate new dictionary (overwrite)")
    parser.add_argument("-c",  "--copy",        action="store_true",  help="Copy images to results folder")
    parser.add_argument("-cr", "--copy_remove", action="store_true",  help="Remove old results from folder")
    args = parser.parse_args()
    assert args.text or args.image, "Text or Image prompt required"
    assert os.path.isdir(args.folder), f"Folder not found: {args.folder}"

    exts = ("jpg", "jpeg", "png", "jfif")

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dict is None:
        args.dict = f"{args.folder}_features.pt"

    model, preprocess = clip.load("ViT-B/32", device=device)
    print(f"Using Device: {device}")
    sim_func = torch.nn.CosineSimilarity()

    with torch.inference_mode():
        if args.text:
            texts = f"{args.format}{args.text}"
            text = clip.tokenize(texts).to(device)
            target_features = model.encode_text(text)
            target_features /= target_features.norm(dim=-1, keepdim=True)
        else:
            target_image = preprocess(Image.open(args.image)).unsqueeze(0).to(device)
            target_features = model.encode_image(target_image)
            target_features /= target_features.norm(dim=-1, keepdim=True)

        image_dict = load_dict(args.initiate, args.dict)
        probs_dict = dict()
        new_counter = 0
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
                    save_dict(image_dict, args.dict)

                sim = get_sim(image_features)
                print(f" | similarity {sim*100:.3f}%")
                probs_dict[file] = sim

    if new_counter != 0:
        save_dict(image_dict, args.dict)

    heap = heapq.nlargest(args.results, probs_dict, key=probs_dict.get)
    a = sorted({image: probs_dict[image] for image in heap}.items(), key=lambda x: x[1], reverse=True)

    print("-"*55)
    print("Results:")
    print("-"*55)
    for i in range(args.results):
        print(f"{a[i][0][:min(len(file), 27)]:29s} {i:03d} | similarity {a[i][1]*100:.3f}%")
    print("-"*55)

    if args.copy:
        image_copy(a)

    end = timer()
    print(f"Processing time:{end-start:.3f}")
