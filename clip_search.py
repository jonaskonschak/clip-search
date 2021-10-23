import argparse
import heapq
import os
import shutil
from timeit import default_timer as timer

import torch
from PIL import Image

from CLIP import clip


def image_copy(a_list, copy_folder):
    if args.copy_remove:
        shutil.rmtree(copy_folder, ignore_errors=True)
    os.makedirs(copy_folder, exist_ok=True)
    for result in range(args.results):
        shutil.copy(f"{a_list[result][0]}", copy_folder + "/")


def load_dict(filename):
    if args.initiate:
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


def get_targets():
    input_targets = dict()
    if args.texts:
        for arg_text in args.texts:
            target_text = f"{args.format}{arg_text}"
            target_text = clip.tokenize(target_text).to(device)
            target_features = model.encode_text(target_text)
            target_features /= target_features.norm(dim=-1, keepdim=True)
            input_targets.update({arg_text: target_features})
    if args.images:
        for arg_image in args.images:
            target_image = preprocess(Image.open(arg_image)).unsqueeze(0).to(device)
            target_features = model.encode_image(target_image)
            target_features /= target_features.norm(dim=-1, keepdim=True)
            base_filename = os.path.basename(arg_image)
            filename = base_filename
            filename_idx = 2
            while filename in input_targets.keys():
                filename = base_filename + f"_{filename_idx}"
                filename_idx += 1
            input_targets.update({filename: target_features})
    return input_targets


def load_and_get_sim(filename):
    global counter
    global new_counter
    if filename.lower().endswith(exts):
        if filename in image_dict:
            image_features = image_dict[filename]
        else:
            # Load image file and get features
            image = preprocess(Image.open(filename)).unsqueeze(0).to(device)
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_dict[filename] = image_features
            new_counter += 1
        counter += 1
        if new_counter % args.save_every == 0 and new_counter != 0:
            torch.save(image_dict, args.dict)
        if counter % args.save_every == 0:
            print(f"Loaded {counter} images. {new_counter} new")
        for (target, features) in targets.items():
            # Get similarity of image and each target
            sim_dict[target][filename] = sim_func(image_features, features).item()


if __name__ == "__main__":
    load_start = timer()
    parser = argparse.ArgumentParser(description="(WIP) A simple tool for searching images "
                                                 "inside a local folder with text/image input using CLIP")
    parser.add_argument("-t",  "--texts",       type=str, nargs="+",    default=None,
                        help="Texts to search for: \"a house\" \"an old man\"")
    parser.add_argument("-i",  "--images",      type=str, nargs="+",    default=None,
                        help="Images to search for: \"/path/to/img1.jpg\" \"path/to/img2.jpg\"")
    parser.add_argument("-r",  "--results",     type=int, default=5,
                        help="Number of search results to return. Default: 5")
    parser.add_argument("-se", "--save_every",  type=int, default=1000,
                        help="Dictionary save frequency. Default: 1000")
    parser.add_argument("-f",  "--folder",      type=str, default="images",
                        help="Folder to scan. Default: images")
    parser.add_argument("-d",  "--dict",        type=str, default=None,
                        help="Stored dictionary file")
    parser.add_argument("-de", "--device",      type=str, default=None,
                        help="Device to use (\"cuda\" or \"cpu\"). Default: cuda if available")
    parser.add_argument("-fo", "--format",      type=str, default="a picture of ",
                        help="Text search formatting. Default=\"a picture of {text prompt}\"")
    parser.add_argument("-cf", "--copy_folder", type=str, default="results",
                        help="Results folder. Default: results")
    parser.add_argument("-in", "--initiate",    action="store_true",
                        help="Initiate new dictionary (overwrite)")
    parser.add_argument("-c",  "--copy",        action="store_true",
                        help="Copy images to results folder")
    parser.add_argument("-cr", "--copy_remove", action="store_true",
                        help="Remove old results from folder")
    parser.add_argument("-rc",  "--recursive",   action="store_true",
                        help="Recursive search")
    args = parser.parse_args()
    assert args.texts or args.images, "Text or Image prompt required"
    assert os.path.isdir(args.folder), f"Folder not found: {args.folder}"

    exts = ("jpg", "jpeg", "png", "jfif")

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dict is None:
        args.dict = f"{args.folder}_features.pt"

    print(f"Using Device: {device}\nLoading model...", end=" ")
    model, preprocess = clip.load("ViT-B/32", device=device)
    print("Done")
    sim_func = torch.nn.CosineSimilarity()
    get_sim = load_and_get_sim
    load_end = timer()
    print(f"Loading time: {load_end - load_start:.3f} seconds")
    start = timer()
    counter = 0
    new_counter = 0
    if args.recursive:
        images = []
        for path, _, files in os.walk(args.folder):
            for file in files:
                if file.lower().endswith(exts):
                    images.append(os.path.join(path, file))
    else:
        images = [os.path.join(args.folder, file) for file in os.listdir(args.folder) if file.lower().endswith(exts)]
    with torch.inference_mode():
        targets = get_targets()
        image_dict = load_dict(args.dict)
        sim_dict = dict()
        for target_name in targets.keys():
            sim_dict[target_name] = dict()
        [get_sim(file) for file in images]
        print(f"Loaded {counter} images. {new_counter} new  | Finished")

    if new_counter != 0:
        torch.save(image_dict, args.dict)
    for t in sim_dict.keys():
        # Sort for highest similarity
        heap = heapq.nlargest(args.results, sim_dict[t], key=sim_dict[t].get)
        a = sorted({image: sim_dict[t][image] for image in heap}.items(), key=lambda x: x[1], reverse=True)
        # Print fancy results
        print("-"*55)
        print(f"Results for \"{args.format + t}:\"")
        print("-"*55)
        for i in range(args.results):
            print(f"{os.path.basename(a[i][0])[:min(len(a[i][0]), 27)]:29s} {i:03d} | similarity {a[i][1]*100:.3f}%")
        print("-"*55)
        folder = f"{args.copy_folder}/" + t.replace(".", "_")
        if args.copy:
            image_copy(a, folder)

    end = timer()
    print(f"Processing time: {end-start:.3f} seconds")
