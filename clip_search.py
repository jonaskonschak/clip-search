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
        shutil.copy(f"{args.folder}/{a_list[result][0]}", copy_folder + "/")


def save_dict(dict_to_save, filename):
    torch.save(dict_to_save, filename)


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


if __name__ == "__main__":
    start = timer()
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

    model, preprocess = clip.load("ViT-B/32", device=device)
    print(f"Using Device: {device}")
    sim_func = torch.nn.CosineSimilarity()

    with torch.inference_mode():
        targets = get_targets()
        image_dict = load_dict(args.dict)
        target_probs_dict = dict()
        for target in targets.keys():
            target_probs_dict[target] = dict()
        counter = 0
        new_counter = 0
        for idx, file in enumerate(os.listdir(args.folder)):
            if file.lower().endswith(exts):
                short_filename = file[:min(len(file), 20)]
                if file in image_dict:
                    image_features = image_dict[file].to(device)
                else:
                    # Load image file and get features
                    image = preprocess(Image.open(f"{args.folder}/{file}")).unsqueeze(0).to(device)
                    image_features = model.encode_image(image)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    image_dict[file] = image_features
                    new_counter += 1
                counter += 1
                if new_counter % args.save_every == 0 and new_counter != 0:
                    save_dict(image_dict, args.dict)
                if counter % args.save_every == 0:
                    print(f"Loaded {counter} images. {new_counter} new")
                for (t, f) in targets.items():
                    # Get similarity of image and each target
                    target_probs_dict[t][file] = sim_func(image_features, f).item()
        print(f"Loaded {counter} images. {new_counter} new  | Finished")

    if new_counter != 0:
        save_dict(image_dict, args.dict)
    for t in target_probs_dict.keys():
        # Sort for highest similarity
        heap = heapq.nlargest(args.results, target_probs_dict[t], key=target_probs_dict[t].get)
        a = sorted({image: target_probs_dict[t][image] for image in heap}.items(), key=lambda x: x[1], reverse=True)
        # Print fancy results
        print("-"*55)
        print(f"Results for \"{args.format + t}:\"")
        print("-"*55)
        for i in range(args.results):
            print(f"{a[i][0][:min(len(file), 27)]:29s} {i:03d} | similarity {a[i][1]*100:.3f}%")
        print("-"*55)
        folder = f"{args.copy_folder}/" + t.replace(".", "_")
        if args.copy:
            image_copy(a, folder)

    end = timer()
    print(f"Processing time: {end-start:.3f} seconds")
