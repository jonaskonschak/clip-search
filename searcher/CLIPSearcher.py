import os
import shutil
from uuid import uuid4
import torch
from PIL import Image
import clip
from utils import *


class CLIPSearcher:
    def __init__(self,
                device:str="cpu",
                model_name:str="ViT-B/32",
                store_path="./stored",
                exts:tuple=("jpg", "jpeg", "jfif", "png")):
        print(f"Using Device {device}")
        self.device = torch.device(device)
        self.model_name = model_name
        self.store_path = store_path
        self.exts = exts
        self.active_path = "."
        self.active_list = []
        self.active_dict = dict()
        self.active_features = None
        os.makedirs(self.store_path, exist_ok=True)
        self.dirs = load_json(os.path.join(self.store_path, "dirs.json"), warn=True)
        self.load_model(self.model_name)

    def load_model(self, model_name):
        print(f"Loading CLIP model {model_name}...", end=" ")
        self.model_name = model_name
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        print("Done.")

    @torch.inference_mode()
    def load_dir(self, path, save_every=1000, recursive:bool=True, load_new:bool=True):
        print(f"Loading dir {path}.")
        path = os.path.normcase(path)
        self.active_path = path
        if path in self.dirs.keys():
            uuid = self.dirs[path]
        else:
            uuid = str(uuid4())
            self.dirs[path] = uuid
        
        print(f"uuid={uuid}.")
        save_json(self.dirs, os.path.join(self.store_path, "dirs.json"))

        store_file = os.path.join(self.store_path, uuid + ".pt")
        if os.path.isfile(store_file):
            try:
                self.active_dict = torch.load(store_file, map_location=self.device)
                print(f"Loaded existing dict:{store_file} with {len(self.active_dict)} images.")
            except FileNotFoundError:
                self.active_dict = dict()
                print("No existing dict found. Creating new dict.")
        else:
            self.active_dict = dict()
        
        filepaths = []
        if load_new:
            for root, _, filenames in os.walk(path):
                for filename in filenames:
                    if filename.endswith(self.exts):
                        filepath = os.path.join(root, filename)
                        filepath = os.path.normcase(filepath)
                        filepaths.append(filepath)
                if not recursive:
                    break
            
            i = 0
            print("Loading new images.")
            for filepath in filepaths:
                if filepath not in self.active_dict.keys():
                    try:
                        self.active_dict[filepath] = self.load_features(filepath)
                        i += 1
                    except Exception as e:
                        print(e)
                        pass
                    if i % save_every == 0 and i != 0:
                        print(f"Loaded {i} new images.")
                        self.save_dict(store_file)
            
            print(f"Done. {i} new images, {len(self.active_dict)} total.")
            self.save_dict(store_file)
        
        self.active_list = list(self.active_dict.keys())
        self.active_features = torch.cat(tuple(self.active_dict.values())).to(torch.half if self.device.type == "cuda" else torch.float)

    @torch.inference_mode()
    def load_features(self, target, text=False):
        if text:
            text = clip.tokenize(target).to(self.device)
            features = self.model.encode_text(text)
            features /= features.norm(dim=-1, keepdim=True)
        else:
            image = Image.open(target)
            image = self.preprocess(image).to(self.device).unsqueeze(0)
            features = self.model.encode_image(image)
            features /= features.norm(dim=-1, keepdim=True)
        return features

    def save_dict(self, filepath):
        torch.save(self.active_dict, filepath)
    
    @torch.inference_mode()
    def search(self, texts:list=None, images:list=None, results=5, print_results=True, outdir=None):
        assert texts or images, "Search failed: Either texts or images need to be specified."
        targets = []
        targets_features_list = []
        if texts:
            targets += texts
            targets_features_list += [self.load_features(text, text=True) for text in texts]
        if images:
            targets += images
            targets_features_list += [self.load_features(image) for image in images]
        
        targets_features = torch.cat(targets_features_list)
        similarity = targets_features @ self.active_features.T
        (topsim, topidx) = torch.topk(similarity, results, dim=-1)
        if print_results:
            self.print_results(targets, topsim, topidx)
        if outdir:
            self.copy(targets, topidx, outdir)
        return targets, topsim, topidx

    def print_results(self, targets, topsim, topidx):
        print("-"*50)
        for t, target in enumerate(targets):
            results = [self.active_list[idx] for idx in topidx[t].tolist()]
            similarities = topsim[t]
            print(f"Results for \"{target}\":")
            print("-"*50)
            for r, result in enumerate(results):
                print(f"{r:02d} | {100.0*similarities[r]:.2f}% | {result}")
            print("-"*50)
    
    def copy(self, targets, topidx, outdir):
        for t, target in enumerate(targets):
            target_outdir = os.path.join(outdir, safe_name(target))
            os.makedirs(target_outdir, exist_ok=True)
            results = [self.active_list[idx] for idx in topidx[t].tolist()]
            for result in results:
                result_path = os.path.join(target_outdir, os.path.basename(result))
                try:
                    shutil.copy(result, result_path)
                except:
                    print(f"Could not copy {result}")
    
    def __repr__(self):
        return f"CLIPSearcher(model={self.model_name}, device={self.device})"