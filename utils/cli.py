import argparse
import os
from searcher import CLIPSearcher

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",
                        "--dir",
                        required=True)
    
    parser.add_argument("-sp",
                        "--store_path",
                        type=str,
                        default=os.path.join(os.path.expanduser("~"), "/clip_search"))

    parser.add_argument("-t",
                        "--texts",
                        nargs="+",
                        default=None)
    
    parser.add_argument("-i",
                        "--images",
                        nargs="+",
                        default=None)

    parser.add_argument("-r",
                        "--results",
                        type=int,
                        default=5)

    parser.add_argument("-o",
                        "--outdir",
                        default=None)

    parser.add_argument("-se",
                        "--save_every",
                        type=int,
                        default=1000)
    
    parser.add_argument("-de",
                        "--device",
                        type=str,
                        default="cuda")

    parser.add_argument("-rc",
                        "--recursive",
                        action="store_true")
    
    parser.add_argument("-dln",
                        "--dont_load_new",
                        action="store_true")

    return parser.parse_args()

def main():
    args = get_args()
    cs = CLIPSearcher(device=args.device, store_path=args.store_path)
    cs.load_dir(args.dir, save_every=args.save_every, recursive=args.recursive, load_new=(not args.dont_load_new))
    cs.search(texts=args.texts, images=args.images, results=args.results, outdir=args.outdir)
