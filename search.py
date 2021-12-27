from searcher import CLIPSearcher
from utils import get_args

if __name__ == "__main__":
    args = get_args()
    print(args)
    cs = CLIPSearcher(device=args.device, store_path=args.store_path)
    cs.load_dir(args.dir, save_every=args.save_every, recursive=args.recursive, load_new=(not args.dont_load_new))
    cs.search(texts=args.texts, images=args.images, results=args.results, outdir=args.outdir)
