from searcher import CLIPSearcher
from utils import get_args

if __name__ == "__main__":
    args = get_args()
    cs = CLIPSearcher(device=args.device)
    cs.load_dir(args.dir, save_every=args.save_every, recursive=args.recursive)
    cs.search(texts=args.texts, images=args.images, results=args.results, outdir=args.outdir)
