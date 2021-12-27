import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",
                        "--dir",
                        required=True)

    parser.add_argument("-t",
                        "--texts",
                        type=list,
                        default=None)
    
    parser.add_argument("-i",
                        "--images",
                        type=list,
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

    return parser.parse_args()