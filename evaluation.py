from conformer import ExP, Conformer
import torch
import torch.nn as nn

config = {
    'nSub': 1
}


def main():
    model = Conformer(config["nSub"])
    model.eval()
    
    exp = ExP(config["nSub"])
    


if __name__ == "__main__":
    main()
