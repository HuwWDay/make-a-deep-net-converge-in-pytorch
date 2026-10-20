"""
Make a Deep Net Converge in PyTorch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Make a deep net converge in PyTorch (Hands-On ML, chapter 11).

Story: a ten-layer sigmoid network with default initialization barely learns and
its first layer receives almost no gradient. Fix it one change at a time on real
Fashion-MNIST data: He init, ReLU, batch norm, dropout, a better optimizer, a
one-cycle schedule, gradient clipping. Then transfer the pretrained body to a new
two-class task, fine-tune, and save the result with its configuration.
"""
import os
import tempfile
import torch
import torch.nn as nn


def main() -> None:
    loaders = fashion_loaders(n_train=5000, n_val=1000)
    xb, yb = next(iter(loaders["train"]))

    # ---- 1. Diagnosis ----
    torch.manual_seed(0)
    sig = gradient_norms(DeepNet(), xb, yb)
    torch.manual_seed(0)
    fixed = gradient_norms(apply_he_init(DeepNet(activation="relu", batchnorm=True)), xb, yb)
    print(f"gradient norm, first layer / last layer: sigmoid+default {sig[0] / sig[-1]:.1e}   relu+He+BN {fixed[0] / fixed[-1]:.2f}")

    # ---- 2. Architecture fixes ----
    configs = {
        "10x sigmoid, default init": {},
        "10x relu, He init": {"activation": "relu", "he_init": True},
        "10x relu, He init, batchnorm": {"activation": "relu", "he_init": True, "batchnorm": True},
    }
    for name, acc in compare_configs(loaders, configs, epochs=2, lr=0.1).items():
        print(f"  {name:32s} val acc {acc:.3f}")
    d = dropout_effect(loaders, rate=0.5, epochs=3)
    print(f"dropout 0.5 on a 3-layer net: train acc {d['no_dropout']['train_acc']:.3f} -> {d['dropout']['train_acc']:.3f}, "
          f"val acc {d['no_dropout']['val_acc']:.3f} -> {d['dropout']['val_acc']:.3f}")

    # ---- 3. Optimization fixes ----
    opts = compare_optimizers(loaders, ["sgd", "momentum", "nesterov", "adam", "adamw"], epochs=2, lr=0.01)
    print("optimizers at lr 0.01, 2 epochs: " + "  ".join(f"{k} {v:.3f}" for k, v in opts.items()))
    oc = one_cycle(loaders, max_lr=0.1, epochs=2)
    print(f"one-cycle (lr {oc['lr_start']:.4f} -> {oc['lr_peak']:.2f} -> {oc['lr_end']:.6f}): val acc {oc['val_acc']:.3f}")
    torch.manual_seed(0)
    c = clipping_effect(DeepNet(activation="relu", n_layers=3), xb, yb, max_norm=1.0)
    print(f"gradient clipping at 1.0: total norm {c['before']:.2f} -> {c['after']:.2f}")

    # ---- 4. Transfer learning ----
    tasks = split_by_class(loaders, held_out=(5, 7))
    r = transfer_experiment(tasks, epochs=2)
    print(f"\nsandal vs sneaker: from scratch {r['scratch']:.3f}   frozen transfer {r['frozen']:.3f}   fine-tuned {r['finetuned']:.3f}")

    # ---- 5. Ship ----
    cfg = dict(activation="relu", batchnorm=True, n_layers=3, n_classes=2)
    torch.manual_seed(0)
    final = apply_he_init(DeepNet(**cfg))
    train_epochs(final, tasks["B"], make_optimizer(final, "momentum", 0.05), epochs=2)
    path = os.path.join(tempfile.gettempdir(), "deepnet_sandal_sneaker.pt")
    save_deepnet(final, cfg, path)
    served = load_deepnet(path)
    xv, yv = next(iter(tasks["B"]["val"]))
    with torch.no_grad():
        acc = float((served(xv).argmax(1) == yv).float().mean())
    print(f"saved and reloaded: {acc:.3f} accuracy on the first validation batch, eval mode = {not served.training}")


if __name__ == "__main__":
    main()

