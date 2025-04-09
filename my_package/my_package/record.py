import matplotlib.pyplot as plt
import os
import pickle
from typing import List, Literal
import argparse

class Record():
    def __init__(self,
                 path:str,
                 keys:List[str],
                 curve_paras:List[List[str]] = [],
                 args:Literal[None, argparse.Namespace] = None, # type: ignore
                 save_model_epoch = 0
                 ):
        assert not os.path.exists(path), "path already exist!"
        os.makedirs(os.path.join(path, "model"))
        self.data = {key:[] for key in keys}
        self.path = path
        self.curve_paras = []
        self.save_model_epoch = save_model_epoch

        if len(curve_paras) == 0:
            curve_paras = [[key] for key in keys]
        self.curve_paras = curve_paras

        self.epoch = 0

        if args is not None:
            with open(os.path.join(path, "paras.txt"), "w", encoding="utf-8") as f:
                for arg in vars(args):
                    f.write(f"{arg} = {getattr(args, arg)}\n")

    def record(self, *args):
        assert len(self.data) == len(args)
        for key, value in zip(self.data.keys(), args):
            if isinstance(value, int):
                self.data[key].append(value)
            else:
                self.data[key].append(float(value))
        self.write_csv()
        self.draw_curves()

    def draw_curves(self):
        for valid_key in self.curve_paras:
            title = "_".join(valid_key)
            self.draw_curve("epoch", "value", title, valid_key)

    def write_csv(self):
        with open(os.path.join(self.path, "record.csv"), "w") as f:
            f.write(", ".join(self.data.keys()) + "\n")
            for datas in zip(*self.data.values()):
                f.write(", ".join([str(i) for i in datas]) + "\n")

    def save_model(self, model, device = "cpu"):
        if self.epoch >= self.save_model_epoch:
            model.to("cpu")
            with open(os.path.join(self.path, "model", f"{self.epoch}.pickle"), "wb") as f:
                pickle.dump(model, f)
            model.to(device)
        self.epoch += 1

    def save_models(self, models, device = "cpu"):
        if self.epoch >= self.save_model_epoch:
            for model in models: model.to("cpu")
            with open(os.path.join(self.path, "model", f"{self.epoch}.pickle"), "wb") as f:
                pickle.dump(models, f)
            for model in models: model.to(device)
        self.epoch += 1

    def draw_curve(self, x_label=None, y_label=None, title="curve", valid_key=None):
        plt.figure()
        if valid_key is None:
            valid_key = self.data.keys()
        for key in valid_key:
            plt.plot(self.data[key], label=key)
        if title is not None: plt.title(title)
        if x_label is not None: plt.xlabel(x_label)
        if y_label is not None: plt.ylabel(y_label)
        plt.legend()
        plt.savefig(os.path.join(self.path, title+".png"))
        plt.close()