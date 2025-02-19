import json, os
from matplotlib import pyplot as plt
import numpy as np


def draw_loss(ckpt_path):
    loss_fn = os.path.join(ckpt_path, "loss_list.json")
    with open(loss_fn, "r") as f:
        l = json.load(f)
        
    plt.clf()
    x, y = [], []
    n_points = min(100, len(l))  # 图上保留几个点
    step = len(l) // n_points 
    for i in range(0, len(l), step):
        x.append(i + 1)
        y.append(np.mean(l[i:i + step]))

    # 绘制loss图片
    plt.title("train loss")
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.plot(x, y)

    fn = os.path.join(ckpt_path, "loss_img.png")
    # print("save loss picture at:", fn)
    plt.savefig(fn)
   


if __name__ == "__main__":
    # 读取loss_list
    ckpt_path = ""
    draw_loss(ckpt_path)