from glob import glob
import os

def rem():
    # path = '/home/Flask/static/color/'
    # path1 = '/home/Flask/static/cnn/*.*'
    # path2 = '/home/Flask/static/inc/*.*'
    # path3 = '/home/Flask/static/xce/*.*'
    # trial = '/home/Flask/stati/colo/'
    # image = glob(path)
    # cnn = glob(path1)
    # inc = glob(path2)
    # xce = glob(path3)
    for j in os.listdir("static/color/"):
        os.remove("static/color/" + j)
    for j in os.listdir("static/cnn/"):
        os.remove("static/cnn/" + j)
    for j in os.listdir("static/inc/"):
        os.remove("static/inc/" + j)
    for j in os.listdir("static/xce/"):
        os.remove("static/xce/" + j)
