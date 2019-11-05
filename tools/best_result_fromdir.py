import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dir",
                    dest="dir",
                    default="./predictions",
                    type=str)
parser.add_argument("--ignore_crop_aug",
                    '-i',
                    dest="ignore_crop_aug",
                    type=int,
                    default=1)
parser.add_argument("--model",
                    dest="model",
                    type=str,
                    default='')
args = parser.parse_args()
root = args.dir

db_name = ["AwA2", "CUB", "SUN"]
db_split = ["SS", "PS"]
rst = {
        "AwA2": {"SS": [-1, ""], "PS": [-1, ""], "PS_G": [-1, -1, -1, ""]},
        "CUB": {"SS": [-1, ""], "PS": [-1, ""], "PS_G": [-1, -1, -1, ""]},
        "SUN": {"SS": [-1, ""], "PS": [-1, ""], "PS_G": [-1, -1, -1, ""]}
        }


ignore_filename_words = ["FiveCrop", "TenCrop", "TenCrop+Resize"] if args.ignore_crop_aug else []
isignore = lambda x: len([w for w in ignore_filename_words if w in x]) != 0
if args.model:
    all_files = [os.path.join(root, f) for f in os.listdir(root) if ".txt" in f and not isignore(f) and args.model in f]
else:
    all_files = [os.path.join(root, f) for f in os.listdir(root) if ".txt" in f and not isignore(f)]

def resultfromtxt(path):
    acc = -1
    tr  = -1
    ts  = -1
    H   = -1
    epoch = -1
    with open(path) as fp:
        lines = fp.readlines()
    for i in range(len(lines) // 2):
        e = lines[2 * i].strip()
        e = int(e[e.rfind('_') + 1: e.find('.pth')])
        a = lines[2 * i + 1].strip()
        if "acc" in a:
            a = float(a[a.find('acc: ') + 5:])
            if a > acc:
                acc = a
                epoch = e
        elif "H" in a:
            a = a.split(",")
            r = a[0]
            r = float(r[r.find(':') + 1:])
            s = a[1]
            s = float(s[s.find(':') + 1:])
            h = a[2]
            h = float(h[h.find(':') + 1:])
            if h > H:
                tr = r
                ts = s
                H = h
                epoch = e
    if acc != -1:
        return acc, epoch
    else:
        return [tr, ts, H], epoch

for f in all_files:
    result, epoch = resultfromtxt(f)
    for db in db_name:
        for split in db_split:
            if db in f and split in f:
                if isinstance(result, list):
                    if result[2] > rst[db]["PS_G"][2]:
                        rst[db]["PS_G"][0] = result[0]
                        rst[db]["PS_G"][1] = result[1]
                        rst[db]["PS_G"][2] = result[2]
                        rst[db]["PS_G"][3] = "{}/{}".format(os.path.basename(f), epoch)
                else:
                    if result > rst[db][split][0]:
                        rst[db][split][0] = result
                        rst[db][split][1] = "{}/{}".format(os.path.basename(f), epoch)

for db in db_name:
    for split in db_split:
        if rst[db][split][0] != -1:
            print("\t[{} {}]: {}\t ---- {}".format(db, split, rst[db][split][0], rst[db][split][1]))

for db in db_name:
    if rst[db]["PS_G"][2] != -1:
        print("\t[{} PS G]: {}, {}, {}\t ---- {}".format(db, rst[db]["PS_G"][0], rst[db]["PS_G"][1], rst[db]["PS_G"][2], rst[db]["PS_G"][3]))
