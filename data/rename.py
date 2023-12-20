import os
# your 1h experiment folder
rootdir = r"/Users/bunkyotop/Documents/20220707 input ML Bingnan/Exp76 conditions with swelling organoids"
for filename in os.listdir(rootdir):
    if "t0" in filename:
        filepath = os.path.join(rootdir, filename)
        newfilepath = os.path.join(rootdir, filename.replace("t0", "t00"))
        os.rename(filepath, newfilepath)
    elif "t1" in filename:
        filepath = os.path.join(rootdir, filename)
        newfilepath = os.path.join(rootdir, filename.replace("t1", "t02"))
        os.rename(filepath, newfilepath)
    elif "t2" in filename:
        filepath = os.path.join(rootdir, filename)
        newfilepath = os.path.join(rootdir, filename.replace("t2", "t04"))
        os.rename(filepath, newfilepath)
    elif "t3" in filename:
        filepath = os.path.join(rootdir, filename)
        newfilepath = os.path.join(rootdir, filename.replace("t3", "t06"))
        os.rename(filepath, newfilepath)
    elif "t4" in filename:
        filepath = os.path.join(rootdir, filename)
        newfilepath = os.path.join(rootdir, filename.replace("t4", "t08"))
        os.rename(filepath, newfilepath)
    elif "t5" in filename:
        filepath = os.path.join(rootdir, filename)
        newfilepath = os.path.join(rootdir, filename.replace("t5", "t10"))
        os.rename(filepath, newfilepath)
    elif "t6" in filename:
        filepath = os.path.join(rootdir, filename)
        newfilepath = os.path.join(rootdir, filename.replace("t6", "t12"))
        os.rename(filepath, newfilepath)
print("files are renamed.")
