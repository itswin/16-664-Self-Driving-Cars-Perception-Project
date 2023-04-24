import os

def classify(subdir, file):
    guid = subdir.split('/')[-1]
    image = file.split('_')[0]
    label = 1

    return guid, image, label

rootdir = './'

n = 0

with open('labels.csv', 'w') as f:
    print(f"""guid/image,label""", file=f)

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            #print os.path.join(subdir, file)
            filepath = subdir + os.sep + file

            if filepath.endswith(".jpg"):
                guid, image, label = classify(subdir, file)
                print(f"""{guid}/{image},{label}""", file=f)
                n += 1

print(f"Processed {n} images.")
