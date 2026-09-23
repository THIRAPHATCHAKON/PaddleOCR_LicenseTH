import os

#Path Input
folder_path = r""

files = [
    f for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f))
]

files.sort()

for i, filename in enumerate(files, start=1):
    old_path = os.path.join(folder_path, filename)

    ext = os.path.splitext(filename)[1]

    new_name = f"{i:04d}{ext}"
    new_path = os.path.join(folder_path, new_name)

    os.rename(old_path, new_path)
    print(f"{filename} -> {new_name}")

print("Ok!")