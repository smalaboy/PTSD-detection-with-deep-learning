from genericpath import isfile
import re
import os
def rename(path):
  chuncks = path.split(os.extsep)
  ext = chuncks[-1]
  name = ''.join(chuncks[:-1])
  # name = re.sub(r'\W+', '', name)
  name = re.sub(r'[^a-zA-Z0-9/]', '', name)
  return name[0:min(30, len(name))].replace(' ', '') + os.extsep + ext

def rename_and_list_files_rec(path):
  files = os.listdir(path)
  new_names = [os.path.abspath(os.path.join(path, rename(f))) for f in files if os.path.isfile(os.path.abspath(os.path.join(path, f)))]
  files = [os.path.abspath(os.path.join(path, f)) for f in files]
  dirs = [f for f in files if os.path.isdir(f)]
  if not dirs:
    return list(zip(files, new_names))
  files = [f for f in files if os.path.isfile(f)]
  out = list(zip(files, new_names))
  for dir in dirs:
    out += rename_and_list_files_rec(dir)
  return out

def rename_files(names_zip, verbose=True):
  for old_n, new_n in names_zip:
    if verbose:
      print("Renaming", old_n, "to", new_n)
    os.rename(old_n, new_n)

names_zip = rename_and_list_files_rec('/raid/home/labusermoctar/ptsd_dataset/videos')

rename_files(names_zip, verbose=True)

with open('./renamed_list.txt', 'w') as f:
  for a, b in names_zip:
    f.write(f"{a}\t{b}\n")
