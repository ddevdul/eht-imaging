import os
import argparse
from pprint import pprint

def main(site_packages_path: str):
    eht_imaging_path: str = os.getcwd()
    print(eht_imaging_path)
    module_directories: list[str] = ["ehtim", "examples"]
    module_full_paths: list[str] = list()
    for directory in module_directories:
        directory_path: str = "/".join([eht_imaging_path, directory])
        sub_dirs: list[str] = os.listdir(directory_path)
        modules: list[str] = ["/".join([eht_imaging_path, directory, sub_dir]) for sub_dir in sub_dirs if sub_dir != "__pycache__" and "." not in sub_dir]
        module_full_paths.append(directory_path)
        module_full_paths.extend(modules)
    pprint(module_full_paths)
    # create the pth files in the site packages folder
    pth_file_name = "ehtim_project_import_paths.pth"
    pth_file_name = os.path.join(site_packages_path, pth_file_name)
    f = open(pth_file_name, "w")
    for path_ in module_full_paths:
        f.write(path_ + "\n")
    f.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--site_packages_path', type=str, required=True)
    main(site_packages_path=parser.parse_args().site_packages_path)