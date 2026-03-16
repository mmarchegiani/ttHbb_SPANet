import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--link", type=str, required=True, help="Input path with list of existing symbolic links")
parser.add_argument("--new", type=str, required=True, help="Input folder containing new folders to link to")

args = parser.parse_args()

# List of existing symbolic links
with open(args.link, 'r') as f:
    existing_links = f.read().splitlines()

# Iterate over the existing links and new folders
for i, link in enumerate(existing_links):
    # Unlink the existing symbolic link
    os.unlink(link)

    # Create a new symbolic link pointing to the new folder assuming that the new folder has the same name as the existing symbolic link
    new_folder = os.path.join(args.new, link)
    os.symlink(new_folder, link)
    print(f"Linked '{link}' to '{new_folder}'")