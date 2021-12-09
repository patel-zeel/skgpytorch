import os

# Manually modify following parameters to customize the structure of your project
path = os.path.abspath(os.path.dirname(__file__)).split("/")
# print(path)
REPO_HOME_PATH = "/".join(path[:-1])
REPO_NAME = path[-1]
PACKAGE_NAME = REPO_NAME
VERSION = "0.1.0"
AUTHOR = "Zeel B Patel"
AUTHOR_EMAIL = "patel_zeel@iitgn.ac.in"
description = "example description"
URL = "https://github.com/patel-zeel/" + REPO_NAME
LICENSE = "MIT"
LICENSE_FILE = "LICENSE"
LONG_DESCRIPTION = "file: README.md"
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"

full_path = os.path.join(REPO_HOME_PATH, REPO_NAME)

# Modify setup.cfg

with open(os.path.join(full_path, "setup.cfg"), "w") as f:
    f.write("[metadata]\n")
    f.write("name = " + PACKAGE_NAME + "\n")
    f.write("version = " + VERSION + "\n")
    f.write("author = " + AUTHOR + "\n")
    f.write("author-email = " + AUTHOR_EMAIL + "\n")
    f.write("description = " + description + "\n")
    f.write("url = " + URL + "\n")
    f.write("license = " + LICENSE + "\n")
    f.write("long_description_content_type = " + LONG_DESCRIPTION_CONTENT_TYPE + "\n")
    f.write("long_description = " + LONG_DESCRIPTION + "\n")

# Modify CI

with open(os.path.join(full_path, ".github/workflows/CI.template"), "r") as f:
    content = f.read()

with open(os.path.join(full_path, ".github/workflows/CI.yml"), "w") as f:
    content = content.replace("<reponame>", REPO_NAME)
    f.write(content)

print("Successful")
