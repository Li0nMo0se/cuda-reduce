#!python

import subprocess
import os
import argparse

sources = ["main.cu"] # add sources
build_dir = "./build"
program_name = "reduce_example"

class Config:
    debug = True

def parse_args(config):
    parser = argparse.ArgumentParser(description='Build System')
    parser.add_argument('-d', '--debug', default=True, help='Debug', action='store_true')
    parser.add_argument('-r', '--release', default=False, help='Release', action='store_true')
    args = parser.parse_args()
    config.debug = args.debug
    config.debug = not args.release

def compile(config):
    if not os.path.isdir(build_dir):
        os.mkdir(build_dir)

    cmd = ["nvcc"] +  sources + ["-o", build_dir + "/" + program_name]
    cmd += ["--extended-lambda"] # for extended lamba __device__
    if config.debug:
        cmd += ["-g"]
    else:
        cmd += ["-O2"]

    print(cmd)
    subprocess.call(cmd)

if __name__ == "__main__":
    config = Config()
    parse_args(config) # Get the build configuration
    compile(config)