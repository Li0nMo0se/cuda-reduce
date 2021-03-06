#!python

import subprocess
import os
import argparse

sources = ["main.cu"] # add sources
build_dir = "./build"
program_name = "reduce_example"

class Config:
    debug = True
    compute = 75

def parse_args(config):
    parser = argparse.ArgumentParser(description='Build System')
    parser.add_argument('-d', '--debug', default=True, help='Debug', action='store_true')
    parser.add_argument('-r', '--release', default=False, help='Release', action='store_true')
    parser.add_argument('--compute', help='GPU Compute capability')
    args = parser.parse_args()
    config.debug = args.debug
    config.debug = not args.release
    if args.compute:
        config.compute = args.compute

def compile(config):
    if not os.path.isdir(build_dir):
        os.mkdir(build_dir)

    cmd = ["nvcc"] +  sources
    cmd += ["--extended-lambda"] # for extended lamba __device__
    cmd += ["-o", build_dir + "/" + program_name] # output
    cmd += ["-arch=compute_" + str(config.compute), "-code=sm_" + str(config.compute)]
    if config.debug:
        cmd += ["-g", "-G"] # debug and device-debug. Define __CUDACC_DEBUG__
    else:
        cmd += ["-O2"]

    print(cmd)
    subprocess.call(cmd)

if __name__ == "__main__":
    config = Config()
    parse_args(config) # Get the build configuration
    compile(config)