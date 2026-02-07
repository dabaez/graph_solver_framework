#!/bin/bash

sudo rm -rf build
cmake -B build
sudo cmake --build build --target install --config Release