#!/bin/bash

cd recipe/build

anaconda login
anaconda upload --user probablyai linux-64/*.tar.bz2
anaconda upload --user probablyai osx-64/*.tar.bz2
anaconda upload --user probablyai win-64/*.tar.bz2
