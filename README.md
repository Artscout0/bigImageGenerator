# Big Image Generator

A personal project with some friends to generate as big of an image (in terms of file size and pixel size)

## How it works

Using a simple python script, it determines the size of the image, and runs a script on the GPU using GLSL to create that image.
It then saves the image as a .tiff, which has the advantage of not being compressed.

## Why?

I was bored one day, and kinda wanted to jokingly mildly sabotage a friend's project, for which a sufficiently large image should suffice.

A couple of days later, I found myself texting with said friend about how to make this project work, optimise it, and with another friend with better hardware about how he should run it.