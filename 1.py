# -*- coding: utf-8 -*-

# PI
from __future__ import print_function
import os, sys
from PIL import Image

for infile in sys.argv[1:]:
	im = Image.open(infile)
#	im.show()
	f, e = os.path.splitext(infile)
	outfile = f + ".bmp"
	if infile != outfile:
		try:
			im.save(outfile)
		except IOError:
			print("Cannot convert " + infile + ".")
	else:
		print('Don\'t need convert.')
	print(im.format, im.size, im.mode)
# box = (100, 100, 1000, 1000)
# opregion = im.crop(box)
# opregion = opregion.transpose(Image.ROTATE_180)
# im.paste(opregion, box)
# 在这个里面为了保证快，所以paste指令执行之前，crop指令只是一个标识。如果这一块的crop的区域之前修改过就哦吼完蛋。用load()来先载入进去可以解决这个问题。
seq = list(im.getdata())
i = 0
for pixel in seq:
    r, g, b = pixel
    l = int(r * 0.299 + g * 0.587 + b * 0.114)
    if l > 79:
        l = 255
    else:
        l =0
    seq[i] = (l, l, l)
    i = i + 1
im.putdata(seq)
im.show()
im.save("output.bmp")
# im.putdata(im.getdata(), 1.5, 1000)