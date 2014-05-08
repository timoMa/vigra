import vigra
from vigra import numpy
from matplotlib import pylab
from time import time


path = "100075.jpg"

data = vigra.impex.readImage(path).astype(numpy.float32)

t0 =time()
res = vigra.filters.nonLocalMean2d(data, sigma=10.0,searchRadius=10,patchRadius=2,nThreads=3,stepSize=2,verbose=True,sigmaMean=30.0)
t1 = time()
print t1-t0
imgs  = [data,res]

for img in imgs:
    for c in range(3):
        cimg=img[:,:,c]
        cimg-=cimg.min()
        cimg/=cimg.max()

f = pylab.figure()
for n, arr in enumerate(imgs):

    f.add_subplot(1, len(imgs), n)
    pylab.imshow(arr.swapaxes(0,1))

pylab.title('denoised')
pylab.show()
