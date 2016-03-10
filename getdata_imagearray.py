import math
import numpy
import random
def getdata_imagearray(IMAGES, winsize, num_patches):

    num_images = IMAGES.shape[2]
    image_size = IMAGES.shape[0]
    sz = winsize
    BUFF = 4
  
    totalsamples = 0
    # extract subimages at random from this image to make data vector X
    # Step through the images

    X = numpy.zeros((winsize*winsize, num_patches))

    for i in range (0 , num_images):

        # Display progress
        print '[%d/%d]' % (i,num_images)

        this_image = IMAGES[:,:,i]

        # Determine how many patches to take
        getsample = num_patches/num_images
        if i == num_images - 1:
            getsample = num_patches-totalsamples

        # Extract patches at random from this image to make data vector X
        for j in range(0,getsample):
            r = BUFF + math.ceil((image_size - sz - 2 * BUFF) * random.random())
            c = BUFF + math.ceil((image_size - sz - 2 * BUFF) * random.random())
            totalsamples = totalsamples + 1
            # X(:,totalsamples)=reshape(this_image(r:r+sz-1,c:c+sz-1),sz^2,1);
            temp = this_image[r-1:r+sz-1, c-1:c+sz-1]
            temp = temp.reshape(sz*sz);
            X[:,totalsamples - 1] = temp - numpy.mean(temp);
    print '\n'
    return X
