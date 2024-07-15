from scipy.ndimage import label
import matplotlib.pyplot as plt
import numpy as np
import matplotlib



def periodiclabel(im,periodic0=False,periodic1=False):
    full_size = np.zeros((im.shape[0]+periodic0,im.shape[1]+periodic1))
    full_size[0:im.shape[0],0:im.shape[1]] = im
    if periodic0:
        full_size[-1,:] = full_size[0,:]
    if periodic1:
        full_size[:,-1] = full_size[:,0]

    full_size_label = label(full_size)[0]

    if periodic0:

        wraprow = full_size_label[-1,:]
        originrow = full_size_label[0,:]
        for i in range(len(wraprow)):
            if wraprow[i]!=0:
                full_size_label[full_size_label==wraprow[i]] = originrow[i]
    if periodic1:
        wraprow = full_size_label[:,-1]
        originrow = full_size_label[:,0]
        for i in range(len(wraprow)):
            if wraprow[i]!=0:
                full_size_label[full_size_label==wraprow[i]] = originrow[i]
    output = full_size_label[:im.shape[0],:im.shape[1]]
    output = np.unique(output, return_inverse=True)[1].reshape(output.shape)
    if ~(output[full_size_label[:im.shape[0],:im.shape[1]] == 0] ==0).all():
        print("BROKEN")
        exit()
    return output,np.max(output)

test_this_out = False
if test_this_out:
    testcase = np.zeros((100,100))
    testcase[0:15,0:15] = 1
    testcase[0:15,-15:] = 1
    testcase[-15:,:] = 1
    testcase[35:50,0:15] = 1
    testcase[35:50,-15:] = 1

    periodic_label(testcase,True,False)

    cmap = matplotlib.colors.ListedColormap ( np.random.rand ( 256,3))

    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
    ax1.imshow(label(testcase)[0],cmap=cmap)
    ax1.set_title("standard label")
    ax2.imshow(periodic_label(testcase,True,False)[0],cmap=cmap)
    ax2.set_title("peridioc 0")
    ax3.imshow(periodic_label(testcase,False,True)[0],cmap=cmap)
    ax3.set_title("periodic 1")
    ax4.imshow(periodic_label(testcase,True,True)[0],cmap=cmap)
    ax4.set_title("doubly periodic")
    plt.show()
