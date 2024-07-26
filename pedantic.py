import numpy as np
import xarray as xr
from tqdm import tqdm
from scipy.ndimage import label
from periodiclabel import periodiclabel
import pickle
import matplotlib.pyplot as plt
import matplotlib
from treelib import Node, Tree
from matplotlib.colors import Normalize
from functools import partial
from matplotlib.widgets import TextBox
import networkx as nx
from periodiclabel import periodiclabel

def testcase(Nx,Ny,mode="complex"):
    X,Y = np.meshgrid(np.asarray(range(Nx))*2*np.pi+10**-2,np.asarray(range(Ny))*2*np.pi+10**-2)
    if mode=="complex":
        return 100*np.cos(0.01*X)*np.sin(0.01*Y) + 100*np.cos(0.03*X)*np.sin(0.03*Y)
    elif mode=="simple":
        return (Y-10)**2 - (X-10)**2


# Identify critical points
# N critical points -> 2N slices

# You find the gradient in all directions
# If they are all the same sign, its a minimum or maximum
# 

def contour_tree(z,plot_crit=True):
    plusi = z - np.roll(z,-1,axis=0)
    plusj = z - np.roll(z,-1,axis=1)
    minusi = z - np.roll(z,1,axis=0)
    minusj = z - np.roll(z,1,axis=1)

    plusiplusj = z- np.roll(z,(-1,-1),axis=(0,1))
    minusiminusj = z - np.roll(z,(1,1),axis=(0,1))

    minusiplusj = z - np.roll(z,(1,-1),axis=(0,1)) 
    plusiminusj = z - np.roll(z,(-1,1),axis=(0,1))

    helix = np.asarray([plusi,plusiminusj,minusj,minusi,minusiplusj,plusj])
    helicity = np.diff(np.sign(helix),axis=0,append=np.expand_dims(np.sign(helix[0]),axis=0))/2
    #helicity = np.diff(np.sign(helix),axis=0,append=np.expand_dims(np.sign(helix[0])-np.sign(helix[5]),axis=0))/2
    helicity = np.sum(np.abs(helicity),axis=0)
    print("how many zeros: ", np.sum((helix==0)[:]))

    if plot_crit:
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
        ax1.imshow(helicity==0)
        ax2.imshow(helicity==4)
        ax3.imshow(helicity==6)
        ax4.imshow(z)
        plt.show()
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
        ax4.scatter(np.where(helicity==0)[1],np.where(helicity==0)[0],c="red")
        ax4.imshow(z)
        ax3.imshow(z)
        ax3.scatter(np.where(helicity==4)[1],np.where(helicity==4)[0],c="red")
        plt.show()
    saddles = np.logical_or(helicity==4,helicity==6)

    extremamask, numextrema = label(helicity==0)
    saddlemask, numsaddle = label(saddles)
    saddlemask[saddlemask!=0] += numextrema

    saddles_ravel = np.ravel_multi_index(np.where(saddles),z.shape)
    saddles_with_depth = zip(saddles_ravel,z[np.where(saddles)])
    saddles_in_order = sorted(saddles_with_depth,key=lambda x: x[1])

    allcritmask = extremamask+saddlemask
    G = nx.Graph()

    for i in range(numextrema+numsaddle):
        G.add_node(i)

    region_map = np.full_like(z,np.nan)
    #fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax = plt.figure().add_subplot(projection='3d')
    for current_saddle in tqdm(saddles_in_order):
        st =[[0,1,1], [1,0,1], [1,1,0]]
        if current_saddle[1]>-3000:
            break
        under_saddle,reg = periodiclabel(z<current_saddle[1],periodic0=True,periodic1=True,st=st)
        include_saddle,reg = periodiclabel(z<=current_saddle[1],periodic0=True,periodic1=True,st=st)

        saddlepoint_coord = np.unravel_index(current_saddle[0],z.shape)

        current_saddle_id = allcritmask[saddlepoint_coord]

        includemask = (include_saddle == include_saddle[saddlepoint_coord])
        under_saddle_regions = np.unique(under_saddle[includemask])


        for i in under_saddle_regions:
            if i !=0:
                for j in np.unique(allcritmask[under_saddle==i]):
                    if j!=0 and j>0 and j!=current_saddle_id:
                        G.add_edge(current_saddle_id,j)
                        #ax.plot(np.deg2rad((np.where(allcritmask==current_saddle_id)[1][0]*2,2*np.where(allcritmask==j)[1][0])),(np.mean(z[allcritmask==current_saddle_id]),np.mean(z[allcritmask==j])))
                        lons = np.asarray([np.where(allcritmask==current_saddle_id)[1][0],np.where(allcritmask==j)[1][0]])
                        lats = np.asarray([np.where(allcritmask==current_saddle_id)[0][0],np.where(allcritmask==j)[0][0]])
                        zs = (np.mean(z[allcritmask==current_saddle_id]),np.mean(z[allcritmask==j]))
                        xs = lats*np.cos(np.deg2rad(lons*2))
                        ys = lats*np.sin(np.deg2rad(lons*2))
                        ax.plot(xs,ys,zs)
                        allcritmask[allcritmask==j] = -j

        #nx.draw(G)
        #plt.show()
    plt.show()
    fig,(ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(under_saddle)
    ax2.imshow(include_saddle)
    plt.show()

    

srtm = xr.open_dataset("SRTM_coarse.nc")
srtm.z.values[srtm.z.values>=0] = 0
srtm = srtm.coarsen(lon=4).mean().coarsen(lat=4).mean()
lons,lats = np.meshgrid(srtm.lon,srtm.lat)

print(srtm.z.shape)

contour_tree(srtm.z.values)
#contour_tree(testcase(100,100))








# Sort them by depth

# Slice below and above them to assign each point and find merging/splitting events
        
