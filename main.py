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

def imshow3d(ax, array, value_direction='z', pos=0, norm=None, cmap=None):
    """
    Display a 2D array as a  color-coded 2D image embedded in 3d.

    The image will be in a plane perpendicular to the coordinate axis *value_direction*.

    Parameters
    ----------
    ax : Axes3D
        The 3D Axes to plot into.
    array : 2D numpy array
        The image values.
    value_direction : {'x', 'y', 'z'}
        The axis normal to the image plane.
    pos : float
        The numeric value on the *value_direction* axis at which the image plane is
        located.
    norm : `~matplotlib.colors.Normalize`, default: Normalize
        The normalization method used to scale scalar data. See `imshow()`.
    cmap : str or `~matplotlib.colors.Colormap`, default: :rc:`image.cmap`
        The Colormap instance or registered colormap name used to map scalar data
        to colors.
    """
    if norm is None:
        norm = Normalize()
    colors = plt.get_cmap(cmap)(norm(array))

    if value_direction == 'x':
        nz, ny = array.shape
        zi, yi = np.mgrid[0:nz + 1, 0:ny + 1]
        xi = np.full_like(yi, pos)
    elif value_direction == 'y':
        nx, nz = array.shape
        xi, zi = np.mgrid[0:nx + 1, 0:nz + 1]
        yi = np.full_like(zi, pos)
    elif value_direction == 'z':
        ny, nx = array.shape
        yi, xi = np.mgrid[0:ny + 1, 0:nx + 1]
        zi = np.full_like(xi, pos)
    else:
        raise ValueError(f"Invalid value_direction: {value_direction!r}")
    ax.plot_surface(xi, yi, zi, rstride=1, cstride=1, facecolors=colors, shade=False)



def get_descendants(region_d,region_p,previous_slice,mask):
    descendants = list(np.unique(previous_slice[mask]))
    for eyed in np.unique(previous_slice[mask]):
        if eyed != 0:
            descendants += region_d[eyed]
    return list(np.unique(descendants))

def build_contour_tree(bedvalues,step=20,start=-2000,stop=0,full=False):
    ## previous slice
    previous_slice = np.full_like(bedvalues,0)
    unique_id = 0
    region_points = {}
    region_depths = {}
    region_descendents = {}
    region_parents = {}
    region_size = {}
    region_map = np.full_like(bedvalues,np.nan)
    for depth in tqdm(range(start,0,step)):
        next_slice =  np.full_like(bedvalues,0)
        labels, c = periodiclabel(bedvalues<depth,False,True)
        for label_number in tqdm(range(1,c+1)):
            label_mask = np.asarray(labels==label_number)
            if len(np.unique(previous_slice[label_mask]))>2 or np.nanmin(np.unique(previous_slice[label_mask]))==0 :
                ## this means that we are going to merge to regions
                new_region_id = unique_id-1
                unique_id -=1
                if full:
                    coords = np.where(label_mask)
                else:
                    coords = np.where(np.logical_and(label_mask,previous_slice==0))
                region_points[new_region_id] = list(np.ravel_multi_index(coords,bedvalues.shape))
                region_map[coords]=new_region_id
                region_depths[new_region_id] = [depth]
                region_descendents[new_region_id] = get_descendants(region_descendents,region_parents,previous_slice,label_mask)
                
                for eyed in np.unique(previous_slice[label_mask]):
                    region_parents[eyed] = new_region_id
                next_slice[label_mask] = new_region_id

            else:
                ## This regions just growing in volume or staying the same
                ## the id will be the previous slices id at that location that isn't 0,
                ## our ids are negative so we can grab that with the minimum
                region_id = np.min(previous_slice[label_mask])
                coords = np.where(np.logical_and(label_mask,previous_slice==0))
                region_map[coords]=region_id
                region_points[new_region_id] += list(np.ravel_multi_index(coords,bedvalues.shape))
                #region_points[region_id].append(np.where(label_mask))
                next_slice[label_mask] = region_id

        previous_slice=next_slice
    return region_points, region_depths, region_descendents,region_parents,region_map


srtm = xr.open_dataset("SRTM_coarse.nc")
srtm.z.values[srtm.z.values>=0] = 0
srtm = srtm.coarsen(lon=4).mean().coarsen(lat=4).mean()
lons,lats = np.meshgrid(srtm.lon,srtm.lat)
#contour_out = build_contour_tree(srtm.z.values,step=500,start=-9000,stop=0)
#with open("contour_out.pickle","wb") as f:
    #pickle.dump(contour_out,f)
with open("contour_out.pickle","rb") as f:
    region_points, region_depths, region_descendents,region_parents,region_map = pickle.load(f)
plt.imshow(region_map)
plt.show()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set(xlabel="x", ylabel="y", zlabel="z")
cmap = matplotlib.colors.ListedColormap ( np.random.rand ( 256,3))
region_map[np.isnan(region_map)]=0
imshow3d(ax,region_map[::-1,:],pos=-8000)
bed = srtm.z.values
coordsx,coordsy = np.meshgrid(range(bed.shape[0]),range(bed.shape[1]))
flatcoordsx = coordsx.flatten()
flatcoordsy = coordsy.flatten()
flatbed = bed.flatten()

average_coords = {}
for i in np.sort(np.unique(region_map)):
    if i!=0:
        coords = np.asarray(region_points[i])
        average_coords[i] = (np.median(flatcoordsy[coords]),np.median(flatcoordsx[coords]),np.median(flatbed[coords]))
for i in np.sort(np.unique(region_map)):
    if i!=0:
        coord0 = average_coords[i]
        ax.scatter(coord0[0],coord0[1],coord0[2])
        if i in region_parents:
            coord1 = average_coords[region_parents[i]]
            ax.plot((coord0[0],coord1[0]),(coord0[1],coord1[1]),(coord0[2],coord1[2]))
plt.show()
#
#tree = Tree()
#tree.create_node('root', 'root')
#for i in np.sort(np.unique(region_map)):
    ##if i in region_parents:
        #tree.create_node(i, i, parent=region_parents[i])
    #else:
        #tree.create_node(i, i,parent='root')
#tree.to_graphviz()
#tree.show()

#
#
#plt.imshow(np.roll(region_map[::-1,:],100,axis=1),cmap=cmap)
#plt.show()
#
#for i in np.sort(np.unique(region_map))[::-1]:
    #if ~np.isnan(i) and np.sum(region_map == i)<2000 and i in region_parents.keys():
        #region_map[region_map==i] = region_parents[i]
#plt.imshow(region_map[::-1,:],cmap=cmap)
#plt.show()
