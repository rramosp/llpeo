import os
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from rlxutils import subplots
import tensorflow as tf
import shapely as sh
import rasterio as rio
from pyproj.crs import CRS
import pathlib
import geopandas as gpd
from progressbar import progressbar as pbar
import hashlib
from itertools import islice
import copy

def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while (batch := tuple(islice(it, n))):
        yield batch

def gethash(s: str):
    k = int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10**15
    k = str(hex(k))[2:].zfill(13)
    return k


class Chipset:
    
    def __init__(self, basedir, number_of_classes = None):
        self.basedir = basedir
        if not self.basedir.endswith("/data"):
            self.basedir += "/data"
        _, _, self.files = list(os.walk(self.basedir))[0]
        self.number_of_classes = number_of_classes

    def __len__(self):
        return len(self.files)
        
    def __iter__(self):
        for file in self.files:
            yield Chip(f"{self.basedir}/{file}", number_of_classes=self.number_of_classes)        
        
    def random(self):
        file = np.random.choice(self.files)
        return Chip(f"{self.basedir}/{file}", number_of_classes=self.number_of_classes)
    
    def get_chip(self, chip_id):
        file = f'{self.basedir}/{chip_id}.pkl'
        return Chip(file, number_of_classes=self.number_of_classes)

class Chip:
    def __init__(self, filename=None, number_of_classes=None):

        # if no filename create and empty instance
        if filename is None:
            return

        if not filename.endswith(".pkl"):
            filename = f"{filename}.pkl"

        self.filename = filename

        with open(filename, "rb") as f:
            self.data = pickle.load(f)        
            
        for k,v in self.data.items():
            exec(f"self.{k}=v")
        
        # older versions stored foreign proportions in different dictionary formats
        self.proportions_flattened = {k: v['proportions'] if isinstance(v, dict) and 'proportions' in v.keys() else v \
                                      for k,v in self.label_proportions.items()}

        # try to infer the number of classes from data in this chip
        if number_of_classes is None:
            # take the max number present in labels or proportions
            number_of_classes = np.max([np.max(np.unique(self.label))+1, 
                                        np.max([int(k)+1 for v in self.proportions_flattened.values() if isinstance(v, dict) for k in v.keys() ])])

        self.number_of_classes = number_of_classes

    def clone(self):
        r = self.__class__()
        r.data = copy.deepcopy(self.data)
        r.number_of_classes = self.number_of_classes
        r.filename = self.filename
        r.proportions_flattened = copy.deepcopy(self.proportions_flattened)
        for k,v in r.data.items():
            exec(f"r.{k}=v")
        return r

    def get_partition_ids(self):
        return list(self.label_proportions.keys())

    def group_classes(self, class_groups):
        """
        creates a new chip with the same contents but with classes grouped by modifying
        the label and the class proportions

        class_groups: list of tuples of classids. Must contain all classes. For instance:
                      [ 0, (2,3), (5,), 4 ] will map 
                          - class 0 to class 1
                          - classes 2 and 3 to class 1
                          - class 5 to class 2
                          - class 4 to class 3

                      tuples will assigned the new classids in the order specified.
                      groups with a single item can be specified with a tuple of len 1
                      or just the class number.

                      input class 0 must be in the first class group
        returns: a new chip with class grouped as specified
        """

        for g in class_groups:
            if not isinstance(g, int) and not isinstance(g, tuple):
                    raise ValueError("groups must be integer (single classes) or tuples")

        label = self.data['label']

        class_groups = [i if isinstance(i, tuple) else (i,) for i in class_groups]

        if not(class_groups[0] == 0 or (isinstance(class_groups[0], tuple) and 0 in class_groups[0])):
            raise ValueError("class 0 must be in the first group")
        

        selected_classids = [i for j in class_groups for i in j]
        number_of_output_classes = len(class_groups)
        
        if sorted(selected_classids)!=list(range(self.number_of_classes)):
            raise ValueError(f"incorrect mapping for {self.number_of_classes} classes. maybe repeated or missing classes")

        class_mapping = {k2:i for i,k1 in enumerate(class_groups) for k2 in k1}

        mapped_label = np.zeros_like(label)
        for original_class, mapped_class in class_mapping.items():
            mapped_label[label==original_class] = mapped_class
            
        r = self.clone()
        r.label = mapped_label
        r.data['label'] = mapped_label
        r.number_of_classes = number_of_output_classes 

        r.proportions_flattened = {k1: {str(k2):0.0 for k2 in range(number_of_output_classes)} for k1 in self.proportions_flattened.keys()}

        # add proportions for classes explicitly specified in class_groups
        for original_class, mapped_class in class_mapping.items():
            original_class = str(original_class)
            mapped_class = str(mapped_class)
            for partition_id in self.proportions_flattened.keys():
                p = self.proportions_flattened[partition_id]
                if not isinstance(p, dict):
                    continue
                if original_class in p.keys():
                    r.proportions_flattened[partition_id][str(mapped_class)] += p[original_class]

        # aggregate proportions for the rest of the classes to class 0
        for original_class in range(self.number_of_classes):
            if not original_class in selected_classids:
                original_class = str(original_class)
                for partition_id in self.proportions_flattened.keys():
                    p = self.proportions_flattened[partition_id]
                    if not isinstance(p, dict):
                        continue
                    if original_class in p.keys(): 
                        r.proportions_flattened[partition_id][str(0)] += p[original_class]
            
        # insert proportions into 'data' dict
        lp = r.data['label_proportions']
        for k,v in r.proportions_flattened.items():
            if k in lp.keys():
                if not isinstance(lp[k], dict):
                    continue

                if 'proportions' in lp[k].keys():
                    lp[k]['proportions'] = v
                else:
                    lp[k] = v 
                    
        # check all proportions add up to 1
        for k,v in lp.items():
            if not isinstance(v, dict):
                continue

            if 'proportions' in v.keys():
                v = v['proportions']
            if np.allclose(sum(v.values()),0, atol=1e-4):
                raise ValueError(f"internal error: mapped proportions do not add up to 1 in {self.filename}, they add up to {sum(v.values())}. check maybe the class_groups")


        return r        

    def remove(self):
        """
        physically remove the file of this chip
        """
        os.remove(self.filename)

    def compute_label_proportions_on_chip_label(self):
        l = pd.Series(self.label.flatten()).value_counts() / 100**2
        return {i: (l[i] if i in l.index else 0) for i in range(self.number_of_classes)}

    def get_polygon(self):
        """
        returns a shapely polygon in degrees as lon/lat (epsg 4326)
        """
        g = self.data['metadata']['corners']
        nw_lat, nw_lon = g['nw']
        se_lat, se_lon = g['se']
        p = sh.geometry.Polygon([[nw_lon, nw_lat], [se_lon, nw_lat], [se_lon, se_lat], [nw_lon, se_lat], [nw_lon, nw_lat]])
        return p

    def to_geotiff(self, filename=None):
        if filename is not None:
            geotiff_filename = filename
        else:
            fpath = pathlib.Path(self.filename)
            geotiff_filename  = self.filename[:-len(fpath.suffix)] + '.tif'
            
        if 'chipmean' in self.data.keys():
            pixels = self.data['chipmean']
        else:
            pixels = self.data['chip']

        maxy, minx = self.metadata['corners']['nw']
        miny, maxx = self.metadata['corners']['se']        

        transform = rio.transform.from_origin(minx, maxy, (maxx-minx)/pixels.shape[1], (maxy-miny)/pixels.shape[0])

        new_dataset = rio.open(geotiff_filename, 'w', driver='GTiff',
                                    height = pixels.shape[0], width = pixels.shape[1],
                                    count=3, dtype=str(pixels.dtype),
                                    crs=CRS.from_epsg(4326),
                                    transform=transform)

        for i in range(3):
            new_dataset.write(pixels[:,:,i], i+1)
        new_dataset.close()

    def plot(self):

        for ax,i in subplots(3, usizex=5, usizey=3.5):
            if i==0: 
                plt.imshow(self.chip) ## XX
                plt.title("/".join(self.filename.split("/")[-1:]))
            if i==1:
                cmap=matplotlib.colors.ListedColormap([plt.cm.gist_ncar(i/self.number_of_classes) \
                                                       for i in range(self.number_of_classes)])
                plt.imshow(self.label, vmin=0, vmax=self.number_of_classes, cmap=cmap, interpolation='none')
                plt.title("label")
                cbar = plt.colorbar(ax=ax, ticks=range(self.number_of_classes))
                cbar.ax.set_yticklabels([f"{i}" for i in range(self.number_of_classes)])  # vertically oriented colorbar
            if i==2:
                n = len(self.proportions_flattened)
                for i, (k,v) in enumerate(self.proportions_flattened.items()):
                    p = [v[str(j)] if str(j) in v.keys() else 0 for j in range(self.number_of_classes)]
                    plt.bar(np.arange(self.number_of_classes)+(i-n/2.5)*.15, p, 0.15, label=k)
                plt.xticks(range(self.number_of_classes), range(self.number_of_classes));
                plt.title("label proportions at\ndifferent partition sizes")
                plt.ylim(0,1); plt.grid();
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        return self
    
    
    