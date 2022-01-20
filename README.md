# octopuslite-reader

```
Load multidimensional image stacks using lazy loading.

A simple class to load OctopusLite data from a directory. Caches data once
it is loaded to prevent excessive I/O to the data server. Can directly
address different channels using the `Channels` enumerator.

Usage
-----
>>> images =  DaskOctopusLiteLoader(path = '/path/to/your/data/',
                                    crop = (1200,1600),
                                    transforms = 'path/to/transform_array.npy',
                                    remove_background = True)
>>> gfp = images["GFP"]
>>> gfp_filenames = images.files("GFP")


Parameters
  ----------
  path : str
      The path to the dataset.
  crop : tuple, optional
      An optional tuple which can be used to perform a centred crop on the data.
  transforms : np.ndarray, optional
      Transforms to be applied to the image stack.
  remove_background : bool, optional
      Use a estimated polynomial surface to remove uneven illumination.
```
