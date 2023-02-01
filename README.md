# octopuslite-reader

```
Load multidimensional image stacks using lazy loading.

A simple class to load OctopusLite data from a directory. Caches data once
it is loaded to prevent excessive I/O to the data server. Can directly
address different channels using the `Channels` enumerator.

Usage
-----
>>> from octopuslite import DaskOctopus, MetadataParser
>>> images =  DaskOctopusLite(
    path = '/path/to/your/data/',
    crop = (1200,1600),
    transforms = 'path/to/transform_array.npy',
    remove_background = True,
    parser = MetadataParser.OCTOPUS,
)
>>> gfp = images["GFP"]
```
