# octopuslite-reader

```
Load multidimensional image stacks using lazy loading.

A simple class to load OctopusLite data from a directory. Caches data once
it is loaded to prevent excessive I/O to the data server. Can directly
address different channels using the `Channels` enumerator.

Usage
-----
>>> octopus =  DaskOctopusLiteLoader('/path/to/your/data/')
>>> gfp = octopus["GFP"]
>>> gfp_filenames = octopus.files("GFP")
```
