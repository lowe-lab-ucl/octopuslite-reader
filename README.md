# octopuslite-reader

```
A simple class to load OctopusLite data from a directory.
Caches data once it is loaded to prevent excessive io to the data server.

  Can directly address fluorescence channels using the
  `Channels` enumerator:

      Channels.BRIGHTFIELD
      Channels.GFP
      Channels.RFP
      Channels.IRFP

  Usage:
      octopus = SimpleOctopusLiteLoader('/path/to/your/data')
      gfp = octopus[Channels.GFP]
```
