# Internals

The internal functions and types of the package, outside the documented API.
They are listed for those reading its source; they are not part of the API and
may change in any release. A package extending TrackingLoops — a host's noise
source through [`despread_noise!`](@ref TrackingLoops.despread_noise!), a custom
correlator or estimator — needs none of them.

```@autodocs
Modules = [TrackingLoops]
Filter = !listed
```
