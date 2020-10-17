# McFarthest Points (2020-10-16):

## Background
This folder contains two notebooks (`McFarPoints.ipynb` and
`McFarPoints_nonsphere.ipynb`) for computing the location that is
farther away from a given restaurant chain than any other point in the
continental US or within a given state. The reference list of
restaraunt chain locations is sourced from [Open Street
Map](https://www.openstreetmap.org/) data (sourced via
[Overpass](https://wiki.openstreetmap.org/wiki/Overpass_API)).

The idea behind these Notebook came from the notion of the "McFarthest
Spot," described in [this
post](http://www.datapointed.net/2009/09/distance-to-nearest-mcdonalds/)
by Stephen Von Worley. As noted by [Atlas
Obscura](https://www.atlasobscura.com/places/mcfarthest-spot-skb), the
McFarthest Spot has moved a bit since it was first computed, and
currently sits in Nevada. We won't be able to recover this location
because Open Street Maps seems to have an incomplete collection of
McDonald's locations.

I also computed McFarthest points within each of the Lower 48 states,
and for Burger King and Wendy's too.

The computation method utilizes a [Voronoi
diagram](https://en.wikipedia.org/wiki/Voronoi_diagram) to obtain
regions of points nearest to each fast food location. The McFarthest
point is extracted from the vertices of this diagram. In the notebook,
`McFarPoints.ipynb`, I constructed a Voronoi diagram on the surface of
a spherical Earth model to compute nearest regions. This method should
produce reasonable results. In the other notebook,
`McFarPoints_nonsphere.ipynb`, I used a simple planar Voronoi diagram
in an equidistant projected coordinate system to estimate the
McFarthest point. This method does not compute "nearest" neighbors in
the true distance units, so the McFarthest point found in this method
is not quite correct.

## Details
### Dependencies
To run the notebook, you will need to install a few dependencies,
 which are summarized in `mcfar.yml`. If you use
 [Anaconda](https://www.anaconda.com/), you can install everything to
 an environment named `mcfar` by running:

```
conda env create -f mcfar.yml
```

To exactly reproduce my environment, use `mcfar_complete.yml` instead.

### Overpass URL
I currently have an Overpass server on my local machine, so the
Notebook points to this local server. If you don't have Overpass
installed locally, you can override this by changing the
`overpass_url` (an alternate is provided in the code).

### Output
If you don't want to bother with running the notebook and just want
the Fast Food location / McFarthest point outputs, then you're in
luck! Those are saved in the outputs folder.
