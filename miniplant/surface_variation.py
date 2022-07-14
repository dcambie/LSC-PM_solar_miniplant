import numpy as np
import random

from pvtrace import (
    Node,
    Box,
    Sphere,
    Material,
    Luminophore,
    Absorber,
    Cylinder,
    Reactor,
    Light,
    Scene,
    Mesh,
    Distribution,
    Surface,
    SurfaceDelegate,
    FresnelSurfaceDelegate,
    isotropic,
)

from pvtrace.geometry.utils import flip, angle_between
from pvtrace.material.utils import (
    cone,
    lambertian,
    fresnel_reflectivity,
    specular_reflection,
    fresnel_refraction,
)


class BoxNormals(object):
    left = (-1, 0,  0)
    right = (1,  0,  0)
    near = (0, -1,  0)
    far = (0,  1,  0)
    bottom = (0,  0, -1)
    top = (0,  0,  1)


def diffuse_reflection(direction, normal):
    """ The diffuse reflection sends a scattering lambertian upward.
    When other directions are desired, it needs to be altered here.
    """
    reflected_direction = lambertian()
    return reflected_direction


class SurfaceMirror(FresnelSurfaceDelegate):
    """ The top surface is covered with a perfect mirror.
    """

    def reflectivity(self, surface, ray, geometry, container, adjacent):
        """ Return the reflectivity of the part of the surface hit by the ray.

            Parameters
            ----------
            surface: Surface
                The surface object belonging to the material.
            ray: Ray
                The ray hitting the surface in the local coordinate system of the `geometry` object.
            geometry: Geometry
                The object being hit (e.g. Sphere, Box, Cylinder, Mesh etc.)
            container: Node
                The node containing the ray.
            adjacent: Node
                The node that will contain the ray if the ray is transmitted.
        """
        # Get the surface normal to determine which surface has been hit.
        normal = geometry.normal(ray.position)

        # Normal are outward facing
        normals = BoxNormals()

        # If a ray hits the top surface set the reflectivity to 1.
        if np.allclose(normal, normals.top):
            return 1.0
        else:
            return super(SurfaceMirror, self).reflectivity(surface, ray, geometry, container, adjacent)

    def reflected_direction(self, surface, ray, geometry, container, adjacent):
        """ Returns the reflected direction vector (ix, iy, iz).

            Parameters
            ----------
            surface: Surface
                The surface object owned by the material.
            ray: Ray
                The incident ray.
            geometry: Geometry
                The geometry being hit.
            container: Node
                The node containing the incident ray.
            adjacent: Node
                The node that would contain the ray if transmitted.
        """
        from pvtrace import Mesh
        if isinstance(geometry, Box):
            normal = geometry.normal(ray.position)
        elif isinstance(geometry, Mesh):
            normal = geometry.normal_from_intersection(ray)
        else:
            normal = geometry.normal(ray.position)

        direction = ray.direction
        reflected_direction = specular_reflection(direction, normal)
        return tuple(reflected_direction.tolist())

    def transmitted_direction(self, surface, ray, geometry, container, adjacent):
        """ Returns the transmitted direction vector (ix, iy, iz).

            Parameters
            ----------
            surface: Surface
                The surface object owned by the material.
            ray: Ray
                The incident ray.
            geometry: Geometry
                The geometry being hit.
            container: Node
                The node containing the incident ray.
            adjacent: Node
                The node that would contain the ray if transmitted.
        """
        n1 = container.geometry.material.refractive_index
        n2 = adjacent.geometry.material.refractive_index
        # Be tolerance with definition of surface normal
        from pvtrace import Mesh
        if isinstance(geometry, Box):
            normal = geometry.normal(ray.position)
        elif isinstance(geometry, Mesh):
            normal = geometry.normal_from_intersection(ray)
        else:
            normal = geometry.normal(ray.position)
        if np.dot(normal, ray.direction) < 0.0:
            normal = flip(normal)
        refracted_direction = fresnel_refraction(ray.direction, normal, n1, n2)
        return tuple(refracted_direction.tolist())


class SurfaceScatterer(FresnelSurfaceDelegate):
    """ The bottom surface is covered with a perfect scatterer.
    For a scatterer after an infinitely small air-gap, transmission direction can be used (otherwise specular).
    """

    def reflectivity(self, surface, ray, geometry, container, adjacent):
        """ Return the reflectivity of the part of the surface hit by the ray.

            Parameters
            ----------
            surface: Surface
                The surface object belonging to the material.
            ray: Ray
                The ray hitting the surface in the local coordinate system of the `geometry` object.
            geometry: Geometry
                The object being hit (e.g. Sphere, Box, Cylinder, Mesh etc.)
            container: Node
                The node containing the ray.
            adjacent: Node
                The node that will contain the ray if the ray is transmitted.
        """
        # Get the surface normal to determine which surface has been hit.
        normal = geometry.normal(ray.position)

        # Normal are outward facing
        normals = BoxNormals()

        # If a ray hits the top surface where x > 0 and y > 0 reflection
        # set the reflectivity to 1.
        if np.allclose(normal, normals.bottom):
            return 1.0
        else:
            return super(SurfaceScatterer, self).reflectivity(surface, ray, geometry, container, adjacent)

    def reflected_direction(self, surface, ray, geometry, container, adjacent):
        """ Returns the reflected direction vector (ix, iy, iz).

            Parameters
            ----------
            surface: Surface
                The surface object owned by the material.
            ray: Ray
                The incident ray.
            geometry: Geometry
                The geometry being hit.
            container: Node
                The node containing the incident ray.
            adjacent: Node
                The node that would contain the ray if transmitted.
        """
        from pvtrace import Mesh
        if isinstance(geometry, Box):
            normal = geometry.normal(ray.position)
        elif isinstance(geometry, Mesh):
            normal = geometry.normal_from_intersection(ray)
        else:
            normal = geometry.normal(ray.position)

        normals = BoxNormals()

        if np.allclose(normal, normals.bottom):
            direction = ray.direction
            reflected_direction = diffuse_reflection(direction, normal)
            return tuple(reflected_direction.tolist())
        else:
            return super(SurfaceScatterer, self).reflected_direction(surface, ray, geometry, container, adjacent)