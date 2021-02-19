from miniplant.scene_creator import _create_scene_common, create_direct_scene, create_diffuse_scene
from miniplant.utils import MyLight
from pvtrace import Scene, Light, Box, Luminophore
from anytree import LevelOrderIter, Node


def get_light_node() -> Node:
    return Node(
        name="Test Light",
        light=Light(),
        parent=None,
    )


def scene_is_valid(scene: Scene):
    """ Basic check on scene validity """
    # I expect a scene
    assert isinstance(scene, Scene)
    # With exactly one light source
    assert len(scene.light_nodes) == 1
    light = scene.light_nodes[0]
    # light is Light
    assert isinstance(light.light, Light)


def scene_has_dye(scene: Scene):
    """ Check if the scene has a box object with a Luminophore """
    # Dye is included
    for node in LevelOrderIter(scene.root):
        # Get LSC-PM
        if hasattr(node, "geometry") and isinstance(node.geometry, Box):
            # Check that dye is included as default
            return any(isinstance(x, Luminophore) for x in node.geometry.material.components)


def test__create_scene_common():
    scene = _create_scene_common(tilt_angle=0, light_source=get_light_node())

    scene_is_valid(scene)
    assert scene_has_dye(scene)


def test__create_scene_common_w_dye():
    scene = _create_scene_common(tilt_angle=0, light_source=get_light_node(), include_dye=True)

    assert scene_has_dye(scene)


def test__create_scene_common_wo_dye():
    scene = _create_scene_common(tilt_angle=0, light_source=get_light_node(), include_dye=False)

    assert not scene_has_dye(scene)


def test_create_diffuse_scene():
    scene = create_direct_scene()

    scene_is_valid(scene)
    assert scene_has_dye(scene)  # Default is dye included


def test_create_diffuse_scene_wo_dye():
    scene = create_direct_scene(include_dye=False)

    scene_is_valid(scene)
    assert not scene_has_dye(scene)


def test_create_direct_scene():
    scene = create_direct_scene()

    scene_is_valid(scene)
    assert scene_has_dye(scene)  # Default is dye included


def test_create_direct_scene():
    scene = create_diffuse_scene()
    # I expect a scene
    assert isinstance(scene, Scene)
    # With exactly one light source
    assert len(scene.light_nodes) == 1
    light = scene.light_nodes.pop()
    # light is MyLight
    assert isinstance(light.light, MyLight)
