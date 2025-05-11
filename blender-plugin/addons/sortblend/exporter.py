#    This file is a part of SORT(Simple Open Ray Tracing), an open-source cross
#    platform physically based renderer.
#
#    Copyright (c) 2011-2023 by Jiayin Cao - All rights reserved.
#
#    SORT is a free software written for educational purpose. Anyone can distribute
#    or modify it under the the terms of the GNU General Public License Version 3 as
#    published by the Free Software Foundation. However, there is NO warranty that
#    all components are functional in a perfect manner. Without even the implied
#    warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
#    General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along with
#    this program. If not, see <http://www.gnu.org/licenses/gpl-3.0.html>.

import bpy
import os
import mathutils
import platform
import tempfile
import struct
import numpy as np
from time import time
from math import degrees
from .log import log, logD
from .strid import SID
from .stream import stream

BLENDER_VERSION = f'{bpy.app.version[0]}.{bpy.app.version[1]}'

def depsgraph_objects(depsgraph: bpy.types.Depsgraph):
    """ Iterates evaluated objects in depsgraph with ITERATED_OBJECT_TYPES """
    ITERATED_OBJECT_TYPES = ('MESH', 'LIGHT')
    for obj in depsgraph.objects:
        if obj.type in ITERATED_OBJECT_TYPES:
            yield obj.evaluated_get(depsgraph)

# Get the list of material for the whole scene, this function will only list materials that are currently
# attached to an object in the scene. Non-used materials will not be needed to be exported to SORT.
def list_materials(depsgraph):
    exported_materials = []
    for ob in depsgraph_objects(depsgraph):
        if ob.type == 'MESH':
            for material in ob.data.materials[:]:
                # make sure it is a SORT material
                if material and material.sort_material:
                    # skip if the material is already exported
                    if exported_materials.count(material) != 0:
                        continue
                    exported_materials.append(material)
    return exported_materials

def get_sort_dir():
    preferences = bpy.context.preferences.addons['sortblend'].preferences
    return_path = preferences.install_path
    if platform.system() == 'Windows':
        return return_path
    
    return_path = os.path.expanduser(return_path)
    return return_path

def get_sort_bin_path():
    sort_bin_dir = get_sort_dir()
    if platform.system() == 'Darwin':   # for Mac OS
        sort_bin_path = sort_bin_dir + "sort_r"
    elif platform.system() == 'Windows':    # for Windows
        sort_bin_path = sort_bin_dir + "sort_r.exe"
    elif platform.system() == "Linux":
        sort_bin_path = sort_bin_dir + "sort_r"
    else:
        raise Exception("SORT is only supported on Windows, Ubuntu and Mac OS")
    return sort_bin_path

intermediate_dir = ''
def get_intermediate_dir(force_debug=False):
    global intermediate_dir
    return_path = intermediate_dir if force_debug is False else get_sort_dir()
    if platform.system() == 'Windows':
        return_path = return_path.replace('\\', '/')
        return return_path + '/'
    if return_path[-1] == '/':
        return return_path
    return return_path + '/'

# Coordinate transformation
# Basically, the coordinate system of Blender and SORT is very different.
# In Blender, the coordinate system is as below and this is a right handed system
#
#        Z
#        ^   Y
#        |  /
#        | /
#        |/
#        ---------> X
#
# In SORT, the coordinate system is as below and this is a left handed system
#
#
#        Y
#        ^   Z
#        |  /
#        | /
#        |/
#        ---------> X
#
# So we need to transform the coordinate system from Blender to SORT.
# The transformation is as below:
#   X -> X
#   Y -> Z
#   Z -> Y
#
# This is a rotation around X axis by -90 degrees.
# The matrix is:
#   [1,  0,  0]
#   [0,  0,  1]
#   [0, -1,  0]
#
# And we need to transform the normal vectors as well.
# The transformation for normal vectors is the same as the transformation for points.
# Because the normal vectors are transformed by the inverse transpose of the transformation matrix.
# And the inverse transpose of the rotation matrix is the same as the rotation matrix itself.

def MatrixSortToBlender():
    """Convert SORT coordinate system to Blender coordinate system"""
    return mathutils.Matrix([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

def MatrixBlenderToSort():
    """Convert Blender coordinate system to SORT coordinate system"""
    return mathutils.Matrix([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

def lookat_camera(camera):
    """Get the view matrix for a camera"""
    # Get the camera matrix
    cam_matrix = camera.matrix_world
    
    # Get the camera location
    cam_loc = cam_matrix.translation
    
    # Get the camera rotation
    cam_rot = cam_matrix.to_quaternion()
    
    # Create the view matrix
    view_matrix = cam_matrix.inverted()
    
    # Transform to SORT coordinate system
    view_matrix = MatrixBlenderToSort() @ view_matrix @ MatrixSortToBlender()
    
    return view_matrix

def export_blender(depsgraph, force_debug=False, is_preview=False):
    """Export the current scene to SORT format"""
    # Create the output directory
    intermediate_dir = create_path(depsgraph.scene, force_debug)
    
    # Create the output file
    with open(os.path.join(intermediate_dir, 'scene.sort'), 'w') as f:
        fs = stream_module.Stream(f)
        
        # Export the scene
    export_scene(depsgraph, is_preview, fs)

    return intermediate_dir

def create_path(scene, force_debug):
    """Create the output directory"""
    if force_debug:
        return get_sort_dir()
    
    # Create a temporary directory
    global intermediate_dir
    intermediate_dir = tempfile.mkdtemp()
    return intermediate_dir

def get_smoke_modifier(obj: bpy.types.Object):
    """Get the smoke modifier from an object"""
    for modifier in obj.modifiers:
        if modifier.type == 'SMOKE':
            return modifier
    return None

def export_scene(depsgraph, is_preview, fs):
    """Export the scene to SORT format"""
    # Helper function to convert a matrix to a tuple
    def matrix_to_tuple(matrix):
        return tuple(matrix[i][j] for i in range(4) for j in range(4))

    # Helper function to convert a vector to a tuple
    def vec3_to_tuple(vec):
        return (vec.x, vec.y, vec.z)

    # Export the scene
    scene = depsgraph.scene

    # Export the camera
    if scene.camera:
    camera = scene.camera
        view_matrix = lookat_camera(camera)
        fs.write('camera\n')
        fs.write('{\n')
        fs.write('    view_matrix %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n' % matrix_to_tuple(view_matrix))
        fs.write('    fov %f\n' % degrees(camera.data.angle))
        fs.write('}\n')
    
    # Export the lights
    for obj in depsgraph_objects(depsgraph):
        if obj.type == 'LIGHT':
            light = obj.data
            if light.type == 'POINT':
                fs.write('light\n')
                fs.write('{\n')
                fs.write('    type point\n')
                fs.write('    position %f %f %f\n' % vec3_to_tuple(obj.location))
                fs.write('    intensity %f %f %f\n' % vec3_to_tuple(light.color * light.energy))
                fs.write('}\n')
            elif light.type == 'SUN':
                fs.write('light\n')
                fs.write('{\n')
                fs.write('    type directional\n')
                fs.write('    direction %f %f %f\n' % vec3_to_tuple(obj.matrix_world.to_quaternion() @ mathutils.Vector((0.0, 0.0, -1.0))))
                fs.write('    intensity %f %f %f\n' % vec3_to_tuple(light.color * light.energy))
                fs.write('}\n')
            elif light.type == 'SPOT':
                fs.write('light\n')
                fs.write('{\n')
                fs.write('    type spot\n')
                fs.write('    position %f %f %f\n' % vec3_to_tuple(obj.location))
                fs.write('    direction %f %f %f\n' % vec3_to_tuple(obj.matrix_world.to_quaternion() @ mathutils.Vector((0.0, 0.0, -1.0))))
                fs.write('    intensity %f %f %f\n' % vec3_to_tuple(light.color * light.energy))
                fs.write('    angle %f\n' % degrees(light.spot_size))
                fs.write('    blend %f\n' % light.spot_blend)
                fs.write('}\n')
            elif light.type == 'AREA':
                fs.write('light\n')
                fs.write('{\n')
                fs.write('    type area\n')
                fs.write('    position %f %f %f\n' % vec3_to_tuple(obj.location))
                fs.write('    normal %f %f %f\n' % vec3_to_tuple(obj.matrix_world.to_quaternion() @ mathutils.Vector((0.0, 0.0, -1.0))))
                fs.write('    intensity %f %f %f\n' % vec3_to_tuple(light.color * light.energy))
                fs.write('    size %f %f\n' % (light.size, light.size_y))
                fs.write('}\n')
    
    # Export the meshes
    for obj in depsgraph_objects(depsgraph):
        if obj.type == 'MESH':
            mesh = obj.to_mesh()
            export_mesh(obj, mesh, fs)
            bpy.data.meshes.remove(mesh)
    
    # Export the materials
    export_materials(depsgraph, fs)
    
    # Export the global configuration
    export_global_config(scene, fs, get_sort_dir())
    
    # Export the smoke
    for obj in depsgraph_objects(depsgraph):
        if obj.type == 'MESH':
            smoke_modifier = get_smoke_modifier(obj)
            if smoke_modifier:
                export_smoke(obj, fs)
    
    # Export the hair
    for obj in depsgraph_objects(depsgraph):
        if obj.type == 'MESH':
            for ps in obj.particle_systems:
                if ps.settings.type == 'HAIR':
                    export_hair(ps, obj, scene, is_preview, fs)
    
    # Export the shader resources
    collect_shader_resources(depsgraph, scene, fs)

# avoid having space in material name
def name_compat(name):
    if name is None:
        return 'None'
    else:
        return name.replace(' ', '_')

# export glocal settings for the renderer
def export_global_config(scene, fs, sort_resource_path):
    # global renderer configuration
    xres = scene.render.resolution_x * scene.render.resolution_percentage / 100
    yres = scene.render.resolution_y * scene.render.resolution_percentage / 100

    sort_data = scene.sort_data

    integrator_type = sort_data.integrator_type_prop
    
    fs.serialize( 0 )
    fs.serialize( sort_resource_path )
    fs.serialize( int(sort_data.thread_num_prop) )
    fs.serialize( int(sort_data.sampler_count_prop) )
    fs.serialize( int(xres) )
    fs.serialize( int(yres) )
    fs.serialize( sort_data.clampping )

    fs.serialize( SID(integrator_type) )
    fs.serialize( int(sort_data.inte_max_recur_depth) )
    if integrator_type == "PathTracing":
        fs.serialize( int(sort_data.max_bssrdf_bounces) )
    if integrator_type == "AmbientOcclusion":
        fs.serialize( sort_data.ao_max_dist )
    if integrator_type == "BidirPathTracing" or integrator_type == "LightTracing":
        fs.serialize( bool(sort_data.bdpt_mis) )
    if integrator_type == "InstantRadiosity":
        fs.serialize( sort_data.ir_light_path_set_num )
        fs.serialize( sort_data.ir_light_path_num )
        fs.serialize( sort_data.ir_min_dist )

# export smoke information
def export_smoke(obj, fs):
    smoke_modifier = get_smoke_modifier(obj)
    if not smoke_modifier:
        fs.serialize( SID('no_volume') )
        return
    
    # making sure there is density data
    domain = smoke_modifier.domain_settings
    if len(domain.density_grid) == 0:
        fs.serialize( SID('no_volume') )
        return

    fs.serialize( SID('has_volume') )
    
    # dimension of the volume data
    x, y, z = domain.domain_resolution
    fs.serialize(x)
    fs.serialize(y)
    fs.serialize(z)

    # the color itself, don't export it for now
    # color_grid = np.fromiter(domain.color_grid, dtype=np.float32)
    # fs.serialize(color_grid)

    FLTFMT = struct.Struct('=f')

    # the density itself
    density_data = bytearray()
    density_grid = np.fromiter(domain.density_grid, dtype=np.float32)
    for density in density_grid:
        density_data += FLTFMT.pack(density)

    fs.serialize(density_data)

# export a mesh
def export_mesh(obj, mesh, fs):
    LENFMT = struct.Struct('=i')
    FLTFMT = struct.Struct('=f')
    VERTFMT = struct.Struct('=ffffffff')
    LINEFMT = struct.Struct('=iiffi')
    POINTFMT = struct.Struct('=fff')
    TRIFMT = struct.Struct('=iiii')

    materials = mesh.materials[:]
    material_names = [m.name if m else None for m in materials]

    vert_cnt = 0
    primitive_cnt = 0
    verts = mesh.vertices
    wo3_verts = bytearray()
    wo3_tris = bytearray()

    global matname_to_id

    # output the mesh information.
    mesh.calc_normals()
    mesh.calc_loop_triangles()

    has_uv = bool(mesh.uv_layers)
    uv_layer = None
    if has_uv:
        active_uv_layer = mesh.uv_layers.active
        if not active_uv_layer:
            has_uv = False
        else:
            uv_layer = active_uv_layer.data

    # Warning this function seems to cause quite some trouble on MacOS during the first renderer somehow.
    # And this problem only exists on MacOS not the other two OS.
    # Since there is not a low hanging fruit solution for now, it is disabled by default
    # generate tangent if there is UV, there seems to always be true in Blender 2.8, but not in 2.7x
    #if has_uv:
    #    mesh.calc_tangents( uvmap = uv_layer_name )

    vert_cnt = 0
    remapping = {}

    mesh_sid = SID( mesh.name )

    for poly in mesh.polygons:
        smooth = poly.use_smooth
        normal = poly.normal[:]

        oi = []
        for loop_index in range(poly.loop_start, poly.loop_start + poly.loop_total):
            # vertex index
            vid = mesh.loops[loop_index].vertex_index
            # vertex information
            vert = verts[vid]

            # uv coordinate
            uvcoord = uv_layer[loop_index].uv[:] if has_uv else ( 0.0 , 0.0 )

            # use smooth normal if necessary
            if smooth:
                normal = vert.normal[:]

            #tangent = mesh.loops[loop_index].tangent

            # an unique key to identify the vertex
            key = (vid, loop_index, smooth)

            # acquire the key if possible, otherwise pack one
            out_idx = remapping.get(key)
            if out_idx is None:
                out_idx = vert_cnt
                remapping[key] = out_idx
                wo3_verts += VERTFMT.pack(vert.co[0], vert.co[1], vert.co[2], normal[0], normal[1], normal[2], uvcoord[0], uvcoord[1])
                vert_cnt += 1
            oi.append(out_idx)

        matid = -1
        matname = name_compat(material_names[poly.material_index]) if len( material_names ) > 0 else None
        matid = matname_to_id[matname] if matname in matname_to_id else -1
        if len(oi) == 3:
            # triangle
            wo3_tris += TRIFMT.pack(oi[0], oi[1], oi[2], matid)
            primitive_cnt += 1
        elif len(oi) == 4:
            # quad
            wo3_tris += TRIFMT.pack(oi[0], oi[1], oi[2], matid)
            wo3_tris += TRIFMT.pack(oi[0], oi[2], oi[3], matid)
            primitive_cnt += 2
        else:
            # no other primitive supported in mesh
            # assert( False )
            log("Warning, there is unsupported geometry. The exported scene may be incomplete.")

    fs.serialize(SID('MeshVisual'))
    fs.serialize(bool(has_uv))
    fs.serialize(LENFMT.pack(vert_cnt))
    fs.serialize(wo3_verts)
    fs.serialize(LENFMT.pack(primitive_cnt))
    fs.serialize(wo3_tris)

    # export smoke data if needed, this is for volumetric rendering
    export_smoke(obj, fs)

    fs.serialize(SID('end of mesh'))

    return (vert_cnt, primitive_cnt)

# export hair information
def export_hair(ps, obj, scene, is_preview, fs):
    LENFMT = struct.Struct('=i')
    POINTFMT = struct.Struct('=fff')

    vert_cnt = 0
    hair_step = ps.settings.display_step if is_preview else ps.settings.render_step
    width_tip = ps.settings.sort_data.fur_tip
    width_bottom = ps.settings.sort_data.fur_bottom

    # extract the material of the hair
    mat_local_index = ps.settings.material
    mat_index = 0xffffffff

    if mat_local_index > 0 and mat_local_index <= len( obj.data.materials ):
        mat_name = name_compat(obj.data.materials[mat_local_index-1].name)
        mat_index = matname_to_id[mat_name] if mat_name in matname_to_id else 0xffffffff

    # for some unknown reason
    steps = 2 ** hair_step

    verts = bytearray()

    world2Local = obj.matrix_world.inverted()
    num_parents = len( ps.particles )
    num_children = len( ps.child_particles )

    hair_cnt = num_parents + num_children
    total_hair_segs = 0

    real_hair_cnt = 0
    for pindex in range(hair_cnt):
        hair = []
        for step in range(0, steps + 1):
            co = ps.co_hair(obj, particle_no = pindex, step = step)

            if co[0] == 0 and co[1] == 0 and co[2] == 0:
                continue

            co = world2Local @ co
            hair.append( co )
            vert_cnt += 1

        if len(hair) <= 1:
            continue

        real_hair_cnt += 1

        verts += LENFMT.pack( len(hair) - 1 )
        for h in hair :
            verts += POINTFMT.pack( h[0] , h[1] , h[2] )
        total_hair_segs += len(hair) - 1

    fs.serialize( SID('HairVisual') )
    fs.serialize( real_hair_cnt )
    fs.serialize( width_tip )
    fs.serialize( width_bottom )
    fs.serialize( mat_index )
    fs.serialize( verts )

    return (vert_cnt, total_hair_segs)

# find the output node, duplicated code, to be cleaned
def find_output_node(material):
    if material and material.sort_material:
        ntree = material.sort_material
        for node in ntree.nodes:
            if getattr(node, "bl_idname", None) == 'SORTNodeOutput':
                return node
    return None

# get the from node of this socket if there is one recursively
def get_from_socket(socket):
    if not socket.is_linked:
        return None

    # there should be exactly one link attached to an input socket, this is guaranteed by Blender
    assert( len( socket.links ) == 1 )
    other = socket.links[0].from_socket
    if other.node.bl_idname == 'NodeReroute':
        return get_from_socket(other.node.inputs[0])
    return other

# This function will iterate through all visited nodes in the scene and populate everything in a hash table
# Apart from collecting shaders, it will also collect all heavy data, like measured BRDF data, texture.
def collect_shader_resources(depsgraph, scene, fs):
    # don't output any osl_shaders if using default materials
    if scene.sort_data.allUseDefaultMaterial is True:
        fs.serialize( 0 )
        return None

    resources = []
    visited = set()
    for material in list_materials(depsgraph):
        # get output nodes
        output_node = find_output_node(material)
        if output_node is None:
            continue

        def collect_resources(mat_node, visited, resources):
            if mat_node is None:
                return
            
            # no need to iterate a node twice
            if mat_node in visited:
                return
            visited.add(mat_node)
            
            # iterate through all source shader nodes.
            for socket in mat_node.inputs:
                input_socket = get_from_socket(socket)  # this is a temporary solution
                if input_socket is None:
                    continue
                collect_resources(input_socket.node, visited, resources)
            
            # if it is a shader group node, recursively iterate its nodes
            if mat_node.isGroupNode():
                # sub tree for the group nodes
                sub_tree = mat_node.getGroupTree()

                # recursively parse the node first
                output_node = sub_tree.nodes.get("Group Outputs")

                # recursively collect more nodes
                collect_resources(output_node, visited, resources)
            else:
                # otherwise simply populate the resources in this node
                mat_node.populateResources(resources)

        collect_resources(output_node, visited, resources)

    fs.serialize( len( resources ) )
    for resource in resources:
        fs.serialize( resource[0] ) # type
        fs.serialize( resource[1] ) # external file name

matname_to_id = {}
def export_materials(depsgraph, fs):
    # if we are in no-material mode, just skip outputting all materials
    if depsgraph.scene.sort_data.allUseDefaultMaterial is True:
        fs.serialize( SID('End of Material') )
        return None

    # this is used to keep track of all visited nodes to avoid duplicated nodes exported.
    visited_shader_unit_types = set()

    # loop through all materials and output them if valid
    i = 0
    materials = list_materials(depsgraph)
    for material in materials:
        # indicating material exporting
        logD( 'Exporting material %s.' %(material.name) )

        # get output nodes
        output_node = find_output_node(material)
        if output_node is None:
            logD( 'Material %s doesn\'t have any output node, it is invalid and will be ignored.' %(material.name) )
            continue
        
        # update the material mapping
        compact_material_name = name_compat(material.name)
        matname_to_id[compact_material_name] = i
        i += 1

        # whether the material has transparent node
        has_transparent_node = False
        # whether there is sss in the material
        has_sss_node = False

        # basically, this is a topological sort to serialize all nodes.
        # each node type will get exported exactly once to avoid duplicated shader unit compliation.
        def collect_shader_unit(shader_node, visited_instance, visited_types, shader_node_connections, node_type_mapping, input_index = -1):
            # no need to process a node multiple times
            if shader_node in visited_node_instances:
                return

            # add the current node to visited cache to avoid it being visited again
            if shader_node.isMaterialOutputNode() is False:
                visited_node_instances.add(shader_node)

            # update transparent and sss flag
            if shader_node.isTransparentNode() is True:
                nonlocal has_transparent_node
                has_transparent_node = True
            if shader_node.isSSSNode() is True:
                nonlocal has_sss_node
                has_sss_node = True

            # this identifies the unique name of the shader
            current_shader_node_name = shader_node.getUniqueName()

            # output node is a bit special that it can be revisited
            if shader_node.isMaterialOutputNode():
                current_shader_node_name = current_shader_node_name + compact_material_name

            # the type of the node
            shader_node_type = shader_node.type_identifier()

            # mapping from node name to node type
            node_type_mapping[shader_node] = shader_node_type

            # grab all source shader nodes
            inputs = shader_node.inputs 
            if input_index >= 0:
                # out of index, simply return, this is because some old assets doesn't have the volume channel
                # a bit tolerance will allow me to still use the render with old assets
                if input_index >= len(shader_node.inputs):
                    return
                else:
                    inputs = [shader_node.inputs[input_index]]

            for socket in inputs:
                input_socket = get_from_socket( socket )  # this is a temporary solution
                if input_socket is None:
                    continue
                source_node = input_socket.node

                source_param = source_node.getShaderOutputParameterName(input_socket.name)
                target_param = shader_node.getShaderInputParameterName(socket.name)

                source_shader_node_name = source_node.getUniqueName()

                # add the shader unit connection
                shader_node_connections.append( ( source_shader_node_name , source_param , current_shader_node_name, target_param ) )

                # recursively collect shader unit
                collect_shader_unit(source_node, visited_node_instances, visited_types, shader_node_connections, node_type_mapping)

            # no need to serialize the same node multiple times
            if shader_node_type in visited_types:
                return
            visited_types.add(shader_node_type)

            # export the shader node
            if shader_node.isGroupNode():
                # shader group should have a new set of connections
                shader_group_connections = []
                # shader group should also has its own node mapping
                shader_group_node_mapping = {}
                # start from a new visited cache
                shader_group_node_visited = set()   
                
                # sub tree for the group nodes
                sub_tree = shader_node.getGroupTree()

                # recursively parse the node first
                output_node = sub_tree.nodes.get("Group Outputs")
                collect_shader_unit(output_node, shader_group_node_visited, visited_types, shader_group_connections, shader_group_node_mapping)

                # it is important to visit the input node even if it is not connected since this needs to be connected with exposed arguments.
                # lacking this node will result in tsl compilation error
                input_node = sub_tree.nodes.get("Group Inputs")
                collect_shader_unit(input_node, shader_group_node_visited, visited_types, shader_group_connections, shader_group_node_mapping)

                # start serialization
                fs.serialize(SID("ShaderGroupTemplate"))
                fs.serialize(shader_node_type)

                fs.serialize(len(shader_group_node_mapping))
                for shader_node, shader_type in shader_group_node_mapping.items():
                    fs.serialize(shader_node.getUniqueName())
                    fs.serialize(shader_type)
                    shader_node.serialize_prop(fs)
                fs.serialize(len(shader_group_connections))
                for connection in shader_group_connections:
                    fs.serialize( connection[0] )
                    fs.serialize( connection[1] )
                    fs.serialize( connection[2] )
                    fs.serialize( connection[3] )
                
                # indicate the exposed arguments
                output_node.serialize_exposed_args(fs)

                # if there is input node, exposed the inputs
                input_node = sub_tree.nodes.get("Group Inputs")
                if input_node is not None:
                    input_node.serialize_exposed_args(fs)
                else:
                    fs.serialize( "" )
            else:
                fs.serialize(SID('ShaderUnitTemplate'))
                fs.serialize(shader_node_type)
                fs.serialize(shader_node.generate_osl_source())
                shader_node.serialize_shader_resource(fs)

        # this is the shader node connections
        surface_shader_node_connections = []
        volume_shader_node_connections = []

        # this hash table keeps track of all visited shader node instance
        visited_node_instances = set()

        # node type mapping, this maps from node name to node type
        surface_shader_node_type = {}
        volume_shader_node_type = {}

        # iterate the material for surface shader
        collect_shader_unit(output_node, visited_node_instances, visited_shader_unit_types, surface_shader_node_connections, surface_shader_node_type, 0)
        # iterate the material for volume shader
        collect_shader_unit(output_node, visited_node_instances, visited_shader_unit_types, volume_shader_node_connections, volume_shader_node_type, 1)

        # serialize this material, it is a real material
        fs.serialize(SID('Material'))
        fs.serialize(compact_material_name)

        if len(surface_shader_node_type) > 1:
            fs.serialize(SID('Surface Shader'))
            fs.serialize(len(surface_shader_node_type))
            for shader_node, shader_type in surface_shader_node_type.items():
                fs.serialize(shader_node.getUniqueName())
                fs.serialize(shader_type)
                shader_node.serialize_prop(fs)
            fs.serialize(len(surface_shader_node_connections))
            for connection in surface_shader_node_connections:
                fs.serialize( connection[0] )
                fs.serialize( connection[1] )
                fs.serialize( connection[2] )
                fs.serialize( connection[3] )
        else:
            fs.serialize( SID('Invalid Surface Shader') )

        if len(volume_shader_node_type) > 1 :
            fs.serialize(SID('Volume Shader'))
            fs.serialize(len(volume_shader_node_type))
            for shader_node, shader_type in volume_shader_node_type.items():
                fs.serialize(shader_node.getUniqueName())
                fs.serialize(shader_type)
                shader_node.serialize_prop(fs)
            fs.serialize(len(volume_shader_node_connections))
            for connection in volume_shader_node_connections:
                fs.serialize( connection[0] )
                fs.serialize( connection[1] )
                fs.serialize( connection[2] )
                fs.serialize( connection[3] )
        else:
            fs.serialize( SID('Invalid Volume Shader') )

        # mark whether there is transparent support in the material, this is very important because it will affect performance eventually.
        fs.serialize( bool(has_transparent_node) )
        fs.serialize( bool(has_sss_node) )

        # volume step size and step count
        fs.serialize( material.sort_material.volume_step )
        fs.serialize( material.sort_material.volume_step_cnt )

    # indicate the end of material parsing
    fs.serialize(SID('End of Material'))