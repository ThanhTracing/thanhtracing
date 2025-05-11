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
import platform
import subprocess
from .. import exporter
from .. import base

# attach customized properties in particles
@base.register_class
class SORTRenderData(bpy.types.PropertyGroup):
    #------------------------------------------------------------------------------------#
    #                                 Renderer Settings                                  #
    #------------------------------------------------------------------------------------#
    # Integrator type
    integrator_types = [ ("PathTracing", "Path Tracing", "", 1),
                         ("BidirPathTracing", "Bidirectional Path Tracing", "", 2),
                         ("LightTracing", "Light Tracing", "", 3),
                         ("InstantRadiosity", "Instant Radiosity", "", 4),
                         ("AmbientOcclusion", "Ambient Occlusion", "", 5),
                         ("DirectLight", "Direct Lighting", "", 6),
                         ("WhittedRT", "Whitted", "", 7) ]
    integrator_type_prop: bpy.props.EnumProperty(
        items=integrator_types,
        name='Integrator',
        description='Select the integrator type'
    )

    # general integrator parameters
    inte_max_recur_depth: bpy.props.IntProperty(
        name='Maximum Recursive Depth',
        default=16,
        min=1,
        description='Maximum recursion depth for ray tracing'
    )

    # maxmum bounces supported in BSSRDF, exceeding the threshold will result in replacing BSSRDF with Lambert
    max_bssrdf_bounces: bpy.props.IntProperty(
        name='Maximum Bounces in SSS path',
        default=4,
        min=1,
        description='Maximum number of bounces in subsurface scattering paths'
    )

    # ao integrator parameters
    ao_max_dist: bpy.props.FloatProperty(
        name='Maximum Distance',
        default=3.0,
        min=0.01,
        description='Maximum distance for ambient occlusion'
    )

    # instant radiosity parameters
    ir_light_path_set_num: bpy.props.IntProperty(
        name='Light Path Set Num',
        default=1,
        min=1,
        description='Number of light path sets for instant radiosity'
    )
    ir_light_path_num: bpy.props.IntProperty(
        name='Light Path Num',
        default=64,
        min=1,
        description='Number of light paths per set for instant radiosity'
    )
    ir_min_dist: bpy.props.FloatProperty(
        name='Minimum Distance',
        default=1.0,
        min=0.0,
        description='Minimum distance between virtual point lights'
    )

    # bidirectional path tracing parameters
    bdpt_mis: bpy.props.BoolProperty(
        name='Multiple Importance Sampling',
        default=True,
        description='Enable multiple importance sampling for bidirectional path tracing'
    )

    #------------------------------------------------------------------------------------#
    #                              Spatial Accelerator Settings                          #
    #------------------------------------------------------------------------------------#
    # Accelerator type
    accelerator_types = [ ("Qbvh", "QBVH", "SIMD(SSE) Optimized BVH" , 0),
                          ("Obvh", "OBVH", "SIMD(AVX) Optimized BVH" , 1 ),
                          ("bvh", "BVH", "Binary Bounding Volume Hierarchy", 2),
                          ("KDTree", "SAH KDTree", "K-dimentional Tree", 3),
                          ("UniGrid", "Uniform Grid", "This is not quite practical in all cases.", 4),
                          ("OcTree" , "OcTree" , "This is not quite practical in all cases." , 5),
                          ("Embree", "Embree", "This is Intel Embree (Experimental)", 6)]
    accelerator_type_prop: bpy.props.EnumProperty(
        items=accelerator_types,
        name='Accelerator',
        description='Select the spatial accelerator type'
    )

    # bvh properties
    bvh_max_node_depth: bpy.props.IntProperty(
        name='Maximum Recursive Depth',
        default=28,
        min=8,
        description='Maximum recursion depth for BVH construction'
    )
    bvh_max_pri_in_leaf: bpy.props.IntProperty(
        name='Maximum Primitives in Leaf Node',
        default=8,
        min=8,
        max=64,
        description='Maximum number of primitives in a BVH leaf node'
    )

    # qbvh properties
    qbvh_max_node_depth: bpy.props.IntProperty(
        name='Maximum Recursive Depth',
        default=28,
        min=8,
        description='Maximum recursion depth for QBVH construction'
    )
    qbvh_max_pri_in_leaf: bpy.props.IntProperty(
        name='Maximum Primitives in Leaf Node',
        default=16,
        min=4,
        max=64,
        description='Maximum number of primitives in a QBVH leaf node'
    )

    # obvh properties
    obvh_max_node_depth: bpy.props.IntProperty(
        name='Maximum Recursive Depth',
        default=28,
        min=8,
        description='Maximum recursion depth for OBVH construction'
    )
    obvh_max_pri_in_leaf: bpy.props.IntProperty(
        name='Maximum Primitives in Leaf Node',
        default=16,
        min=8,
        max=64,
        description='Maximum number of primitives in an OBVH leaf node'
    )

    # kdtree properties
    kdtree_max_node_depth: bpy.props.IntProperty(
        name='Maximum Recursive Depth',
        default=28,
        min=8,
        description='Maximum recursion depth for KDTree construction'
    )
    kdtree_max_pri_in_leaf: bpy.props.IntProperty(
        name='Maximum Primitives in Leaf Node',
        default=8,
        min=8,
        max=64,
        description='Maximum number of primitives in a KDTree leaf node'
    )

    # octree properties
    octree_max_node_depth: bpy.props.IntProperty(
        name='Maximum Recursive Depth',
        default=16,
        min=8,
        description='Maximum recursion depth for Octree construction'
    )
    octree_max_pri_in_leaf: bpy.props.IntProperty(
        name='Maximum Primitives in Leaf Node',
        default=16,
        min=8,
        max=64,
        description='Maximum number of primitives in an Octree leaf node'
    )

    #------------------------------------------------------------------------------------#
    #                                 Clampping Settings                                 #
    #------------------------------------------------------------------------------------#
    clampping: bpy.props.FloatProperty(
        name='Clampping',
        default=0,
        min=0,
        description='Clamp values to prevent fireflies'
    )

    #------------------------------------------------------------------------------------#
    #                                 Sampling Settings                                  #
    #------------------------------------------------------------------------------------#
    sampler_count_prop: bpy.props.IntProperty(
        name='Count',
        default=1,
        min=1,
        description='Number of samples per pixel'
    )

    #------------------------------------------------------------------------------------#
    #                                 Debugging Settings                                 #
    #------------------------------------------------------------------------------------#
    detailedLog: bpy.props.BoolProperty(
        name='Output Detailed Output',
        default=False,
        description='Whether outputing detail log information in blender plugin'
    )
    profilingEnabled: bpy.props.BoolProperty(
        name='Enable Profiling',
        default=False,
        description='Enabling profiling will have a big impact on performance, only use it for simple scene'
    )
    allUseDefaultMaterial: bpy.props.BoolProperty(
        name='No Material',
        default=False,
        description='Disable all materials in SORT, use the default one'
    )
    thread_num_prop: bpy.props.IntProperty(
        name='Thread Num',
        default=0,
        min=0,
        max=128,
        description='Force specific number of threads, 0 means the number of threads will match number of physical cores'
    )

    @classmethod
    def register(cls):
        bpy.types.Scene.sort_data = bpy.props.PointerProperty(
            name="SORT Data",
            type=cls,
            description="SORT render settings"
        )

    @classmethod
    def unregister(cls):
        del bpy.types.Scene.sort_data

def OpenFile(filename):
    if platform.system() == 'Darwin':     # for Mac OS
        os.system(f"open \"{filename}\"")
    elif platform.system() == 'Windows':  # for Windows
        os.system(f"\"{filename}\"")
    elif platform.system() == "Linux":    # for linux
        os.system(f"xdg-open \"{filename}\"")

def OpenFolder(path):
    if platform.system() == 'Darwin':     # for Mac OS
        subprocess.call(["open", "-R", path])
    elif platform.system() == 'Windows':  # for Windows
        subprocess.call(f"explorer \"{path}\"")
    elif platform.system() == "Linux":    # for linux
        os.system(f"xdg-open \"{path}\"")

class SORTRenderPanel:
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"
    COMPAT_ENGINES = {'SORT'}

    @classmethod
    def poll(cls, context):
        return context.scene.render.engine in cls.COMPAT_ENGINES

@base.register_class
class RENDER_PT_IntegratorPanel(SORTRenderPanel, bpy.types.Panel):
    bl_label = 'Renderer'
    bl_order = 0

    def draw(self, context):
        layout = self.layout
        data = context.scene.sort_data
        
        layout.prop(data, "integrator_type_prop")
        integrator_type = data.integrator_type_prop
        
        if integrator_type not in {"WhittedRT", "DirectLight", "AmbientOcclusion"}:
            layout.prop(data, "inte_max_recur_depth")
            
        if integrator_type == "PathTracing":
            layout.prop(data, "max_bssrdf_bounces")
        elif integrator_type == "AmbientOcclusion":
            layout.prop(data, "ao_max_dist")
        elif integrator_type == "BidirPathTracing":
            layout.prop(data, "bdpt_mis")
        elif integrator_type == "InstantRadiosity":
            layout.prop(data, "ir_light_path_set_num")
            layout.prop(data, "ir_light_path_num")
            layout.prop(data, "ir_min_dist")

@base.register_class
class RENDER_PT_AcceleratorPanel(SORTRenderPanel, bpy.types.Panel):
    bl_label = 'Accelerator'
    bl_order = 1

    def draw(self, context):
        layout = self.layout
        data = context.scene.sort_data
        
        layout.prop(data, "accelerator_type_prop")
        accelerator_type = data.accelerator_type_prop
        
        if accelerator_type == "bvh":
            layout.prop(data, "bvh_max_node_depth")
            layout.prop(data, "bvh_max_pri_in_leaf")
        elif accelerator_type == "Qbvh":
            layout.prop(data, "qbvh_max_node_depth")
            layout.prop(data, "qbvh_max_pri_in_leaf")
        elif accelerator_type == "Obvh":
            layout.prop(data, "obvh_max_node_depth")
            layout.prop(data, "obvh_max_pri_in_leaf")
        elif accelerator_type == "KDTree":
            layout.prop(data, "kdtree_max_node_depth")
            layout.prop(data, "kdtree_max_pri_in_leaf")
        elif accelerator_type == "OcTree":
            layout.prop(data, "octree_max_node_depth")
            layout.prop(data, "octree_max_pri_in_leaf")

@base.register_class
class RENDER_PT_ClamppingPanel(SORTRenderPanel, bpy.types.Panel):
    bl_label = 'Clampping'
    bl_order = 2

    def draw(self, context):
        layout = self.layout
        data = context.scene.sort_data
        layout.prop(data, "clampping")

@base.register_class
class RENDER_PT_SamplerPanel(SORTRenderPanel, bpy.types.Panel):
    bl_label = 'Sample'
    bl_order = 3

    def draw(self, context):
        layout = self.layout
        layout.prop(context.scene.sort_data, "sampler_count_prop")

@base.register_class
class SORT_export_debug_scene(bpy.types.Operator):
    bl_idname = "sort.export_debug_scene"
    bl_label = "Export SORT Scene"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        depsgraph = context.evaluated_depsgraph_get()
        exporter.export_blender(depsgraph, True, True)
        return {'FINISHED'}

@base.register_class
class SORT_open_log(bpy.types.Operator):
    bl_idname = "sort.open_log"
    bl_label = "Open Log"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        logfile = exporter.get_sort_dir() + "log.txt"
        OpenFile(logfile)
        return {'FINISHED'}

@base.register_class
class SORT_openfolder(bpy.types.Operator):
    bl_idname = "sort.openfolder_sort"
    bl_label = "Open SORT folder"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        OpenFolder(exporter.get_sort_dir())
        return {'FINISHED'}

@base.register_class
class RENDER_PT_DebugPanel(SORTRenderPanel, bpy.types.Panel):
    bl_label = 'Debug'
    bl_order = 4

    def draw(self, context):
        layout = self.layout
        data = context.scene.sort_data
        
        layout.operator("sort.export_debug_scene")
        
        split = layout.split()
        left = split.column(align=True)
        left.operator("sort.open_log")
        right = split.column(align=True)
        right.operator("sort.openfolder_sort")

        layout.prop(data, "detailedLog")
        layout.prop(data, "profilingEnabled")
        layout.prop(data, "allUseDefaultMaterial")
        layout.prop(data, "thread_num_prop")
