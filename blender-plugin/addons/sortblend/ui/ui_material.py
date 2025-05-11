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
from .. import material
from .. import base

class SORTMaterialPanel:
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "material"
    COMPAT_ENGINES = {'SORT'}

    @classmethod
    def poll(cls, context):
        return context.scene.render.engine in cls.COMPAT_ENGINES

class SORT_new_material_base(bpy.types.Operator):
    bl_label = "New"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        # currently picked object
        obj = bpy.context.object

        # add the new material
        mat = bpy.data.materials.new('Material')

        # initialize default sort shader nodes
        mat.sort_material = bpy.data.node_groups.new(
            f'SORT_({mat.name})',
            type=material.SORTShaderNodeTree.bl_idname
        )

        output = mat.sort_material.nodes.new('SORTNodeOutput')
        default = mat.sort_material.nodes.new('SORTNode_Material_Diffuse')
        output.location[0] += 200
        output.location[1] += 200
        default.location[1] += 200
        mat.sort_material.links.new(default.outputs[0], output.inputs[0])

        # add a new material slot or assign the newly added material in the picked empty slot
        materials = obj.data.materials
        cur_mat_id = obj.active_material_index
        if cur_mat_id >= 0 and cur_mat_id < len(materials) and materials[cur_mat_id] is None:
            materials[cur_mat_id] = mat
        else:
            materials.append(mat)

        return {'FINISHED'}

@base.register_class
class SORT_new_material(SORT_new_material_base):
    """Add a new material"""
    bl_idname = "sort_material.new"

@base.register_class
class SORT_new_material_menu(SORT_new_material_base):
    """Add a new material"""
    bl_idname = "sort.material_new_menu"

@base.register_class
class MATERIAL_PT_MaterialSlotPanel(SORTMaterialPanel, bpy.types.Panel):
    bl_label = 'Material Slot'
    bl_order = 0

    def draw(self, context):
        layout = self.layout
        ob = context.object

        if ob:
            row = layout.row()
            row.template_list(
                "MATERIAL_UL_matslots",
                "",
                ob,
                "material_slots",
                ob,
                "active_material_index",
                rows=4
            )
            col = row.column(align=True)
            col.operator("object.material_slot_add", icon='ADD', text="")
            col.operator("object.material_slot_remove", icon='REMOVE', text="")
            if ob.mode == 'EDIT':
                row = layout.row(align=True)
                row.operator("object.material_slot_assign", text="Assign")
                row.operator("object.material_slot_select", text="Select")
                row.operator("object.material_slot_deselect", text="Deselect")
        split = layout.split(factor=0.75)
        if ob:
            split.template_ID(ob, "active_material", new="sort_material.new")
            row = split.row()
            if context.material_slot:
                row.prop(context.material_slot, "link", text="")
            else:
                row.label()
        elif context.material:
            split.template_ID(context.space_data, "pin_id")
            split.separator()

@base.register_class
class SORT_OT_use_sort_node(bpy.types.Operator):
    """Use SORT Shader Node"""
    bl_idname = "sort.use_sort_node"
    bl_label = "Use SORT Shader Node"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        mat = context.material
        mat.sort_material = bpy.data.node_groups.new(
            f'SORT_({mat.name})',
            type=material.SORTShaderNodeTree.bl_idname
        )

        output = mat.sort_material.nodes.new('SORTNodeOutput')
        default = mat.sort_material.nodes.new('SORTNode_Material_Diffuse')
        output.location[0] += 200
        output.location[1] += 200
        default.location[1] += 200
        mat.sort_material.links.new(default.outputs[0], output.inputs[0])

        return {"FINISHED"}

@base.register_class
class SORT_OT_node_socket_restore_shader_group_input(bpy.types.Operator):
    """Move socket"""
    bl_idname = "sort.node_socket_restore_shader_group_input"
    bl_label = "Restore Shader Group Input"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        # get current edited tree
        tree = context.material.sort_material

        # get property location for placing the input node
        loc, _ = material.get_io_node_locations(tree.nodes)

        # create an input node and place it on the left of all nodes
        node_type = 'sort_shader_node_group_input' if material.is_sort_node_group(tree) else 'SORTNodeExposedInputs'
        node_input = tree.nodes.new(node_type)
        node_input.location = loc
        node_input.selected = False
        node_input.tree = tree

        return {"FINISHED"}

@base.register_class
class SORT_OT_node_socket_base(bpy.types.Operator):
    """Move socket"""
    bl_idname = "sort.node_socket_base"
    bl_label = "Move Socket"
    bl_options = {'REGISTER', 'UNDO'}

    type: bpy.props.EnumProperty(
        items=(
            ('up', 'Up', 'Move socket up'),
            ('down', 'Down', 'Move socket down'),
            ('remove', 'Remove', 'Remove socket'),
        ),
        name='Operation',
        description='Operation to perform on the socket'
    )
    pos: bpy.props.IntProperty(
        name='Position',
        description='Socket position in the list'
    )
    node_name: bpy.props.StringProperty(
        name='Node Name',
        description='Name of the node containing the socket'
    )

    def execute(self, context):
        node = context.space_data.edit_tree.nodes[self.node_name]
        tree = node.id_data
        kind = node.node_kind
        io = getattr(node, kind)
        socket = io[self.pos]

        if self.type == 'remove':
            io.remove(socket)
            if material.is_sort_node_group(tree):
                # update instances
                for instance in material.instances(tree):
                    sockets = getattr(instance, material.map_lookup[kind])
                    sockets.remove(sockets[self.pos])
            else:
                # update root shader inputs
                shader_input_node = tree.nodes.get('Shader Inputs')
                sockets = getattr(shader_input_node, material.map_lookup[kind])
                sockets.remove(sockets[self.pos])
        else:
            step = -1 if self.type == 'up' else 1
            count = len(io) - 1

            def calc_new_position(pos, step, count):
                return max(0, min(pos + step, count - 1))

            new_pos = calc_new_position(self.pos, step, count)
            io.move(self.pos, new_pos)

            if material.is_sort_node_group(tree):
                # update instances
                for instance in material.instances(tree):
                    sockets = getattr(instance, material.map_lookup[kind])
                    new_pos = calc_new_position(self.pos, step, len(sockets))
                    sockets.move(self.pos, new_pos)
            else:
                shader_input_node = tree.nodes.get('Shader Inputs')
                sockets = getattr(shader_input_node, material.map_lookup[kind])
                new_pos = calc_new_position(self.pos, step, len(sockets))
                sockets.move(self.pos, new_pos)

        material.update_cls(tree)
        return {"FINISHED"}

@base.register_class
class SORT_OT_node_socket_move(SORT_OT_node_socket_base):
    """Move socket"""
    bl_idname = "sort.node_socket_move"
    bl_label = "Move Socket"

@base.register_class
class SORT_OT_node_socket_remove(SORT_OT_node_socket_base):
    """Remove socket"""
    bl_idname = "sort.node_socket_remove"
    bl_label = "Remove Socket"

@base.register_class
class SORT_OT_node_socket_restore_input_node(bpy.types.Operator):
    """Move socket"""
    bl_idname = "sort.node_socket_restore_group_input"
    bl_label = "Restore Group Input"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        # get current edited tree
        tree = context.space_data.edit_tree

        # get property location for placing the input node
        loc, _ = material.get_io_node_locations(tree.nodes)

        # create an input node and place it on the left of all nodes
        node_type = 'sort_shader_node_group_input' if material.is_sort_node_group(tree) else 'SORTNodeExposedInputs'
        node_input = tree.nodes.new(node_type)
        node_input.location = loc
        node_input.selected = False
        node_input.tree = tree

        return {"FINISHED"}

@base.register_class
class SORT_OT_node_socket_restore_output_node(bpy.types.Operator):
    """Move socket"""
    bl_idname = "sort.node_socket_restore_group_output"
    bl_label = "Restore Group Output"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        # get current edited tree
        tree = context.space_data.edit_tree

        # get property location for placing the output node
        _, loc = material.get_io_node_locations(tree.nodes)

        # create an output node and place it on the right of all nodes
        node_type = 'sort_shader_node_group_output' if material.is_sort_node_group(tree) else 'SORTNodeExposedOutputs'
        node_output = tree.nodes.new(node_type)
        node_output.location = loc
        node_output.selected = False
        node_output.tree = tree

        return {"FINISHED"}

@base.register_class
class MATERIAL_PT_MaterialParameterPanel(SORTMaterialPanel, bpy.types.Panel):
    bl_label = 'Material Parameters'
    bl_order = 1

    @classmethod
    def poll(cls, context):
        return context.material and context.material.sort_material

    def draw(self, context):
        layout = self.layout
        mat = context.material
        if mat and mat.sort_material:
            layout.operator("sort.use_sort_node", text="Use SORT Shader Node")

@base.register_class
class MATERIAL_PT_MaterialVolumePanel(SORTMaterialPanel, bpy.types.Panel):
    bl_label = 'Volume'
    bl_order = 2

    @classmethod
    def poll(cls, context):
        return context.material and context.material.sort_material

    def draw(self, context):
        layout = self.layout
        mat = context.material
        if mat and mat.sort_material:
            layout.prop(mat.sort_material, "volume_step")
            layout.prop(mat.sort_material, "volume_step_cnt")

@base.register_class
class MATERIAL_PT_SORTInOutGroupEditor(SORTMaterialPanel, bpy.types.Panel):
    bl_label = "SORT In/Out Group Editor"
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_order = 3

    @classmethod
    def poll(cls, context):
        return (context.space_data.type == 'NODE_EDITOR' and
                context.space_data.tree_type == material.SORTShaderNodeTree.bl_idname)

    def draw(self, context):
        def set_attrs(cls, **kwargs):
            for key, value in kwargs.items():
                setattr(cls, key, value)

        def draw_socket(col, socket, index):
            row = col.row(align=True)
            row.prop(socket, "name", text="")
            row.operator("sort.node_socket_move", text="", icon='TRIA_UP').type = 'up'
            row.operator("sort.node_socket_move", text="", icon='TRIA_DOWN').type = 'down'
            row.operator("sort.node_socket_remove", text="", icon='X').type = 'remove'
            set_attrs(row.operator("sort.node_socket_move", text="", icon='TRIA_UP'),
                     type='up', pos=index, node_name=socket.node.name)
            set_attrs(row.operator("sort.node_socket_move", text="", icon='TRIA_DOWN'),
                     type='down', pos=index, node_name=socket.node.name)
            set_attrs(row.operator("sort.node_socket_remove", text="", icon='X'),
                     type='remove', pos=index, node_name=socket.node.name)

        layout = self.layout
        tree = context.space_data.edit_tree
        if tree:
            layout.operator("sort.node_socket_restore_group_input", text="Add Input")
            layout.operator("sort.node_socket_restore_group_output", text="Add Output")
            layout.separator()
            layout.label(text="Inputs:")
            for socket in tree.inputs:
                draw_socket(layout, socket, socket.index)
            layout.separator()
            layout.label(text="Outputs:")
            for socket in tree.outputs:
                draw_socket(layout, socket, socket.index)
