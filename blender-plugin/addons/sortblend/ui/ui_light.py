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
import bl_ui
from .. import base

class SORTLightPanel(bl_ui.properties_data_light.DataButtonsPanel):
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "data"
    COMPAT_ENGINES = {'SORT'}

    @classmethod
    def poll(cls, context):
        return super().poll(context) and context.scene.render.engine in cls.COMPAT_ENGINES

@base.register_class
class LIGHT_PT_SORTPanel(SORTLightPanel, bpy.types.Panel):
    bl_label = 'Light Property'
    bl_order = 0

    def draw(self, context):
        layout = self.layout
        light = context.light
        
        layout.prop(light, 'type', expand=True)
        layout.prop(light, 'color')
        layout.prop(light, 'energy')

        light_type = light.type
        if light_type == 'SPOT':
            layout.prop(light, "spot_size", text="Spot Light Range")
            layout.prop(light, "spot_blend", text="Spot Light Blend")
        elif light_type == 'AREA':
            layout.prop(light, 'shape')
            shape = light.shape
            if shape == 'SQUARE':
                layout.prop(light, "size")
            elif shape == 'RECTANGLE':
                layout.prop(light, "size")
                layout.prop(light, "size_y")
            elif shape == 'DISK':
                layout.prop(light, "size")
