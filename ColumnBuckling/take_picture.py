import sys
# # Set main
# edit_geometry_properties_object.py:150 out_data = {
# 'main' : AltGeometry(self, 'main', color=(1.0, 1.0, 1.0), line_width=8, opacity=1.0, point_size=1, bar_scale=0.0, representation='main', is_visible=True),
# }
# self.on_update_geometry_properties(out_data)
# # Set single-single point constraints visibility
# out_data = {
# 'SPC=3' : AltGeometry(self, 'SPC=3', color=(1.0, 0.0, 1.0), line_width=5, opacity=1.0, point_size=20, bar_scale=0.0, representation='point', is_visible=True),
# }
# self.on_update_geometry_properties(out_data)
# Show buckling mode
self.cycle_results(case=12)
# Take screenshot
self.on_take_screenshot('column_sol105.png', magnify=5)
# Exit GUI
sys.exit()
