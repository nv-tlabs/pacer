# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

import open3d as o3d
import open3d.visualization.rendering as rendering
import imageio
from tqdm import tqdm
# from momentum.mmt_isaac_wrapper import Momentum_Holder
import joblib
import numpy as np


def main():
    render = rendering.OffscreenRenderer(640, 480)

    yellow = rendering.MaterialRecord()
    yellow.base_color = [1.0, 0.75, 0.0, 1.0]
    yellow.shader = "defaultLit"

    color = rendering.MaterialRecord()
    color.base_color = [0.0, 0.5, 0.5, 1.0]
    color.shader = "defaultLit"

    # cyl.compute_vertex_normals()
    # cyl.translate([-2, 0, 1.5])

    # render.scene.add_geometry("cyl", mesh, color)
    render.setup_camera(60.0, [0, 0, -1], [0, 0, 0], [0, 1, 0])

    render.scene.scene.set_sun_light([0.707, 0.0, -.707], [1.0, 1.0, 1.0], 75000)
    render.scene.scene.enable_sun_light(True)
    render.scene.show_axes(True)

    writer = imageio.get_writer("/hdd/zen/dev/nv/crossroad/output/renderings/test_orig.mp4", fps=30, macro_block_size=None)

    for i in tqdm(range(500)):
        # render.scene.remove_geometry('cyl')
        # render.scene.add_geometry("cyl", mesh, color)
        img = render.render_to_image()
        writer.append_data(np.asarray(img))
    writer.close()


if __name__ == "__main__":
    main()