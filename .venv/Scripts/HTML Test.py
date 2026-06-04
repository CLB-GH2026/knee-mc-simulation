
import numpy as np
import trimesh
import time
import pmcx
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion
from pathlib import Path
import webbrowser
import os

# Write a minimal test HTML to verify Plotly isosurface works in browser
test_html = """<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div id="plot" style="width:800px; height:600px;"></div>
    <script>
        // Simple 3x3x3 sphere isosurface
        var x = [], y = [], z = [], v = [];
        for (var i = -1; i <= 1; i += 0.1) {
            for (var j = -1; j <= 1; j += 0.1) {
                for (var k = -1; k <= 1; k += 0.1) {
                    x.push(i); y.push(j); z.push(k);
                    v.push(i*i + j*j + k*k);
                }
            }
        }
        var data = [{
            type: 'isosurface',
            x: x, y: y, z: z, value: v,
            isomin: 0.3, isomax: 0.7,
            surface: {count: 1},
            colorscale: 'Hot',
            caps: {x: {show: false}, y: {show: false}, z: {show: false}}
        }];
        Plotly.newPlot('plot', data, {});
    </script>
</body>
</html>"""

with open("test_isosurface.html", "w") as f:
    f.write(test_html)

abs_test = os.path.abspath("test_isosurface.html")
print(f"  Opening test isosurface: {abs_test}")
webbrowser.open(f"file:///{abs_test}")