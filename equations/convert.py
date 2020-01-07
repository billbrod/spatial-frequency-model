"""use MathJax CLI to convert tex files into svg images

This simple script calls `tex2svg` (from the MathJax CLI) to convert all
tex files found in the same directory as this file (e.g.,
`spatial-frequency-model/equations/`) into separate svg images with the
same name.

Usage:

`python convert.py`

This script accepts no arguments.

"""
import subprocess
import os.path as op
from glob import glob

dir_path = op.dirname(op.realpath(__file__))
for tex_file in glob(f'{dir_path}/*.tex'):
    svg_file = tex_file.replace('tex', 'svg')
    ps = subprocess.Popen(["cat", f"{tex_file}"], stdout=subprocess.PIPE)
    with open(svg_file, 'w') as outfile:
        subprocess.run(["xargs", "-0", "-t", "-I", "%", "tex2svg", "%"],
                       stdin=ps.stdout, stdout=outfile)
