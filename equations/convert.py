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
