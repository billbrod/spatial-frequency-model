# spatial-frequency-model

Model from the spatial frequency preferences paper

# Webapp 

This webapp is live [on my
website](https://wfbroderick.com/spatial-frequency-model/), but it takes a long
time to load and so you may be better off following the instructions below to
get it running.

## Docker container

The recommended way to run this app is using the provided [docker
image](https://hub.docker.com/r/billbrod/spatial-frequency-model) and
`docker-compose`. Download and install
[docker](https://docs.docker.com/engine/install/) and
[docker-compose](https://docs.docker.com/compose/install/), make sure the docker
daemon is running (if you're not sure, try running `sudo dockerd` in the
terminal), then run:

```
sudo docker-compose up
```

to download the image from dockerhub and start it up. Open
`http://localhost:8050/spatial-frequency-model` in your browser to view the app.

### Build Dockerfile

If you want to build the `Dockerfile` yourself (for testing local changes in
this repo or because the dockerhub image has gone down), just run:

```
sudo docker build --tag spatial-frequency-model:latest ./
```

from this directory, then `docker run -p 8050:8050 spatial-frequency-model` and
open `http://localhost:8050/spatial-frequency-model` in your browser.

### Closing the container

The image will run until you kill it, and the only way I've found to
do that is to find the docker container ID and tell docker to kill it
(for some reason, pressing `ctrl+c` doesn't work). So in a new
terminal window, run:

```
sudo docker ps
```

which will list all running docker images. One of these should, under
`IMAGE`, say `billbrod/spatial-frequency-model` or
`spatial-frequency-model`. Copy the value under `CONTAINER ID` and
then run:

```
sudo docker kill [CONTAINER ID]
```

which should kill the other `docker run` process.

## Run locally

If you want to run locally, you can either [build the
`Dockerfile`](#build-dockerfile) or install the package locally. If you want to
do that, I recommend using `conda` or something similar to manage your python
environments (see [below](#using-conda-for-python-environments) for more
information if you don't know how to set up an environment).

Once you have set up and activated the environment you want to install the
package into, run the following:

```
pip install dash
pip install -e .
python webapp/app.py
```

And open `http://localhost:8050/spatial-frequency-model` in your browser (if it
doesn't do so automatically). The `dash` version probably doesn't matter for
anything we do, but the docker container uses `1.18.1` if you want to match it.

The app uses a variety of `.svg` images of equations (since MathJax
and other LaTeX renderers apparently don't work in Dash right
now). They can be found in the `equations` directory and are generated
using the MathJax command line interface from the `.tex` files found
there as well. A small python script, `equations/convert.py`, is used
to script this conversion. The Docker image will automatically
re-generate these every time it's built (in case the `.tex` files
change), but you can generate them manually:

1. Install `node.js` and `npm`. (On Ubuntu: `sudo apt install nodejs
   npm`)
2. Use `npm` to install the MathJax command line interface: `npm
   install --global mathjax-node-cli`
3. Run the included python script: from this directory, run `python
   equations/convert.py` (it takes no arguments; it will convert all
   `.tex` files found in the same directory).

### Using conda for python environments

To use conda to manage your python environments, make sure you have `conda`
installed (if you don't already have it installed, see
[here](https://docs.conda.io/en/latest/miniconda.html) for how to install
`miniconda`, which is a minimal install, make sure to pay attention to how to
set up your shell after installing it), then from this directory run the
following on your command line:

```
conda create -n spatial-frequency python==3.7
conda activate spatial-frequency
```

This will create a new python 3.7 environment called `spatial-frequency` which
will only contain the basics, and then "activate" that environment. Once an
environment has been activated, all python-related commands (e.g., `python`,
`pip`) will use *those* versions. This allows you to keep different versions and
different packages in different environments. After you've created and activated
this new environment, follow the steps above to install the webapp.

