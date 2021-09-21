# spatial-frequency-model

Model from the spatial frequency preferences paper. This repo has two
components: 
- [model code](#model-code): if you want to play around the model without the
  full [repo for the
  paper](https://github.com/billbrod/spatial-frequency-preferences).
- [webapp](#webapp): for exploring how the model's predictions respond to
  changes in parameter values.

If you use this model in an academic publication, please cite the paper **LINK**
and the Zenodo doi for the Github with code associated with the paper **LINK**

# Model code

To use the model code present here, you need to first set up the python
environment. The following will walk you through it:

- Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) on your
  system for python 3.7.
- Navigate to this directory and run the following:

``` sh
conda create -n sfm python==3.7
conda activate sfm
pip install -e .
```

Now, when you open python, you can access the model by importing this library
and making use of it:

``` python
import sfm
model = sfm.model.LogGaussianDonut()
```

See the example notebook in the paper repo for how to use the model:
[Binder](https://mybinder.org/v2/gh/billbrod/spatial-frequency-preferences/HEAD?filepath=notebooks),
[github repo](https://github.com/billbrod/spatial-frequency-preferences). Also
see [the
readme](https://github.com/billbrod/spatial-frequency-preferences#model-parameters)
for how to use the parameter values presented in the paper.

# Webapp 

This webapp is live [on my
website](https://wfbroderick.com/spatial-frequency-model/), but you can also do
the following to run it locally.

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

from this directory, then `sudo docker-compose up` and open
`http://localhost:8050/spatial-frequency-model` in your browser.

### Closing the container

If you opened this image with `docker-compose`, as recommended, you should be
able to press `ctrl+c` once to kill it gracefully. If that doesn't work (or you
started it in background mode), you may have to find the docker container ID and
tell docker to kill. In a new terminal window, run:

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

which should kill the open process.

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

