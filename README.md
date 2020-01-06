# spatial-frequency-model
Model from the spatial frequency preferences paper

# Docker container

The recommended way to run this app is using the provided [docker
image](https://hub.docker.com/r/billbrod/spatial-frequency-model). Download
and install docker, make sure the docker daemon is running (if you're
not sure, try running `sudo dockerd` in the terminal), then run:

```
sudo docker pull billbrod/spatial-frequency-model
```

to download it from dockerhub.

Then to run:

```
sudo docker run -p 8050:8050 billbrod/spatial-frequency-model
```

and open `http://localhost:8050/` in your browser

## Build Dockerfile

If you want to build the `Dockerfile` yourself (for testing local
changes in this repo), just run:

```
sudo docker build --tag spatial-frequency-model:latest ./
```

from this directory, then `docker run -p 8050:8050
spatial-frequency-model` and open `http://localhost:8050/` in your
browser.

## Closing the container

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

# Run locally

If you want to run locally, you can either [build the
`Dockerfile`](#build-dockerfile) or create the python environment
locally. To do that, make sure you have `conda` installed (if you
don't already have it installed, see
[here](https://docs.conda.io/en/latest/miniconda.html) for how to
install `miniconda`, which is a minimal install), then from this
directory run the following on your command line:

```
conda env create -f webapp_environment.yml
conda activate dash
pip install -e .
python webapp/app.py
```

And open `http://localhost:8050/` in your browser (if it doesn't do so
automatically)
