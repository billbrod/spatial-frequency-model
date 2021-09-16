FROM continuumio/miniconda3:latest
ARG conda_env=dash

# need to get mathjax-node-cli
RUN apt -y update
# for which we need nodejs version 12 or greater; only up to 10 is included in
# the default repos, so we add this one
RUN apt -y install curl
RUN curl -sL https://deb.nodesource.com/setup_16.x | bash -
RUN apt install -y nodejs
RUN npm install --global mathjax-node-cli

# move over this directory
RUN mkdir /src
COPY . /src/spatial-frequency-model

# install dash
RUN /bin/bash -c "pip install dash==2.0"

# install other packages
RUN /bin/bash -c "pip install /src/spatial-frequency-model/"

# re-generate svgs for app
RUN /bin/bash -c "python /src/spatial-frequency-model/images/equations/convert.py"

# run webapp
ENTRYPOINT /bin/bash -c "python /src/spatial-frequency-model/webapp/app.py"
