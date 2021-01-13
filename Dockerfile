FROM continuumio/miniconda3:4.9.2
ARG conda_env=dash

# need to get mathjax-node-cli
RUN apt -y update
RUN apt install -y nodejs npm
RUN npm install --global mathjax-node-cli

# move over this directory
RUN mkdir /src
COPY . /src/spatial-frequency-model

# install dash
RUN /bin/bash -c "pip install dash==1.18.1"

# install other packages
RUN /bin/bash -c "pip install /src/spatial-frequency-model/"

# re-generate svgs for app
RUN /bin/bash -c "python /src/spatial-frequency-model/images/equations/convert.py"

# run webapp
ENTRYPOINT /bin/bash -c "python /src/spatial-frequency-model/webapp/app.py"
