FROM continuumio/miniconda3:4.7.12
ARG conda_env=dash

# need to get mathjax-node-cli
RUN apt -y update
RUN apt install -y nodejs npm
RUN npm install --global mathjax-node-cli

# move over this directory
RUN mkdir /src
COPY . /src/spatial-frequency-model

# set up our conda environment
RUN conda env create -f /src/spatial-frequency-model/webapp_environment.yml
RUN echo "source activate $conda_env" > ~/.bashrc
ENV PATH /opt/conda/envs/$conda_env/bin:$PATH

# activate our conda environment
RUN /bin/bash -c "source activate $conda_env"
RUN /bin/bash -c "pip install /src/spatial-frequency-model/"

# re-generate svgs for app
RUN /bin/bash -c "python /src/spatial-frequency-model/equations/convert.py"

# run webapp
ENTRYPOINT /bin/bash -c "python /src/spatial-frequency-model/webapp/app.py"
