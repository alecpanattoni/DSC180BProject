ARG BASE_CONTAINER=ucsdets/datahub-base-notebook:2022.3-stable

FROM $BASE_CONTAINER

LABEL maintainer="UC San Diego ITS/ETS <ets-consult@ucsd.edu"

USER root

RUN apt update

# install python

USER jovyan

# download necesaary packages

RUN pip install --no-cache-dir numpy pandas aif360 sklearn tensorflow matplotlib

# Override command to disable running jupyter notebook at launch
CMD ["/bin/bash"]
