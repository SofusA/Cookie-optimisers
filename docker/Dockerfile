FROM pytorch/pytorch

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# System packages
RUN apt-get update && apt-get install -y \
  wget

# Conda packages
RUN conda install -y numpy jupyter matplotlib scikit-learn nltk bokeh scikit-image cython pandas seaborn

RUN conda install -y -c conda-forge regex jupyterlab pyhamcrest spacy 

RUN pip install torchtext gym
RUN pip install git+https://github.com/jakevdp/JSAnimation.git

WORKDIR /work
CMD ["bash"]
