FROM continuumio/miniconda3:4.11.0

ENV ENVIRONMENT=selectml

LABEL maintainer="darcy.ab.jones@gmail.com"
LABEL pipeline.name="${ENVIRONMENT}"

RUN apt-get update \
 && apt-get install -y procps libtinfo6 \
 && apt-get clean -y

ENV PATH="/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

COPY environment.yml /
RUN conda env create --force -f /environment.yml \
 && conda clean -a --yes \
 && sed -i '/conda activate base/d' ~/.bashrc

ENV CONDA_PREFIX="/opt/conda/envs/${ENVIRONMENT}"
ENV PATH="${CONDA_PREFIX}/bin:${PATH}"
ENV PYTHONPATH="${CONDA_PREFIX}/lib/python3.6/site-packages:${PYTHONPATH}"

ENV CPATH="${CPATH}:${CONDA_PREFIX}/include"
ENV LIBRARY_PATH="${LIBRARY_PATH}:${CONDA_PREFIX}/lib"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib"

CMD [ "/bin/bash" ]
