FROM dolfinx/dolfinx:v0.7.3

WORKDIR /home/app        

USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}

RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository universe
RUN apt-get install -y psmisc texlive-full 

WORKDIR /home/app         
RUN git clone https://github.com/janoschpreuss/wave-uc-dg-repro.git
WORKDIR /home/app/wave-uc-dg-repro



