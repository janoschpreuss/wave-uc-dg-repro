FROM dolfinx/dolfinx:v0.7.3

WORKDIR /home/app        

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository universe

RUN apt-get install -y vim emacs
RUN pip3 install --no-cache-dir numpy scipy matplotlib sympy pypardiso meshio gmsh 
RUN apt-get install -yq psmisc texlive-full 


WORKDIR /home/app         
RUN git clone https://github.com/janoschpreuss/wave-uc-dg-repro.git
WORKDIR /home/app/wave-uc-dg-repro



