FROM jodogne/orthanc:latest

WORKDIR /root/

RUN wget https://orthanc.uclouvain.be/downloads/linux-standard-base/orthanc-dicomweb/1.16/libOrthancDicomWeb.so
RUN mv ./libOrthancDicomWeb.so /usr/local/share/orthanc/plugins/