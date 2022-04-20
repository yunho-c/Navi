FROM sweetvishnya/centos6.9-build

# OG caffe instructions

# complains about libstdc ELF but doesn't seem critical
RUN yum -y install protobuf-devel leveldb-devel snappy-devel opencv-devel boost-devel hdf5-devel 

RUN yum -y install gflags-devel glog-devel lmdb-devel

# glog
# RUN wget https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/google-glog/glog-0.3.3.tar.gz && \
#     tar zxvf glog-0.3.3.tar.gz && \
#     cd glog-0.3.3 && \
#     ./configure && \
#     make && make install

# gflags
# RUN wget https://github.com/schuhschuh/gflags/archive/master.zip && \
#     unzip master.zip && \
#     cd gflags-master && \
#     mkdir build && cd build && \
#     export CXXFLAGS="-fPIC" && cmake .. && make VERBOSE=1 && \
#     make && make install

