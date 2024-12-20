FROM nvcr.io/nvidia/pytorch:23.08-py3

RUN apt update && apt install -y htop mc
RUN apt install -y git
# YOLOX
#RUN cd /workspace/ && git clone https://github.com/Megvii-BaseDetection/YOLOX.git
RUN cd /workspace/ && git clone https://github.com/neduchal/YOLOX_experiment.git YOLOX
RUN cd /workspace/YOLOX &&  pip3 install -v -e .

# Torch2trt
RUN cd /workspace/ &&git clone https://github.com/NVIDIA-AI-IOT/torch2trt
RUN cd /workspace/torch2trt && python setup.py install

RUN cd /workspace/ && mkdir models
#RUN cd /workspace/models/ && wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
#RUN cd /workspace/models/ && wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth
#RUN cd /workspace/models/ && wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth
#RUN cd /workspace/models/ && wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth
#RUN cd /workspace/models/ && wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_darknet.pth

RUN pip uninstall -y opencv-python opencv
RUN pip install opencv-python==4.8.0.74 opencv-contrib-python==4.8.0.74

# Nezle pouzivat cvWaitKey (nejaka issue s opencv -- zatim nevyreseno).