FROM diyer22/tiny_cv2:4.5.5-py38-ubuntu20.04
ENV LC_ALL=C.UTF-8 LANG=C.UTF-8 DEBIAN_FRONTEND=noninteractive FORCE_CUDA="1"

RUN pip3 install --no-cache-dir -U pip wheel setuptools 
RUN pip3 install --no-cache-dir MegEngine numpy Pillow

RUN mkdir -p /ws
WORKDIR /ws/
# Download the pretrained MegEngine model from github
RUN wget -q --no-check-certificate https://github.com/yl-data/yl-data.github.io/files/8461965/crestereo_eth3d.mge-v1.zip && unzip -qq crestereo_eth3d.mge-v1.zip && rm crestereo_eth3d.mge-v1.zip

# Install debug tools
RUN pip3 install --no-cache-dir ipython && apt install -y vim
# && sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.2/zsh-in-docker.sh)" -- -t robbyrussell

COPY . /ws/CREStereo
WORKDIR /ws/CREStereo
RUN pip3 install --no-cache-dir -r requirements.txt
CMD python3 test.py --model_path ../crestereo_eth3d.mge --left img/test/left.png --right img/test/right.png --size 1024x1536 --output /tmp/disparity.png
