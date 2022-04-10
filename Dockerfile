FROM diyer22/cv2_py38
ENV LC_ALL=C.UTF-8 LANG=C.UTF-8 DEBIAN_FRONTEND=noninteractive FORCE_CUDA="1"

RUN apt update -y

RUN pip3 install --no-cache-dir -U pip wheel setuptools 
RUN pip3 install --no-cache-dir MegEngine numpy Pillow

# Install debug tools
RUN pip3 install --no-cache-dir ipython && apt install -y vim
# && sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.2/zsh-in-docker.sh)" -- -t robbyrussell

COPY . /ws/CREStereo
WORKDIR /ws/CREStereo
# RUN pip3 install --no-cache-dir -r requirements.txt
CMD python3 test.py --model_path crestereo_eth3d.mge --left img/test/left.png --right img/test/right.png --size 1024x1536 --output /tmp/disparity.png
