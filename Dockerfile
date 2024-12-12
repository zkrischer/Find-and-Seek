FROM chrischoy/minkowski_engine

RUN apt-get -y update && apt-get -y upgrade

RUN pip install open3d wandb

RUN apt-get install -y firefox nano mesa-utils sudo

RUN adduser --disabled-password --gecos '' newuser \
    && adduser newuser sudo \
    && echo '%sudo ALL=(ALL:ALL) ALL' >> /etc/sudoers


# ENV DEBIAN_FRONTEND=noninteractive
# ENV DISPLAY=:0
RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
  && rm -rf /var/lib/apt/lists/*# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute


RUN mkdir -p /findit
# RUN chown newuser /findit
# USER newuser
WORKDIR /findit
COPY . .



# CMD ["python run_UI.py --user_name=test_user --pretraining_weights=weights/checkpoint1099.pth --dataset_scenes=data/interactive_dataset"]
