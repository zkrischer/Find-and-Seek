FROM chrischoy/minkowski_engine

WORKDIR /usr/local/app

COPY . .

RUN apt-get -y update && apt-get -y upgrade

RUN pip install open3d

ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:0

# CMD ["python run_UI.py --user_name=test_user --pretraining_weights=weights/checkpoint1099.pth --dataset_scenes=data/interactive_dataset"]
