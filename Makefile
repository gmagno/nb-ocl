
#--- SETUP -----------------------------------------------#

SHELL := /bin/bash

.PHONY: clean
clean:
	find . \( -name __pycache__ \
		-o -name "*.pyc" \
		-o -name .pytest_cache \
		-o -path "./dist" \
		-o -path "./nb_ocl.egg-info" \
		! -path ".venv/*" \
	\) -exec rm -rf {} +


#--- DEV -------------------------------------------------#




#--- DOCKER ----------------------------------------------#

.PHONY: docker-build-all
docker-build-all:
	docker build -f docker/Dockerfile_all -t godwitlabs/nbm_all:0.1 .

.PHONY: docker-run-bash-all
docker-run-bash-all:
	xhost +si:localuser:root\
		&& docker run \
			--device=/dev/input/ \
			--device=/dev/dri/ \
			--runtime=nvidia \
			-ti --rm \
			-e DISPLAY \
			-v /tmp/.X11-unix:/tmp/.X11-unix \
			-v ${PWD}/nbm:/home/app/nbm \
			godwitlabs/nbm_all:0.1 bash\
		&& xhost -si:localuser:root

.PHONY: docker-run-all
docker-run-all:
	xhost +si:localuser:root\
		&& docker run \
			--device=/dev/input/ \
			--device=/dev/dri/ \
			--runtime=nvidia \
			-ti --rm \
			-e DISPLAY \
			-v /tmp/.X11-unix:/tmp/.X11-unix \
			godwitlabs/nbm:0.1 \
		&& xhost -si:localuser:root

.PHONY: docker-build-icpu
docker-build-icpu:
	docker build -f docker/Dockerfile_icpu -t godwitlabs/nbm_icpu:0.1 .

.PHONY: docker-run-bash-icpu
docker-run-bash-icpu:
	xhost +si:localuser:root\
		&& docker run \
			--device=/dev/input/ \
			--device=/dev/dri/ \
			-ti --rm \
			-e DISPLAY \
			-v /tmp/.X11-unix:/tmp/.X11-unix \
			godwitlabs/nbm_icpu:0.1 bash \
		&& xhost -si:localuser:root

.PHONY: docker-build-igpu
docker-build-igpu:
	docker build -f docker/Dockerfile_igpu -t godwitlabs/nb_cl:0.1 .

.PHONY: docker-run-bash-igpu
docker-run-bash-igpu:
	xhost +si:localuser:root\
		&& docker run \
			--device=/dev/input/ \
			--device=/dev/dri/ \
			-ti --rm \
			-e DISPLAY \
			-v /tmp/.X11-unix:/tmp/.X11-unix \
			godwitlabs/nb_cl:0.1 bash \
		&& xhost -si:localuser:root
