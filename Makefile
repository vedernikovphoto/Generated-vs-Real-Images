# Paths
PREPROCESS_SCRIPT = src/preprocessing/extract_frames.py

install:
	pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

install_venv:
	@bash setup.sh

# Download Real images dataset 1: ABOshipsDataset
download_real_data_1:
	curl -L "https://zenodo.org/records/4736931/files/ABOshipsDataset.zip?download=1" -o images.zip
	python -c "import zipfile; zipfile.ZipFile('images.zip').extractall('data/real')"
	rm images.zip
	python src/preprocessing/clean_abo_data.py --top_folder data/real/ABOshipsDataset/Seaships --frame_step 3

# Download Real images dataset 2: Singapore Maritime Dataset (Part: Visible On-Shore)
download_real_data_2:
	gdown "https://drive.google.com/uc?id=1HnHyQzhzzDlYh15y9_K1mNZX3grlSDMM" -O dataset.rar
	unrar x dataset.rar data/real2/
	rm dataset.rar
	python $(PREPROCESS_SCRIPT) --video_folder data/real2/VIS_Onshore/Videos --output_folder data/real2/images --frame_step 30

# Download Real images dataset 3: Singapore Maritime Dataset (Part: Visible On-Board)
download_real_data_3:
	gdown "https://drive.google.com/uc?id=13Peb4GUNVotQXEHFq-ARCJ-QFmeAgcMG" -O dataset.zip
	python -c "import zipfile; zipfile.ZipFile('dataset.zip').extractall('data/real3')"
	rm dataset.zip
	python $(PREPROCESS_SCRIPT) --video_folder data/real3/VIS_Onboard/Videos --output_folder data/real3/images --frame_step 10

# Download Real images dataset 4: Singapore Maritime Dataset (Part: Near-infra Red On-Shore)
download_real_data_4:
	gdown "https://drive.google.com/uc?id=13wKWzHqkDQHMHjfuUWjgzwhTrnMoEE8P" -O dataset.rar
	unrar x dataset.rar data/real4/
	rm dataset.rar
	python $(PREPROCESS_SCRIPT) --video_folder data/real4/NIR/Videos --output_folder data/real4/images --frame_step 30

# Download Generated images SimuShips dataset
download_generated_data:
	curl -L "https://zenodo.org/records/7003924/files/images.zip?download=1" -o images.zip
	python -c "import zipfile; zipfile.ZipFile('images.zip').extractall('data/generated')"
	rm images.zip

train:
	export PYTHONPATH=$(PWD) && python src/train.py config/config.yaml

inference:
	export PYTHONPATH=$(PWD) && python src/inference.py
