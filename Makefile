setup:
	python3 -m venv ../.RedWine_Quality_Prediction

install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt

configure:
	make setup
	make install
