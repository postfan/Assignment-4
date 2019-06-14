position.pdf: Assignment_3.py
	python Assignment_3.py

velocity.pdf: Assignment_3.py
	python Assignment_3.py 

position_error.pdf: Assignment_3.py
	python Assignment_3.py 

velocity_error.pdf: Assignment_3.py
	python Assignment_3.py 

template.pdf: position.pdf velocity.pdf position_error.pdf velocity_error.pdf template.tex
	pdflatex template.tex


