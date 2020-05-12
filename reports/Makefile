.PHONY: wc

report.pdf: report.tex anthology.bib acl2020.bib references.bib
	-/bin/rm report.bbl
	pdflatex report
	-bibtex report
	pdflatex report
	pdflatex report

ltg.bib: ./ltg/bib/etc/*.bib ./ltg/bib/?/*.bib
	make -C ./ltg/bib cat
	cp ./ltg/bib/ltg.bib .

update:
	svn update;
	( cd ./ltg; svn update; )

wc:
	egrep -v '^%' report.tex \
        | awk '/\\begin\{document\}/ {body=1;} {if(body) print;}' \
        | wc -w
