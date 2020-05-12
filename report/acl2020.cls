\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{acl2020}[Targeted Sentiment Analysis]
\LoadClass[11pt,english,a4paper]{article}

% ACL
\usepackage{natbib}
\usepackage[hyperref]{acl2020}
\usepackage{times}
\usepackage{latexsym}
\renewcommand{\UrlFont}{\ttfamily\small}
\usepackage{booktabs}
\newcommand{\BibTex}{B\textsc{ib}\Tex}

% may be commented out: improve layout if kept
\usepackage{microtype}

% Språk og tegn
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage[norsk]{babel}
\RequirePackage{tikz}
\RequirePackage{parskip}

% Fonter
\RequirePackage{tgpagella}
\RequirePackage[small,euler-digits]{eulervm}

% Lister
\RequirePackage{enumitem}
\newlist{deloppgaver}{enumerate}{1}
\setlist[deloppgaver]{label={\textbf{(\alph*)}},leftmargin=*}

% Matematikk
\RequirePackage{amsmath,amssymb}
\RequirePackage[mathscr]{euscript}
\RequirePackage{siunitx}

% Noen nyttige forkortelser
\newcommand{\set}[1]{\ensuremath{\{#1\}}}
\newcommand{\tuple}[1]{\ensuremath{\langle #1 \rangle}}
\newcommand{\imp}{\ensuremath{\rightarrow}}
\newcommand{\union}{\ensuremath{\cup}}
\newcommand{\snitt}{\ensuremath{\cap}}
\newcommand{\listfunc}{ :\,: }
\newcommand{\funcname}[2]{\begin{scriptsize}#1\end{scriptsize}\ensuremath{#2}}
\newcommand{\lequal}{\Leftrightarrow}

% Annet
\RequirePackage{graphicx} 

% Referanser
% \usepackage{babel, csquotes}
% \usepackage[backend=biber,sortcites]{biblatex}

