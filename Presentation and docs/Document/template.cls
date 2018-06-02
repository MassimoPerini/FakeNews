\LoadClass[a4paper,12pt,oneside,dvipsnames]{book} % Classe du document

%--------------------- Packages ------------------------

\RequirePackage[english]{babel} %Langue du document
\RequirePackage[utf8]{inputenc} %Caractères spéciaux
\RequirePackage[section]{placeins}%Pour placement de section
\RequirePackage[T1]{fontenc} %Quelques lettres qui sont pas inclus dans UTF-8
\RequirePackage{mathtools} %Paquet pour des équations et symboles mathématiques
\RequirePackage{siunitx} %Pour écrire avec la notation scientifique (Ex.: \num{2e+9})
\RequirePackage{float} %Pour placement d'images
\RequirePackage{graphicx} %Paquet pour insérer des images
\RequirePackage[justification=centering]{caption} %Pour les légendes centralisées
\RequirePackage{subcaption}
\RequirePackage{wallpaper}
\RequirePackage{nomencl}
%\makenomenclature
\RequirePackage{fancyhdr}
%\pagestyle{fancy}
%\fancyheadoffset{1cm}
%\setlength{\headheight}{2cm}
\RequirePackage{url}
\RequirePackage[hidelinks]{hyperref}%Paquet pour insérer légendes dans des sous-figures comme Figure 1a, 1b
\RequirePackage[left=2.5cm,right=2.5cm,top=2cm,bottom=3.5cm]{geometry} %Configuration de la page 4.4 4.4

\RequirePackage{titlesec}
\RequirePackage{graphicx}
\RequirePackage{listings}
\RequirePackage{hyperref}
\RequirePackage{xcolor}
\RequirePackage{fancyhdr}
\RequirePackage{lastpage}
\RequirePackage{etoolbox}
\RequirePackage{caption}
\titleformat{\chapter}[display]{\normalfont\bfseries}{}{0pt}{\Huge}



%-------------------- Informations sur le rapport ----------------------

\newcommand{\UE}[1]{\renewcommand{\UE}{#1}}
\newcommand{\subject}[1]{\renewcommand{\subject}{#1}}
\newcommand{\titlec}[1]{\renewcommand{\titlec}{#1}}
\newcommand{\supervisor}[1]{\renewcommand{\supervisor}{#1}}
\newcommand{\authors}[1]{\renewcommand{\authors}{#1}}

\newcommand{\createmargins}{
\makenomenclature
\pagestyle{fancy}
\fancyheadoffset{1cm}
\setlength{\headheight}{2cm}
\lhead{\includegraphics[scale=0.06]{images/polimi_logo_text.png}} %Affichage de l'image au top de la page
\rhead{\nouppercase{\leftmark}}
\rfoot{\thepage}
\cfoot{\textbf{\titlec}}
\lfoot{\UE}
}

\newcommand{\createcover}{
\begin{titlepage}

\ThisLRCornerWallPaper{0.6}{images/first_bg.jpg}
	\centering %Centraliser le contenu
	\includegraphics[width=0.28\textwidth]{images/polimi_main_logo.png}\par\vspace{1cm} %Insertion du logo
	{\scshape\LARGE Politecnico di Milano \par} %Nom de l'université
	\vspace{1cm}%Espace de 1,5cm
	{\scshape\Large \UE \\ \subject \\ Final Report\par} %sous-titlec
	\vspace{1cm}%Espace de 1cm
    \rule{\linewidth}{0.2 mm} \\[0.4 cm]
	{\huge\bfseries \titlec \par} \
    \rule{\linewidth}{0.2 mm} \\[1 cm]
	\vspace{1cm}%Espace de 3cm
    
	\begin{minipage}{0.5\textwidth} %Minipage pour faire deux colonnes
		\begin{flushleft} \large %Envoyer à gauche
		\emph{\textbf{Authors :}}\\ %Pour le titlec au dessus des noms à gauche
        \authors\\ %Remplacer pour chacun
		\end{flushleft}
	\end{minipage}
	~
	\begin{minipage}{0.4\textwidth}
		\begin{flushright} \large
		\emph{\textbf{Teachers :}} \\
		 \supervisor \\
		\end{flushright}
	\end{minipage}\\[4cm]
    
	\vfill
	{\large \today\par} %Affichage de la date

\end{titlepage}
}

%\newcommand{\tabledematieres}{
%\tableofcontents
%\newpage
%}

\newcommand{\insererfigure}[4]{
\begin{figure}[ht]
\centering
\includegraphics[height=#2]{#1}
\caption{#3}
\label{fig: #4}
\end{figure}
}




