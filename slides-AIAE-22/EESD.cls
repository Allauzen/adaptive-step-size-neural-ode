% Mahmoud S. Shaqfa - EESD lab. - EPFL
% Email: mahmoud.shaqfa@epfl.ch

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{EESD}
\LoadClass[aspectratio = 169, 11pt, xcolor={usenames,dvipsnames}]{beamer}
% I used 16:9 aspect ratio of the slides; To get the default (4:3) remove the specifier above in-between the [aspectratio = 169] or simply change the value to 43
% Other possible values are: 1610, 149, 54, 43 and 32.
% \documentclass[aspectratio=1610]{beamer}
% Sets aspect ratio to 16:10, and frame size to 160mm by 100mm.
% 77
% \documentclass[aspectratio=169]{beamer}
% Sets aspect ratio to 16:9, and frame size to 160mm by 90mm.
% \documentclass[aspectratio=149]{beamer}
% Sets aspect ratio to 14:9, and frame size to 140mm by 90mm.
% \documentclass[aspectratio=141]{beamer}
% Sets aspect ratio to 1.41:1, and frame size to 148.5mm by 105mm.
% \documentclass[aspectratio=54]{beamer}
% Sets aspect ratio to 5:4, and frame size to 125mm by 100mm.
% \documentclass[aspectratio=43]{beamer}
% The default aspect ratio and frame size to 128mm by 96mm. You need not specify this option.
% \documentclass[aspectratio=32]{beamer}
% Sets aspect ratio to 3:2, and frame size to 135mm by 90mm.

% ---- My Colors Specifiers ----
\definecolor{mypink}{rgb}{0.97, 0.56, 0.65}
\definecolor{myviolet}{rgb}{0.6, 0.4, 0.8}
\definecolor{myblue}{rgb}{0.61, 0.77, 0.89}

\definecolor{green1}{rgb}{0.00, 0.45, 0.47} % darker green
\definecolor{green2}{rgb}{0.73, 0.88, 0.82} % light green

\definecolor{violet1}{rgb}{0.59, 0.08, 0.39} % darker violet
\definecolor{violet2}{rgb}{0.85, 0.78, 0.85} % light violet


\definecolor{beamerfooter1}{rgb}{0.0, 0.2, 0.4}
\definecolor{beamerfooter2}{rgb}{0.0, 0.2, 0.4}
\definecolor{beamerfooter3}{rgb}{0.0, 0.2, 0.4}

\setbeamercolor*{header color}{fg=white,bg=black}
\setbeamercolor*{footer color1}{fg=white}%,bg=beamerfooter1} % pink
\setbeamercolor*{footer color2}{fg=white}%,bg=beamerfooter2} % dark pink
\setbeamercolor*{footer color3}{fg=white}%,bg=beamerfooter3} % dark red

\setbeamertemplate{blocks}[rounded][shadow=true]

\setbeamercolor{block body}{fg = black, bg = beamerfooter1}
\setbeamercolor{block title}{fg=white, bg=beamerfooter2}

\setbeamercolor{block body example}{fg = black, bg = green2}
\setbeamercolor{block title example}{fg = white, bg = green1}

\setbeamercolor{block body alerted}{fg = black}
\setbeamercolor{block title alerted}{fg=white, bg=beamerfooter1}

\pgfdeclarehorizontalshading[beamerfooter1,beamerfooter2,beamerfooter3]
    {beamer@frametitleshade}{\paperheight}{
    color(0pt)=(beamerfooter3);
    color(0.3333\paperwidth)=(beamerfooter2);
    color(1.056\paperwidth)=(beamerfooter1)
}

\setbeamertemplate{frametitle}{\vspace{20pt}\color{beamerfooter3}\textbf\insertframetitle}

% ---- Bibliography Specifiers ----
\setbeamertemplate{bibliography item}[text] % Regular numbering (Formal)

% ---- Itemize Specifier ----
\setbeamertemplate{itemize items}[square]
\setbeamertemplate{enumerate items}[square]

% ---- Frame Title Specifier ----
\addtobeamertemplate{frametitle}{}{\vspace{0pt}} % increase vspace between the title and text


\newcommand{\rom}[1]{\uppercase\expandafter{\romannumeral #1\relax}} % Add Romans numbering

\usecolortheme[named=beamerfooter3]{structure}
\setbeamertemplate{headline}{}
\setlength{\footnotesep}{0.05cm}

% Table of contents size subsections and subsubsections
\setbeamerfont{subsection in toc}{size=\scriptsize}
\setbeamerfont{subsubsection in toc}{size=\scriptsize}

% Table of contents (Enumeration shapes)
\setbeamertemplate{section in toc}[square]
\setbeamertemplate{subsection in toc}[square]
\setbeamertemplate{subsubsection in toc}[square]

\setbeamercovered{transparent} % Transparent Text When Use "Pauses"!
\setbeamertemplate{navigation symbols}%{default}

\logo{}

\addtobeamertemplate{footnote}{}{\vspace{1.5ex}}

% -------- Special frames ---------
\newcommand{\coverpage}[1]{
{
    \setbeamertemplate{headline}{
        \leavevmode
          \hbox{
            \begin{beamercolorbox}[wd=1.009\textwidth, ht=2.5ex, dp=1.125ex]{}
            \end{beamercolorbox}
          }
        }
\setbeamertemplate{footline}
{
    \leavevmode%
    \setbox\beamer@tempbox=\hbox{%
        \begin{beamercolorbox}[wd=.333333\paperwidth,ht=2.25ex,dp=1ex, center]{footer color3}%
            \usebeamerfont{author in head/foot}\hspace{2ex}\insertshortauthor
        \end{beamercolorbox}%
        \begin{beamercolorbox}[wd=.333333\paperwidth,ht=2.25ex,dp=1ex,center]{footer color2}%
            % \usebeamerfont{title in head/foot}\insertshorttitle
        \end{beamercolorbox}%
        \begin{beamercolorbox}[wd=.333333\paperwidth,ht=2.25ex,dp=1ex,right]{footer color1}%
            \usebeamerfont{title in head/foot}\insertshorttitle{}\hspace*{6em}~~~~~~~~\hspace*{2ex} 
        \end{beamercolorbox}%
        }%
        \beamer@tempdim=\ht\beamer@tempbox%
        \advance\beamer@tempdim by 4pt%
        \begin{pgfpicture}{0pt}{0pt}{\paperwidth}{20pt}
            \pgfpathrectangle{\pgfpointorigin}{\pgfpoint{\paperwidth}{\beamer@tempdim}}
            \pgfusepath{clip}
            \pgftext[left,base]{\pgfuseshading{beamer@frametitleshade}}
        \end{pgfpicture}
        \vskip-\beamer@tempdim%
        \box\beamer@tempbox%    
}%
    \setbeamercolor{background canvas}{}
    \begin{frame}[t, noframenumbering, allowframebreaks]{}
        #1
    \end{frame}
}
}

% Define and customize the headline style of slides
\setbeamertemplate{headline}{%
\leavevmode%
  \hbox{%
    \begin{beamercolorbox}[wd=1.000\textwidth, ht=2.5ex, dp=1.125ex]{header color}%
        \ifx\insertsubsection\empty % no subsection
            {{~~}\insertsection}%
        \else % subsection exists
        \ifx\insertsubsubsection\empty % subsection but no subsubsection
            {~\insertsection \textcolor{white}{$~~\bullet$} ~\S~\insertsubsection}%
        \else % subsection and subsubsection exist
            {~\insertsection \textcolor{white}{$~~\bullet$} ~\S~\insertsubsection \textcolor{white}{$~~\bullet$} ~\S~\insertsubsubsection}%
        \fi
        \fi
    \end{beamercolorbox}%
  }
}

\newcommand{\breakingframe}[1]{
{

    \setbeamertemplate{footline}
{
    \leavevmode%
    \setbox\beamer@tempbox=\hbox{%
        \begin{beamercolorbox}[wd=.333333\paperwidth,ht=2.25ex,dp=1ex, center]{footer color3}%
            \usebeamerfont{author in head/foot}\hspace{2ex}\insertshortauthor
        \end{beamercolorbox}%
        \begin{beamercolorbox}[wd=.333333\paperwidth,ht=2.25ex,dp=1ex,center]{footer color2}%
            \usebeamerfont{title in head/foot}\insertshorttitle
        \end{beamercolorbox}%
        \begin{beamercolorbox}[wd=.333333\paperwidth,ht=2.25ex,dp=1ex,right]{footer color1}%
            \usebeamerfont{date in head/foot}\insertshortdate{}\hspace*{6em}~~~~~~~~\hspace*{2ex} 
        \end{beamercolorbox}%
        }%
        \beamer@tempdim=\ht\beamer@tempbox%
        \advance\beamer@tempdim by 4pt%
        \begin{pgfpicture}{0pt}{0pt}{\paperwidth}{20pt}
            \pgfpathrectangle{\pgfpointorigin}{\pgfpoint{\paperwidth}{\beamer@tempdim}}
            \pgfusepath{clip}
            \pgftext[left,base]{\pgfuseshading{beamer@frametitleshade}}
        \end{pgfpicture}
        \vskip-\beamer@tempdim%
        \box\beamer@tempbox%    
}%

    \setbeamercolor{background canvas}{bg=mypink}
    \begin{frame}[t, noframenumbering, allowframebreaks]{}
        #1
    \end{frame}
}
}

\setbeamertemplate{footline}
{
    \leavevmode%
    \setbox\beamer@tempbox=\hbox{%
        \begin{beamercolorbox}[wd=.333333\paperwidth,ht=2.25ex,dp=1ex]{footer color3}%
            \usebeamerfont{author in head/foot}\hspace{2ex}\insertshortauthor
        \end{beamercolorbox}%
        \begin{beamercolorbox}[wd=.333333\paperwidth,ht=2.25ex,dp=1ex,center]{footer color2}%
            \usebeamerfont{title in head/foot}\insertshorttitle
        \end{beamercolorbox}%
        \begin{beamercolorbox}[wd=.333333\paperwidth,ht=2.25ex,dp=1ex,right]{footer color1}%
            \usebeamerfont{date in head/foot}\insertshortdate{}\hspace*{2em}
            \insertframenumber{} / \inserttotalframenumber\hspace*{2ex} 
        \end{beamercolorbox}%
        }%
        \beamer@tempdim=\ht\beamer@tempbox%
        \advance\beamer@tempdim by 4pt%
        \begin{pgfpicture}{0pt}{0pt}{\paperwidth}{20pt}
            \pgfpathrectangle{\pgfpointorigin}{\pgfpoint{\paperwidth}{\beamer@tempdim}}
            \pgfusepath{clip}
            \pgftext[left,base]{\pgfuseshading{beamer@frametitleshade}}
        \end{pgfpicture}
        \vskip-\beamer@tempdim%
        \box\beamer@tempbox%    
}%
