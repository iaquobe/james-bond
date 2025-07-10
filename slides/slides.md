---
title: "Cultural Analytics of Gender Representation in Film: A James Bond Case Study"
author: "Jakob Lambert-Hartmann"
date: "2025"
graphics: true
theme: metropolis
---


## Motivation: Indiana Jones


::: columns
:::: {.column width=45%}
- Indiana Jones: 
    - "Tough guy"
    - Fearless
    - Gets what he wants
- Female Characters:
    - Love interest
    - Always afraid 
    - Hysterical
::::
:::: {.column width=55%}
\begin{figure}
\centering
\begin{tikzpicture}
\node[anchor=south west, inner sep=0] (img3) at (0, 5) {\includegraphics[width=4cm]{./images/th-3753753526.jpg}};
\node[anchor=south west, inner sep=0] (img1) at (0.5, 3.5) {\includegraphics[width=4cm]{./images/th-39866010.jpg}};
\node[anchor=south west, inner sep=0] (img3) at (1, 2) {\includegraphics[width=4cm]{./images/th-1616846030.jpg}};
\node[anchor=south west, inner sep=0] (img2) at (2, 1) {\includegraphics[width=4cm]{./images/th-144364692.jpg}};
\end{tikzpicture}
\end{figure}
::::
:::


## Research Question: When Did This Change?

::: columns
:::: {.column width=55%}
- When did this change? 
- Was it a gradual change or a sudden one?
- Is change correlated to some events/movements?
- Is change observable with AI tools?
::::
:::: {.column width=45%}
\begin{figure}
\centering
\includegraphics[width=4cm]{./images/bart-ai.jpeg}
\end{figure}
::::
:::

## Corpus: James Bond 50th Anniversary Collection


::: columns
:::: {.column width=45%}
- Long-running franchise: 1962-2008
- Near infinite amount of movies (22) with: 
    - almost same crew
    - almost same cast
    - almost same plot
- Bonus: maybe I can deduce the cost from taxes later
::::
:::: {.column width=45%}
\includegraphics[width=4cm]{./images/boxset.png}
::::
:::

## Methods

::: columns
:::: {.column width=55%}
1. **"Privatkopie":**\
    DVD $\rightarrow$ usable format (VLC)
2. **Scene detection:**\
    movies $\rightarrow$ scene-frames (PySceneDetect)
3. **Person detection:**\
    scene-frame $\rightarrow$ persons (YOLO)
4. **Trait/gender detection:** \  
    person $\rightarrow$ trait + gender (CLIP)
5. **Plotting:**\
    data $\rightarrow$ nice plots (matplotlib)
::::
:::: {.column width=40%}
\includegraphics[width=4cm]{./images/pipeline.png}
::::
:::


## Limitations


::: columns
:::: {.column width=55%}
1. Bias in models
2. James Bond may not be representative of other movies
3. Only visual aspects of movie (but Bond is not subtle)
::::
:::: {.column width=40%}
\includegraphics[width=4cm]{./images/ursula.jpg}
::::
:::


# Thanks! Input? Feedback?
