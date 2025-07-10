---
graphics: true
title: "Cultural Analytics of Gender Representation in Film: A James Bond Case Study"
titlepage: false
author: Jakob Lambert-Hartmann
bibliography: bib.bib
---

#### Research Question 

Media has a long history of portraying gender in their stereotypical gender roles. 
Especially older blockbuster movies such as Indiana Jones [@Schmidt2022] portray the
man (Dr. Jones) as fearless "tough guy" who gets what he wants.
Women on the other hand are portrayed as love interest, hysterical, and afraid. 
This portrayal seems out of date, and movies today, albeit still not perfect
in their gender representation, would be unlikely to release with such messages.
But when did this change occur?
Was the change gradual, or did some events/movements bring 
forward bigger leaps of change?

#### Body of Research

To limit confounding factors, we have chosen to research gender representation 
within one long-running series: James Bond. 
The series first movie was released in 1962 and since then new 
movies were released in regular intervals. 
The movies within the series share the same genre, 
and with the small changes to cast, crew, and plot over the years, 
they are a great opportunity to observe development over a long period of time. 

#### Prior Works 

There are already essays on the representation of women in the James Bond franchise. 
But they are either limited in scope,
only regarding a subset of series [@freshessaysPortrayalWomen], 
or in detail lacking evidence to back up their claims [@vintageshowbizBondGirls]. 

#### Dataset

The corpus chosen consists of the James Bond 50th Anniversary DVD Collection. 
It was digitized as a private copy.
Due to copyright restrictions, it will not be distributed with the codebase. 
But the code-base is corpus agnostic, so it is possible to run it with any other movies.

#### Methods

The methods for this research consist of three main components 
(pipeline can be seen in figure \ref{pipeline}): 

1. **Scene detection:**
    splitting the movies into scenes and saving one frame per scene. 
    Achieved with PySceneDetect [@scenedetectHomePySceneDetect]
2. **Person detection**
    Which detects persons and their bounding boxes for each scene-frame.
    Achieved with YOLO [@pypiUltralytics]
3. **Trais detection**
    Which detects the gender and traits of a detected person. 
    This is Achieved with CLIP [@githubGitHubOpenaiCLIP]

#### Limitations 

Due to missing availability of the movie scripts,
only visual features of the movies are taken into consideration. 
Information that is not visually apparent will thus not be considered 
in this study. 

Additionally, while choosing only one movie series helps 
to reduce confounding factors, this limits generalizability. 
Changes in gender portrayal specific to James Bond, 
may not necessarily be indicative or a larger cultural shift.


\newpage
# Attachment

\begin{figure}
    \centering
    \includegraphics[width=0.7\textwidth]{./images/pipeline.png}
    \caption{Data pipeline: Movie is split into scene-frames. 
            Person detection is run for each scene-frame. 
            Trait and gender prediction is run for each person detected in a scene-frame.}
    \label{pipeline}
\end{figure}


