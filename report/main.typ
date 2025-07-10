#import "@preview/elsearticle:0.4.2": *

#show: elsearticle.with(
  title: "Cultural Analytics of Gender Representation in Film: A James Bond Case Study",
  authors: (
    (
      name: "Jakob Lambert-Hartmann",
      affiliation: "Universität Leipzig",
      corr: "wt76geru@studserv.uni-leipzig.de",
      id: "a",
    ),
  ),
  journal: none,
  abstract: none,
  keywords: none,
  format: "review",
  // line-numbering: true,
)

= Introduction
<introduction>

Media has a long history of portraying gender in their stereotypical gender roles. 
Especially older blockbuster movies such as Indiana Jones @Schmidt2022 portray the
man (Dr. Jones) as fearless "tough guy" who gets what he wants.
Women on the other hand are portrayed as love interest, hysterical, and afraid. 
This portrayal seems out of date, and movies today, albeit still not perfect
in their gender representation, would be unlikely to release with such messages.
But when did this change occur?
Was the change gradual, or did some events/movements bring 
forward bigger leaps of change?

In this document we investigate how the portrayal of Women has changed in 
the last decades. 
Specifically, we have chosen the James Bond movie series as subject of research. 
To make our findings quantifiable, the movies were analyzed using pretrained
AI models, notably YOLO @pypiUltralytics and OpenAICLIP @githubGitHubOpenaiCLIP.
Given the computer guided approach, we also investigate whether those models 
are suited for such analysis

In the process we have created an analysis framework,
which can be used for other research questions. 
This resulting framework is corpus agnostic, 
and can be adapted to detect different personality traits. 
It is open source and free to use. 
// TODO cite github




= Dataset
<dataset>

The corpus chosen for our analysis is the James Bond 50th Anniversary DVD Collection. 
It was chosen for several reasons: 

1. *Long Running Series* 
  The Collection contains movies starting from 1962 to 2008 allowing 
  for an observation period of 46 years
2. *Series Containing Many Movies* 
  The Collection contains 22 movies.
  The release interval varies from movie to movie,
  but overall they were released mostly regualarly. 
  The longest time between releases was 6 years, 
  but the average is around 2 years. 
3. *Limiting Confounding Factors* 
  By limiting our observation to one series, we limit confounding factors. 
  James Bond movies all are of the same genre, 
  mostly consist of the same slowly changing cast and crew, 
  and follow similar plots.
  This makes comparisson easier than comparing films of completely different genres 
  (such as Indiana Jones, and Barbie)
4. *Notoriety For Gender Representation*
  The James Bond series notorious for it's representation of women, 
  even coining the term Bond girl. 
  // TODO: citations wikipedia  and birmingham university business school blog
  The prior research on this Series also allows us to validate our methodology, 
  so that it can be applied to other corpora

The movies were digitized using VLC.
Due to copyright restrictions, they are not provided with the code-base. 
The code-base however is corpus agnostic
and can be applied to any other collection of movies.


= Methods
<methods>

To retrieve the portrayal of women from the raw movie data 
several steps were required. 
First, the individual movies were split into their scenes and saved as images. 
Then, object detection was performed to get bounding boxes of persons in the scene images. 
Finally, trait and gender detection was performed. 

== Scene detection

First, the video data was split into scene data. 
This process was automated with PySceneDetect @scenedetectHomePySceneDetect. 
For every scene detected by PySceneDetect,
one frame in the middle of the scene was saved as image 
(see @fig-scene-detection). 
This resulted in 800-2000 frames per movie totalling 31000 frames. 

#figure(
  caption: [scenes detected with PySceneDetect],
  grid(
    columns: 3,
    gutter: 5pt,
    align: alignment.bottom,
    image("./assets/scenes/scene-01.jpg", height: 3cm),
    image("./assets/scenes/scene-02.jpg", height: 3cm),
    image("./assets/scenes/scene-03.jpg", height: 3cm),
    image("./assets/scenes/scene-06.jpg"),
    image("./assets/scenes/scene-07.jpg"),
    image("./assets/scenes/scene-08.jpg"),
  )
)<fig-scene-detection>

== Person detection

The next step applied person detection on every scene frame. 
This was performed with YOLO @pypiUltralytics. 
The pretrained yolo11 model was used, limited to only person detection. 
Furthermore the minimun confidence was set to 0.5. 
This helped to limit the detected persons to the people in the foreground. 

#figure(
  caption: [persons detected with YOLO11],
  grid(
    columns: 3,
    gutter: 5pt,
    align: alignment.bottom,
    image("./assets/persons/scene-001.jpg", height: 3cm),
    image("./assets/persons/scene-002.jpg", height: 3cm),
    image("./assets/persons/scene-003.jpg", height: 3cm),
    image("./assets/persons/scene-004.jpg"),
    image("./assets/persons/scene-005.jpg"),
    image("./assets/persons/scene-006.jpg"),
  )
)<fig-person-detection>

== Trait Detection

This step performs trait and gender detection for each bounding box, 
where YOLO detected a person. 
Trait and Gender detection are both performed using CLIP @githubGitHubOpenaiCLIP. 
The images used are cropped along the bounding boxes to only contain the detected person. 
A batch size of 100 images was chosen for performance. 

Text prompts were used to determine trait and gender (see @prompt-table).
Multiple prompts were used per trait to lessen the effect of 
the individual prompts. 
To get a single prediction value from those, 
the mean of an ensemble was calculated, 
and the softmax was then calculated from both opposing means. 

// TODO: cite https://medium.com/@satojkovic/prompt-ensemble-in-zero-shot-classification-using-clip-e8e1b7b23bb1

#figure(
  caption: [prompt ensembles for trait detection],
  table(
    columns: 3,
    align: left,
    stroke: none,
    table.header([*trait*], [*negative*], [*positive*]),
    table.hline(),

    [*gender*], ["a photo of a woman"], ["a photo of a man"],
    table.hline(),

    [*agency*], 
      list(marker: none,
          ["a person overwhelmed with the situation"],
          ["a person who is passive"],
          ["a submissive person"]),
      list(marker: none,
          ["a person in control of the situation"],
          ["a person who gets what they want"],
          ["a dominant person"]),
    table.hline(),


    [*sexualization*],
        list(marker: none,
            ["a sexualized person"],
            ["a person in a swimsuit"],
            ["a scantily clad person"],
        ),
        list(marker: none,
          ["a professional person"],
          ["a person in formal attire"],
          ["an adequatly dressed person"],
        ),
  )
)<prompt-table>


The following formula (see @formula-scoring) used to tranform 
logits to a score from 0 to 1, 
indicating how strongly the negative traits apply.
To determine the degree of sexualization for instance, 
the mean would be calculated from the logits corresponding to
"a sexualized person", "a person in a swimsuit", "a scantily clad person", 
and the mean would be calculated from the logits corresponding to
"a professional person", "a person in formal attire", "an adequatly dressed person". 
Then the softmax would be calculated from both means. 


#figure(
  caption: [scoring formula],
  $ 
  "softmax_trait" &= sigma(overline("logits"_"trait"^-), 
                           overline("logits"_"trait"^+))\
  $
)<formula-scoring>

#figure(
  image("./assets/traits/traits-000.png"),
)

#figure(
  image("./assets/traits/traits-002.png"),
)


#figure(image("./assets/examples/plot-012.png"),
  caption: [
    While the gender is correclty infered, 
    only the arm is visible. 
    The model still predicts the degree of sexualization and dominance. 
])<fig-background-person>

#figure(image("./assets/examples/plot-008.png"),
  caption: [
    The gender is correctly infered, 
    but the degree of sexualization is clearly wrong. 
    It's seems reasonable to assume that the naked man 
    in the background has a high impact on the degree of sexualization, 
    even though he is not the person of interest. 
])<fig-partial-person>

=== Limitations

This approach has limitations as can bee seen in @fig-partial-person, 
where traits of a partial person are analyzed, and in @fig-background-person, 
where a person in the background impacts the detected traits. 
Additionally, the model and the prompt bias can also contribute to 
a difference between genders. 
However, it should still be possible to monitor change over the period of the 
22 films to observe change of representation within one gender. 


= Findings
<findings>

In contrast to the representation of women in Indiana Jones,
the models predicted that the women in James Bond were not significanly 
more helpless than men (see @fig-helplessness).
This fits the subjective observation we have made, 
in which the women are commonly portrayed as confident and seductive. 
Regarding sexualized representation however, 
women were depicted significantly more sexualized then men (see @fig-sexualization). 
There is a trend toward less sexualization both in men and women. 
Where the sexualization scores for men an women in 1962 were at 0.26 and 0.56,
in 2008 those scores dropped to 0.22 and 0.50.

#figure(image("./assets/scatter-helplessness.png"),
  caption: [degree of helplessness in films by gender including trend lines
])<fig-helplessness>

#figure(image("./assets/scatter-sexualization.png"),
  caption: [degree of sexualization in films by gender including trend lines
])<fig-sexualization>

#figure(
  caption: [degree of sexualization by film title and gender],
  image("./assets/sexualization-title.png")
)<fig-sexualization-title>

When plotting sexualization by titles instead, some outliers become apparent, 
notably _From Russia with Love: 1963_, _On Her Majesty's Secret Service: 1969_, 
and _A View to a Kill: 1987_ (see @fig-sexualization-title).
Those outliers can be well explained by the plot. 
In _From Russia with Love_ James Bond is accompanied 
on his mission by a female spy whereas in other movies most 
women are only aquaintances he sleeps with.
Thus there is more screentime to normalize the value. 
_On Her Majesty's Secret Service_ also strongly differs from other 
James Bond movies, as the 


#figure(image("./assets/sexualization-actor.png"),
caption: [
  Movies with Sean Connery as James Bond actor have the most 
  sexualized representation of women, 
  whereas George Lazenby has the least sexualized representation.
])<fig-sexualization-actor>

#figure(image("./assets/sexualization-director.png"),
  caption: [
    Marc Forster has the most sexualized depiction of women, 
    whereas Peter R. Hunt the least sexualized depiction. 
])<fig-sexualization-director>




= Further Work 

Adding more traits, especially since there was not big difference between 

Adding more Prompts, as some prompts may induce bias. 
A suit for instance is attributed to a man.




// = Prior Works 
//
// There are already essays on the representation of women in the James Bond franchise. 
// But they are either limited in scope,
// only regarding a subset of series @freshessaysPortrayalWomen, 
// or in detail lacking evidence to back up their claims @vintageshowbizBondGirls. 

#bibliography("./bib/bib.bib", style: "ieee")
