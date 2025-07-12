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
)


#pagebreak()
= Introduction
<introduction>

Media has a long history of portraying gender in their stereotypical gender roles. 
Especially older blockbuster films such as Indiana Jones @Schmidt2022 portray the
man (Dr. Jones) as fearless "tough guy" who gets what he wants.
Women on the other hand are portrayed as love interest, hysterical, and afraid. 
This portrayal seems out of date, and films today, albeit still not perfect
in their gender representation, would be unlikely to release with such messages.
But when did this change occur?
Was the change gradual, or did some events/movements bring 
forward bigger leaps of change?

In this document we investigate how the portrayal of women has changed in 
the last decades. 
Specifically, we have chosen the James Bond film series as subject of research. 
To make our findings quantifiable, the films were analyzed using pretrained
AI models, notably YOLO @pypiUltralytics and CLIP @githubGitHubOpenaiCLIP.
Given the computer guided approach, we also investigate whether 
this method yields valid and meaningful results. 

In the process we have created an analysis framework,
which can be used for other research questions. 
This resulting framework is corpus agnostic, 
and can be adapted to detect different personality traits. 
It is open source and free to use @githubGitHubIaquobejamesbond.

= Research Question 

In this document we question whether women 
in James Bond are depicted in stereotypical gender roles, 
and whether this depiction has changed over the years. 
Based on our observation from Indiana Jones, 
we have chosen to measure two traits: 
Sexualization and Agency 
(describing whether a person is in control of the situation,
or whether the person is helpless).

= Related Works

Neuendorf (2009) @springerShakenStirred
analyzed the portrayal of women in the first 20 James Bond films. 
A trend was discovered that women in later movies were 
subjected to more violence and were involved in more sexual activities. 
However the authors also pointed out that the roles of women 
changed from very limited roles to more autonomous and active roles. 
In our study we hope eliminate the factor of the changing roles, 
as our analysis focuses on the appearance rather than the activity. 

= Dataset
<dataset>

The corpus chosen for our analysis is the James Bond 50th Anniversary 
DVD Collection. 
It was chosen for several reasons: 
*Long Running Series:*
The collection contains films starting from 1962 to 2008 allowing 
for an observation period of 46 years.
*Series Containing Many Films:*
The Collection contains 22 films.
The release interval varies from film to movie,
but overall they were released mostly regularly. 
The longest time between releases was 6 years, 
but the average is around 2 years. 
*Limiting Confounding Factors:*
By limiting our observation to one series, we limit confounding factors. 
James Bond films all are of the same genre, 
mostly consist of the same slowly changing cast and crew, 
and follow similar plots.
This makes comparison easier than comparing films of completely different genres 
(such as Indiana Jones, and Barbie).
*Notoriety For Gender Representation:*
The James Bond series is notorious for its representation of women, 
even coining the term Bond girl @wikipediaBondGirl. 
The prior research on this Series also allows us to validate our methodology, 
to test whether it can be applied to other corpora.
The choice of only using James Bond movies however also 
limits the generalizability to other films. 

The films were digitized using VLC.
Due to copyright restrictions, they are not provided with the code-base. 
The code-base however is corpus agnostic
and can be applied to any other collection of films.


= Methods
<methods>

To retrieve the portrayal of women from the raw film data 
several steps were required. 
First, the individual films were split into their scenes and saved as images. 
Then, object detection was performed to get bounding boxes of persons in the scene images. 
Finally, trait and gender detection was performed on the detected bounding boxes. 

== Scene detection

First, the video data was split into scene data. 
We used PySceneDetect @scenedetectHomePySceneDetect to automate this process. 
For every scene detected by PySceneDetect,
one frame in the middle of the scene was saved as image.
This resulted in 800-2000 frames per film totaling 31000 frames. 


== Person detection

The next step applied person detection on every scene frame. 
This was performed with YOLO @pypiUltralytics. 
The pretrained yolo11 model was used, limited to only person detection. 
Furthermore the minimum confidence was set to 0.5. 
This was used in order to limit person detection to people in the foreground.

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

== Trait and Gender Detection

The last step in the pipeline performs trait and gender detection
for each person bounding box, which YOLO detected. 
Trait and Gender detection are both performed using CLIP @githubGitHubOpenaiCLIP. 
*Images* used are cropped along the bounding boxes to only 
contain the detected person and the images were 
processed in a batch size of 100 images.
*Text* prompts were used to determine gender, 
and prompt ensembles were used for traits to 
improve performance @PromptEnsemble.

#table(
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
          ["an adequately dressed person"],
        ),
  )


The following formula used to transform 
logits to a score from 0 to 1, 
indicating how strongly the negative traits apply.
To determine the degree of sexualization for instance, 
the mean would be calculated from the logits $a$ corresponding to
one ensemble.


$ 
"score" &= sigma(overline("logits"_"trait"^-), 
                         overline("logits"_"trait"^+))\

arrow("a"_-) &= vec("a"_"\"sexualized\"", 
                   "a"_"\"swimsuit\"", 
                   "a"_"\"scantily clad\"")\

arrow("a"_+) &= vec("a"_"\"professional\"", 
                   "a"_"\"formal attire\"", 
                   "a"_"\"adequatly dressed\"")\

"score" &= sigma(overline("a"_+), overline("a"_-))

$

=== Limitations

This approach has limitations as can bee seen in @fig-partial-person, 
where traits of a partial person are analyzed, and in @fig-background-person, 
where a person in the background impacts the detected traits. 
Additionally, the model and the prompt bias can also contribute to 
a difference between genders. 
Finally, the chosen prompt ensembles describe vague concepts such as "sexualized"
or "dominant", which may be difficult for a model to understand. 
A change to concrete prompts may increase the performance. 
However, by averaging over multiple such terms, 
we hope that we can still achieve meaningful results. 
We will investigate the validity of the prompts later in @findings


#grid(
  columns: 2,
  gutter: 10pt,
  grid.cell([ 
    #figure(image("./assets/traits/traits-002.png"), 
      caption: [
        The model correctly recognizes five persons in the image. 
        It recognizes the woman and interprets her as dominant and 
        professional. This aligns with the film, 
        as the woman is the leader of the depicted group
      ]
    )
  ]),
  grid.cell([
    #figure(image("./assets/examples/plot-008.png"),
    caption: [
      The gender is correctly inferred, 
      but the degree of sexualization is clearly wrong. 
      It seems the naked man in the background contributes to this score, 
      even though he is not the person of interest
  ])<fig-partial-person>]),
  grid.cell([
    #figure(image("./assets/traits/traits-000.png"),
      caption: [The model recognizes to persons in the image. 
      It recognizes the woman as woman
      and that her depiction is sexualized
    ]
  )]),
  grid.cell([
    #figure(image("./assets/examples/plot-012.png"),
      caption: [
        While the gender is correctly inferred, 
        only the arm is visible. 
        The model still predicts the degree of sexualization and dominance. 
    ])<fig-background-person>
  ])
)


= Findings
<findings>
<findings>

We visualized the agency score by gender and film
and fitted a linear regression model to the data (see @fig-helplessness). 
We found that, in contrast to the representation of women in Indiana Jones,
the models predicted that women in James Bond were not significantly 
more helpless than men.
This fits the subjective observation we have made, 
that women in James Bond are commonly portrayed as confident and seductive. 

#figure(
grid(
    columns: 2,
    gutter: 10pt,
    grid.cell([
      #figure(image("./assets/scatter-helplessness.png"),
        caption: [
          degree of helplessness in films by gender. 
          Observed data and fitted linear regression model
      ])<fig-helplessness>]),

    grid.cell([
      #figure(image("./assets/scatter-sexualization.png"),
        caption: [
          degree of sexualization in films by gender.  
          Observed data and fitted linear regression model
      ])<fig-sexualization>]),

  )
)

For the sexualized representation however, 
there was a stark difference between gender (see @fig-sexualization). 
The mean sexualization score in men is 0.27,
whereas in women it is 0.56.
The Linear regression model shows a trend toward
less sexualization both in men and women. 
Where the sexualization scores for men and women in 1962 were at 0.28 and 0.60,
in 2008 those scores dropped to 0.26 and 0.53.




#figure(image("./assets/sexualization-title.png"))

To validated our methods we plotted sexualization scores by titles 
where some outliers became apparent, 
notably _From Russia with Love: 1963_
and _On Her Majesty's Secret Service: 1969_.
Those outliers can be well explained by the film plots. 
In _From Russia with Love_ James Bond is accompanied 
on his mission by a female spy whereas in other films most 
women are only acquaintances he sleeps with.
Thus, there is more screen-time to normalize the sexualization score of women. 
In _On Her Majesty's Secret Service_ James Bond marries a woman. 
The film is thus more romantic and less sexualized than others in the series


#figure(image("./assets/sexualization-actor.png"))

To further validate our methods we investigated the 
mean sexualization score by James Bond Actor @theweekDanielCraig. 
Daniel Craig is deemed to be the least sexualized James Bond authors, 
however with our methods this could not be validated. 
The corpus used for this observation however only contained two 
films (_Casino Royale_ and _Quantum of Solace_), 
missing _Skyfall_, _Spectre_, and _No Time to Die_. 
Including those missing titles into the analysis might yield different results. 


= Conclusion 

In conclusion, we found that there is a slight trend towards less sexualization 
in the James Bond film Series. 
This trend exists both in men and women, but is weaker in men. 
We could not observe sudden changes, 
which could have been expected 
during waves of feminism for instance @feminismwaves. 
Instead the change was gradual and slow. 

Outliers helped in validating our methods, 
however the strongly varying score also highlights,
that the limited sample size of 22 
is not sufficient to generalize from the portrayal of women in James Bond 
to the overall portrayal of women in film. 
Further studies with larger corpora are warranted 
in order to monitor general trends.
Other long-running film series would be well suited to 
limit confounding factors.


#bibliography("./bib/bib.bib", style: "ieee")
