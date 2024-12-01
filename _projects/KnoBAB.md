---
layout: page
title: KnoBAB
description: A fast LTLf Log-SAT Solver and miner with Data Payload
img: assets/img/KnoBAB2.png
importance: 1
category: GitHub
related_publications: true
---

KnoBAB provides an ecosystem for temporal data analysis, supporting both multivariate time series analysis and log data.
Both of these data models are represented into an internal columnar database enabling fast data querying and mining, while
outperforming state of the art algorithms.


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/benchmark_burattin_scenario_1.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/Datamer.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/All_acc.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Bechmarks showing that our solutions outperform state of the art data-aware formal verification (left: {% cite info14030173 %}), formal synthesis (center: {% cite grades23 %}), and multivariate time series classification (right: {% cite ideas2024a %}) solutions.
</div>
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="https://www.mdpi.com/information/information-14-00173/article_deploy/html/images/information-14-00173-g002.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Temporal data is loaded into a columnar database (above), while declarative queries are rewritten into algebraic temporal operators (below) {% cite info14030173 %}.
</div>

While specification mining, formal verification, and formal synthesis tasks need the prompt interaction with C++ code, we are working towards fully-supporting the Multivariate Time Series classification task in Python, for which we now provide a separate Python pipeline via [EMeriTAte](https://github.com/datagram-db/knobab/tree/v3.x/EMeriTAte).

## Invited Talks and Presentations

* Giacomo Bergami. "[Apprentissage Relationnel: La Logique et les Algorithmes remettent en cause les techniques d'apprentissage profond](https://drive.google.com/file/d/1DCCESlIrs_2S0suFWjv7lVtWms02UPhb/view?usp=sharing)". LIFO – Universitè d’Orléans, 24ᵗʰ of April, 2024.
* Giacomo Bergami. "[KnoBAB: Making Logic Fast](https://github.com/datagram-db/knobab/blob/v3.x/InvitedTalks/AAAI23-SSS.pdf)" [LT<sub>f</sub>@AAAI-SSS](https://ltlf-symposium.github.io/program/), 29ᵗʰ of March, 2023
* Giacomo Bergami. "[KnoBAB: Fast Business Process Management with Temporal Reasoning](https://github.com/datagram-db/knobab/blob/v3.x/InvitedTalks/MeetUp23.pdf)" [North East Data Scientist MeetUp](https://www.techdiary.co.uk/event/7393b4f3-01aa-4627-9d03-e2ae9fa1a345/visit), 16ᵗʰ of March, 2023

