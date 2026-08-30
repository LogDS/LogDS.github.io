---
layout: page
title: DatagramDB
description: An implementation of a main-memory database supporting the General Semistructured Model with a Graph Grammar-driven Query Language
img: assets/img/datagramdb.png
importance: 3
category: GitHub
related_publications: true
---

[DatagramDB](https://github.com/datagram-db/datagram-db) implements the Generalised Semistructured Model representing in an uniform representation relational, graph, semistructured {% cite amsdottorato8348 %} and time series {% cite zegadlo %}. This was made possible by generalising KnoBAB's data model so to better support a generic object-oriented data representation {% cite math12172677 %}. 

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/07example_lifts.png" title="Indexed time series in DatagramDB." class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Representing both data and indexing structures for time series in DatagramDB {% cite zegadlo %}.
</div>

By exploiting a declarative and expressing query language leveraging the key concepts from Graph Grammars, we can rewrite sentences parsed from dependency graphs and rewrite them into a syntax-invariant representation. This key technology enables the full logical representation of human-language sentences as advocated by the [LaSSI](/projects/LaSSI/) project. This solution outperforms graph databases implementing common graph query languages, thus motivating the need for our system for processing multiple sentences at a time {% cite math12172677 %}.

The project's [wiki](https://github.com/datagram-db/datagram-db/wiki) provides a full description on the query language and on the possible ways to set-up the project.
