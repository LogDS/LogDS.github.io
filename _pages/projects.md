---
layout: page
title: projects
permalink: /projects/
description: A collection of active projects under the LogDS Group
nav: true
nav_order: 3
display_categories: [GitHub, Research, fun]
horizontal: false
---

LogDS group revolves around very theoretical research projects having an immediate fallback in practical use case scenarios. Some of my interests are the following:

  * Theoretical Background:
     * Verified and Explainable Artificial Intelligence (Temporal, Natural Language Processing)
     * Multimodal and Object-Oriented Database Theory and Query Languages
     * Inconsistency Metrics
  * Practical Use Case Scenarios:
     * Recommendation Systems
     * Data Analytics
     * Evolutionary Microservice Architectures for/with Data Science
     * Zero Model-Fitting (no Neural-Network, no Bayesian Model, no Markov Logic Network) AI

My personal website provides a [full list](https://jackbergus.github.io/projects/#projects-as-a-student) of older projects I was involved in as a student.

<!-- pages/projects.md -->
<div class="projects">
{% if site.enable_project_categories and page.display_categories %}
  <!-- Display categorized projects -->
  {% for category in page.display_categories %}
  <a id="{{ category }}" href=".#{{ category }}">
    <h2 class="category">{{ category }}</h2>
  </a>
  {% assign categorized_projects = site.projects | where: "category", category %}
  {% assign sorted_projects = categorized_projects | sort: "importance" %}
  <!-- Generate cards for each project -->
  {% if page.horizontal %}
  <div class="container">
    <div class="row row-cols-1 row-cols-md-2">
    {% for project in sorted_projects %}
      {% include projects_horizontal.liquid %}
    {% endfor %}
    </div>
  </div>
  {% else %}
  <div class="row row-cols-1 row-cols-md-3">
    {% for project in sorted_projects %}
      {% include projects.liquid %}
    {% endfor %}
  </div>
  {% endif %}
  {% endfor %}

{% else %}

<!-- Display projects without categories -->

{% assign sorted_projects = site.projects | sort: "importance" %}

  <!-- Generate cards for each project -->

{% if page.horizontal %}

  <div class="container">
    <div class="row row-cols-1 row-cols-md-2">
    {% for project in sorted_projects %}
      {% include projects_horizontal.liquid %}
    {% endfor %}
    </div>
  </div>
  {% else %}
  <div class="row row-cols-1 row-cols-md-3">
    {% for project in sorted_projects %}
      {% include projects.liquid %}
    {% endfor %}
  </div>
  {% endif %}
{% endif %}
</div>
