---
layout: default
---

|<img src="/images/avatar.jpg" alt="avatar" style="width: 100px;"/>|[Academic CV with publications](docs/GiacomoVianello_Nov2017.pdf)|

# Research interests

* Searches for transients in astronomical data: gamma-ray, x-ray, optical
* Gamma-ray Bursts (GRBs): spectral and temporal properties, population studies, modeling
* GRBs as electromagnetic counterparts to Gravitational Wave events
* Multi-wavelength and multi-messenger astrophysics
* X-rays dust scattering: dust models, X-ray halos, X-ray rings
* Astrophysics instrumentation
* Data analysis methods and software: Bayesian methods, machine learning, Maximum Likelihood, numerical methods
* Statistical methods for astrophysics


<div class="posts">
  {% for post in site.posts %}
    <article class="post">

      <h1><a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a></h1>

      <div class="entry">
        {{ post.excerpt }}
      </div>

      <a href="{{ site.baseurl }}{{ post.url }}" class="read-more">Read More</a>
    </article>
  {% endfor %}
</div>






