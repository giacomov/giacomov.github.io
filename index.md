---
layout: default
---


<img src="images/avatar.jpg" alt="avatar" style="width: 100px;"/>



* [Academic CV with publications](docs/GiacomoVianello_Nov2017.pdf)



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






