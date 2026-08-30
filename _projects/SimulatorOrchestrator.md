---
layout: page
title: SimulatorOrchestrator
description: A simulator for osmotic and 6G smart-city architectures.
img:
importance: 3
category: GitHub
img: assets/img/SOo.png
related_publications: true
---

[SimulatorOrchestrator](https://github.com/LogDS/SimulatorOrchestrator) simulates the [Presentation](https://en.wikipedia.org/wiki/Presentation_layer) and the [Session](https://en.wikipedia.org/wiki/Session_layer) layer of an osmotic architecture where IoT devices are communicating directly to the cloud. Differently from other simulators, the IoT devices are not meant to be static, but are travelling within a Smart-City scenario. Furthermore, given the recently studied overlaps between the Cell-Free architecture and the Edge infrastructure for the Osmotic environment, we designed new packet routing algorithms outperforming previous solutions designed for Software Designed Networks over 5G infrastructures {% cite s25051591 %}. We are currently investigating on the possibility of tracking down patients on life-support and ambulances sending data within a Smart-City environment {% cite simbridger2 %}: this idea got us a [Best Paper Award](https://www.ehpwas.org/program.html) at eHPWAS'24.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/image_01_sumo.png" title="An external tool generates the vehicular traces." class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/image_02_osmotic.png" title="SimulatorBridger injects this information to generate communications patterns as soon as a vehicle approaches an Edge node." class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/image_03_osmotic.png" title="The Central Agent and Cell-Free CPU node orchestrates different Edge subnetworks so to balance the communication times." class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The principle of the simulator is quite easy: the simulator takes as a input the cars' position in time as an input (left), for then injecting such information to a specific agent within the simulation interacting with the communication infrastructure (center) governed by a central agent (right)  {% cite simbridger1 %}.
</div>

The simulator enables testing a network infrastructure over multiple osmotic configurations {% cite simbridger2 %}, several different packet routing algorithms {% cite ISC224 %}, while both considering both vehicular traces {% cite simbridger1 %} as well as simplistic trace counting {% cite icit %}.

Using the simulator is quite easy: all you have to do is to manually set the ```.yaml``` file configurations and use the simulator code directly from Java by using a code similar to the following:

{% raw %}

```java
class MyClass {
	public static void main(String[] args) {
           // TODO
	}
}
```

{% endraw %}


If you ar an unyielding Python simp, you can still invoke our simulator via [JPype](https://jpype.readthedocs.io/en/latest/).
