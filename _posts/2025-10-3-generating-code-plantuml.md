---
layout: post
title: Code generation from PlantUML
tags: xmi, plantuml, uml, java, code-generation
date: 2025-10-03
categories: sample-posts
tabs: true
description: This post provides a brief explanation on how to connect visualization tools to code generation, without using propietary software.
---

[PlantUML](https://plantuml.com/) syntax linearizes UML diagrams to a text file, similarly to [XMI](https://systemarchitect.info/sysml-mbse/different-types-of-xmi/): as for any tool supporting PlantUML as a UML representation, it should always be possible to generate code, as you are ultimately generating code from UML, independently of its serialisation format (UXF, XMI, etc.). The rationale behind it is to define a PlantUML parser which then generates classes and relationships according to the classes and relationships listed in the PlantUML file. Given this information, you can easily generate a class per instantiated class, as well as generate the fields as indicated by the relationships, while taking specific care of cardinality and aggregations, which should be ultimately represented as fields.

**XMI Solution**

Despite it should be possible to generate the UML directly from PlantUML, the only existing tool requires legacy NodeJS support and, during its operation with more recent versions of NodeJS, generates empty classes. An alternative solution is then to first generate an XMI file, for then ultimately generating the Java classes. This solution should be the less preferred one, as it currently seems that different tools support different XMI representations, [one conflicting with the other](https://plantuml.com/xmi). As far as I am concerned, ArgoUML was the tool better supporting PlantUML's XMI syntax. Thus, you can generate ArgoUML XMI from PlantUML as follows:

```
java -jar plantuml-1.2025.7.jar demofile.puml -txmi:argo
```

This will generate a `demofile.xmi` file. Next, you can import the XMI file into ArgoUML: this will generate a PlantUML model. At the time of the writing, this will generate an empty class diagram with some class and relationships provided separately. Notwithstanding the former, it should always be possible to drag the classes on the empty class diagram file, which should automatically import the relationships occurring among the classes. After doing this, you can select all the classes for then generating the corresponding Java code.
Please consider that ArgoUML does not corretly implement the difference between extension and implemnetation (interfaces), thus you might still generate some incorrect Java code, which should be fixed manually. 

**Legacy NodeJS Solution**

If you have access to a [legacy NodeJs version](https://github.com/jupe/puml2code/issues/59), you can try to use this tool: this is limited to class diagrams, though but in essence, is generating code depending on the aforementioned description: <https://github.com/jupe/puml2code>. Given a PlantUML file `demofile.puml`, you can then generate some java code as follows:

```
puml2code -i demofile.puml -l java
```
