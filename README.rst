
SimpleITK Registration
======================

An Interactive, Python-based Introduction to Registration the Insight Toolkit (ITK)
-----------------------------------------------------------------------------------

|MICCAI 2015|

2015 MICCAI Conference
~~~~~~~~~~~~~~~~~~~~~~

Brian Avants, University of Pennsylvania
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Hans Johnson, University of Iowa
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bradley Lowekamp, Medical Science & Computing and National Institutes of Health
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Matthew McCormick, Kitware Inc
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Nick Tustison, University of Virginia
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ziv Yaniv, TAJ Technologies Inc. and National Institutes of Health
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

|Circle CI|

.. |MICCAI 2015| image:: Data/MiccaiBanner.png
   :target: http://www.itk.org/Wiki/SimpleITK/Tutorials/MICCAI2015
.. |Circle CI| image:: https://circleci.com/gh/InsightSoftwareConsortium/SimpleITKTutorialMICCAI2015.svg?style=svg
   :target: https://circleci.com/gh/InsightSoftwareConsortium/SimpleITKTutorialMICCAI2015

Who Should Attend
-----------------

-  Do you want your students to gain practical experience with
   registration while minimizing their programming load?
-  Do you want to easily experiment with various ITK registration
   configurations, or optimize the settings of a specific registration
   configuration?

If you answered yes to either of these questions, then this tutorial is
for you.

|SimpleITK|

The goal of this half-day tutorial is to introduce students and
researchers to SimpleITK’s interface for the ITK version 4 registration
framework. SimpleITK is, as the name suggests, a simpler interface to
ITK. It provides a procedural interface and bindings to several
interpreted languages, facilitating fast experimentation with ITK
algorithms. In this tutorial we will use the Python programming language
and the IPython Notebook interactive environment to explore the various
features of the ITKv4 registration framework. Key features presented
include: uniform treatment of linear, deformable and composite
transformations, embedded multi-resolution registration and self
calibrating optimizers. Using a hands on approach, participants will
experiment with various registration tasks, learning how to use

.. |SimpleITK| image:: Data/SimpleITKLogo.png
   :target: http://www.simpleitk.org/

Program
-------

1. *08:30am:* `Setup and
   introduction <1_Setup_and_introduction.ipynb>`__ *Matthew McCormick
   and Hans Johnson*
2. *09:00am:* `SimpleITK basics <2_SimpleITK_basics.ipynb>`__: loading
   data, image access, image transformations, image resampling, basic
   filters *Matthew McCormick*
3. *10:00am:* The ITKv4 registration framework *Nick
   Tustison and Brian Avants*
4. *10:30am:* Coffee break.
5. *10:45am:* `Registration 1 <3_Registration_1.ipynb>`__: composite
   transform, transformation initialization, embedded multi-resolution,
   scale parameter estimation, optimization termination criteria *Ziv
   Yaniv*
6. *11:30am:* `Registration 2 <4_Registration_2.ipynb>`__: nonrigid
   registration, Bspline and displacement field transformations *Ziv
   Yaniv*
7. *12:30pm:* Lunch.
