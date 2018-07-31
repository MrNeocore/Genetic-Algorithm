# Travelling Salesman Problem (TSP) using Genetic Algorithms

## Context :
- Original developement date : 09/2017 -> Very old code considering how much I learned recently.
- Developped as part of an Biologically Inspired Computation coursework at HWU.
- Very limited exposure to some critical numerical libraries (i.e. NumPy / pandas)

## Features 
- Multiple mutation / cross-over operators 
- Entirely customizable : easily add new functions, change parameters etc
- Benchmark utility
- Live loss plot
- Profiled code (well, at least is it not the slowest raw Python code...)
  - Cythonized fitness function (haversine distance)

## Libraries
Pretty much only raw Python (yes, I loved for loops at the time)


## Feedback from lecturer :
- Grade : 83 / 100

## Overall (self-analysis 31/07/18)  : 
- Stability : MOK, remember having memory issues at some point (fixed)
- Readability  : Not too bad but not a lot of comments
- Code quality : Decent, if we ignore the fact that it is raw Python, hence very nested code.
- Performance  : NO, just no. At least I used a profiler and Cythonized the heaviest function, but not using NumPy is a big No No.
