# Installation

Falcondale is a simple Python library one can import by the usual installation calls.

```
pip install falcondale
```

Even though pip is one of the most used environment management mechanisms, we encourage our readers to use [Poetry](https://python-poetry.org/) or [PDM](https://pdm.fming.dev/latest/) in order to count with a robust dependency solving mechanism. Falcondale installation will also install all additional dependencies needed, in particular those related to quantum computing frameworks and their compatible versions. This can make it a little bit heavy during installation of initial loading of the classes indicated below (depending on your machine mostly), so we ask you for a little bit of patient at this stage.

Once installed, simply invoke the objects that will help you build your QML models.

```py
from falcondale import Project, Model
```

More about the usage can be found in the following usage examples. But we would like to follow with some of the features that Falcondale counts with.
