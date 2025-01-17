---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

```{code-cell} ipython3
:tags: [remove-cell]

import sys
sys.path.append('../../../../')
```

(unit_operation_guide)=
# Unit Operation Models
A {class}`UnitOperation <CADETProcess.processModel.UnitBaseClass>` is a class that represents the physico-chemical behavior of an apparatus and holds its model parameters.
For an overview of all unit operation models currently available in **CADET-Process**, refer to {mod}`~CADETProcess.processModel`.
Each unit operation model can be associated with binding models that describe the interaction of components with the surface of a chromatographic stationary phase.
For more information about binding models, refer to {ref}`binding_models_guide`.
Moreover, `ReactionModels` can be used to model chemical reactions.
For more information, refer to {ref}`reaction_models_guide`.
To describe more complex operating modes, multiple unit operations can be connected in a {class}`~CADETProcess.processModel.FlowSheet`.
For more information about the {class}`~CADETProcess.processModel.FlowSheet` class, refer to {ref}`flow_sheet_guide`

To instantiate a unit operation model, the corresponding class needs to be imported from the {mod}`CADETProcess.processModel` module.
For example, to use a {class}`~CADETProcess.processModel.Cstr` use the following:

```{code-cell} ipython3
:tags: [hide-cell]

from CADETProcess.processModel import ComponentSystem
component_system = ComponentSystem(2)
```

```{code-cell} ipython3
from CADETProcess.processModel import Cstr
unit = Cstr(component_system, 'tank')
```

All parameters are stored in the {attr}`~CADETProcess.processModel.UnitBaseModel.parameters` attribute.
```{code-cell} ipython3
print(unit.parameters)
```

Note that some parameters might have default values.
To only show required parameters, inspect `required_parameters`.

```{code-cell} ipython3
print(unit.required_parameters)
```

## Polynomial Coeffients
Some parameters represent polynomial coefficients.
For example, the concentration of an {class}`~CADETProcess.processModel.Inlet` represents the coefficients of a cubic polynomial (in time).

```{code-cell} ipython3
:tags: [hide-cell]

from CADETProcess.processModel import Inlet
inlet = Inlet(component_system, 'inlet')
```

To specify a constant value for each component, use a list with length `n_comp`.

```{code-cell} ipython3
inlet.c = [1, 2]
```

To specify the polynomial coefficients, a list of lists needs to be set.
For each component, the coefficients are added in ascending order where the first entry is the constant term, the second is the linear term etc.
E.g. consider a gradient where the first component concentration has a constant term of $0~mM$ and increases linearly with slope $1 mM \cdot s^{-1}$, and the second component starts at $2~mM$ and decreases with a quadratic term of $-1~mM \cdot s^{-2}$.
Note, missing coefficients are always added internally.

```{code-cell} ipython3
inlet.c = [[0, 1], [2, 0, -1]]
print(inlet.c)
```

Similarly, the polynomial coefficients for the unit flow rate can be set for {class}`Inlets <CADETProcess.processModel.Inlet>` and {class}`Cstrs <CADETProcess.processModel.Cstr>`.
For example, a constant flow rate can be set with:

```{code-cell} ipython3
inlet.flow_rate = 1
print(inlet.flow_rate)
```

And a linearly increasing flow rate with:

```{code-cell} ipython3
inlet.flow_rate = [1,2]
print(inlet.flow_rate)
```

Since these parameters are mostly used in dynamic process models, they are usually modified using {class}`Events <CADETProcess.dynamicEvents.Event>`.


For an example, refer to {ref}`SSR process <lwe_example>`.

### Discretization
Some of the unit operations need to be spatially discretized.
The discretization parameters are stored in a {class}`DiscretizationParametersBase` class.
For example, consider a {class}`~CADETProcess.processModel.LumpedRateModelWithoutPores`.

```{code-cell} ipython3
:tags: [hide-cell]

from CADETProcess.processModel import LumpedRateModelWithoutPores
lrm = LumpedRateModelWithoutPores(component_system, 'lrm')
```

The discretization parameters can be imported and configured manually.
For example, to use the finite volume implementation (see {class}`~CADETProcess.processModel.LRMDiscretizationFV`) import the corresponding class and configure the parameters.

```{code-cell} ipython3
from CADETProcess.processModel import LRMDiscretizationFV

lrm_discretization_fv = LRMDiscretizationFV()
print(lrm_discretization_fv.parameters)
```

Notable parameters are:
- {attr}`~CADETProcess.processModel.LRMDiscretizationFV.ncol`: Number of axial column discretization cells. Default is 100.
- {attr}`~CADETProcess.processModel.LRMDiscretizationFV.weno_parameters`: Discretization parameters for the WENO scheme
- {attr}`~CADETProcess.processModel.LRMDiscretizationFV.consistency_solver`: Consistency solver parameters for Cadet.

Then, set the {attr}`~CADETProcess.processModel.LumpedRateModelWithoutPores.discretization` attribute.

```{code-cell} ipython3
lrm.discretization = lrm_discretization_fv
```

Note that by default, all models are already pre-configured with the finite volume discretization.
Hence, it is usually not necessary to manually import and set this attribute.

**CADET** also offers a discontinuous Galerkin (DG) discretization scheme.

```{note}
DG functionality is still work in progress.
For it to work, the DG-version of **CADET** needs to be compiled.
```

To use DG, import the corresponding {class}`~CADETProcess.processModel.LRMDiscretizationDG` class and set the attribute.
Alternatively, it is also possible to simply set `discretization_scheme='DG'` in the constructor.

```{code-cell} ipython3
lrm_dg = LumpedRateModelWithoutPores(component_system, 'lrm_dg', discretization_scheme='DG')
print(lrm_dg.discretization)
```

(solution_recorder_guide)=
## Solution Recorder
To store the solution of a unit operation, a solution recorder needs to be configured.
In this recorder, a flag can be set to store different partitions (e.g. inlet, outlet, bulk, etc.) of the solution of that unit operation.
Consider the {attr}`~CADETProcess.processModel.LumpedRateModelWithoutPores.solution_recorder` of a {class}`~CADETProcess.processModel.LumpedRateModelWithoutPores`.

```{code-cell} ipython3
print(lrm.solution_recorder)
```

By default, only the inlet and outlet of each unit are stored after simulation.
To also store the solution of bulk, particle liquid, or particle solid, configure the corresponding attributes.
For example, to store the particle solid phase solution, set the following:

```{code-cell} ipython3
lrm.solution_recorder.write_solution_solid = True
```

In the solution recorder, it can also be configured whether derivatives or sensitivities of that unit are to be stored.
For more information on the solution, refer to {ref}`simulation_results_guide`.
