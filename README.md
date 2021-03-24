# hierarchical_nu

A Bayesian hierarchical model for source-nu associations.

## Installation

The package can currently be installed with:

```
pip install git+https://github.com/cescalara/hierarchical_nu.git
```

In future, hierachical_nu will also be available on PyPI!

The above command will go ahead and install any dependencies that you may be missing to run the core code. 

### Setting up Stan

The hierarchical model is implemented in [Stan](https://mc-stan.org), using the [CmdStan](https://github.com/stan-dev/cmdstan) and [CmdStanPy](https://github.com/stan-dev/cmdstanpy) interfaces. CmdStanPy will be installed as needed using pip if you follow the above instructions. However if you have not set up and compiled CmdStan before, the extra step detailed below is needed. See the CmdStanPy [installation docs](https://mc-stan.org/cmdstanpy/installation.html) for more information.


You can set up CmdStan by running the following python code:

```python
import cmdstanpy
cmdstanpy.install_cmdstan()
```

Or via the command line on MacOS/Linux:

```
install_cmdstan
```

This will make and install CmdStan in the `~/.cmdstan` directory.

## Examples

You can find some example notebooks stored as markdown files in the `examples/` directory. To run these notebooks, use the [jupytext](https://github.com/mwouts/jupytext) package to open the markdown files.

The first time that you use hierarchical_nu, some longer calculations will be run and cached locally. This is a one-time cost, so please be patient. 

More documentation coming soon!
