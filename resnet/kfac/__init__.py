"""Top-level module for K-FAC."""
from __future__ import annotations

import kfac.assignment as assignment
import kfac.base_preconditioner as base_preconditioner
import kfac.distributed as distributed
import kfac.enums as enums
import kfac.gpt_neox as gpt_neox
import kfac.layers as layers
import kfac.preconditioner as preconditioner
import kfac.scheduler as scheduler
import kfac.tracing as tracing
import kfac.warnings as warnings

__version__ = '0.4.1'
